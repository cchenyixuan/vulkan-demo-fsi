"""
simulator_v1.py — V1 SPH simulator, single GPU.

Structure mirrors utils/sph/simulator.py (V0) with the V1 additions that
survived the single-GPU cut:
  * set 0 (particle SoA, 10 bindings incl. density scratch + extension_fields)
  * set 1 (voxel cells: inside / incoming counts + index lists + base offset)
  * set 2 unused (kept as an empty placeholder layout so common.glsl's
    descriptor numbering is unchanged)
  * set 3 (global_status 20-uint block, overflow log, inlet template,
    dispatch indirect, material_parameters, defrag scratch counter)
  * defrag.comp with copy-back (scratch set 4 → set 0) on a fixed cadence

Pid layout is [1, POOL_SIZE] and voxel layout is [1, NX*NY*NZ]; slot 0 is the
dead / unallocated sentinel everywhere (see shaders/README.md). The ghost
pool / ghost voxel spec constants that common.glsl still declares are pinned
to 0, so every multi-GPU branch inside the kernels is dead-code-eliminated
and the layout collapses to the V0 one. The multi-GPU ghost / migration
drivers live on the v4-multigpu-orchestration branch, not here.

Per-step command buffer (leapfrog, 5 kernels):
    predict → update_voxel → correction → density → (scratch→primary copy)
    → force

Density staging: scratch+copy. density.comp writes `density_pressure_scratch`
(set 0 binding 2), then vkCmdCopyBuffer copies scratch → primary inside the
same step cmd so force.comp sees ρ_{n+1}, P_{n+1}.
"""

import math
import pathlib
import struct
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from vulkan import *
from vulkan._vulkancache import ffi

from utils.sph.case import Case, KIND_ROTOR
from utils.sph.vulkan_context import VulkanContext


# ============================================================================
# Module-level constants
# ============================================================================

SHADER_DIR = pathlib.Path(__file__).resolve().parents[1] / "shaders" / "spv"

# Hot kernels: same per-step / bootstrap usage as V0.
SHADER_NAMES_HOT = [
    "initialize_voxelization",
    "predict",
    "update_voxel",
    "build_neighbor_list",
    "correction",
    "density",
    "force",
    "bootstrap_half_kick",
]
SHADER_NAME_DEFRAG = "defrag"


# Spec const ids (mirrors experiment/v1/shaders/common.glsl id ranges).
# We build VkSpecializationInfo directly (V0's build_specialization_info is
# Case-driven and doesn't know about the V1 ghost ids).
SPEC_ID_SMOOTHING_LENGTH                    = 0
SPEC_ID_SPEED_OF_SOUND                      = 1
SPEC_ID_DELTA_COEFFICIENT                   = 2
# id 3 reserved
SPEC_ID_POWER_PARAMETER                     = 4
SPEC_ID_CFL_NUMBER                          = 5
SPEC_ID_TIMESTEP                            = 6
SPEC_ID_OWN_ORIGIN_X                        = 7
SPEC_ID_OWN_ORIGIN_Y                        = 8
SPEC_ID_OWN_ORIGIN_Z                        = 9
SPEC_ID_STRICT_BIT_EXACT                    = 10
SPEC_ID_GRID_DIMENSION_X                    = 11   # V1: EXTENDED nx
SPEC_ID_GRID_DIMENSION_Y                    = 12
SPEC_ID_GRID_DIMENSION_Z                    = 13
SPEC_ID_REGULARIZATION_XI                   = 14
SPEC_ID_REGULARIZATION_DET_THRESHOLD        = 15
SPEC_ID_REGULARIZATION_MAX_FROBENIUS        = 16
SPEC_ID_GRAVITY_X                           = 17
SPEC_ID_GRAVITY_Y                           = 18
SPEC_ID_GRAVITY_Z                           = 19
SPEC_ID_VOXEL_ORDER                         = 20
SPEC_ID_MICROPOLAR_THETA                    = 21
SPEC_ID_DIMENSION                           = 30
SPEC_ID_NEIGHBOR_Z_RANGE                    = 31
SPEC_ID_KERNEL_COEFFICIENT                  = 32
SPEC_ID_KERNEL_GRADIENT_COEFFICIENT         = 33
SPEC_ID_EPS_H_SQUARED                       = 40
SPEC_ID_PST_MAIN_SHIFT_COEFFICIENT          = 41
SPEC_ID_PST_ANTI_SHIFT_COEFFICIENT          = 42
SPEC_ID_USE_KCG_CORRECTION                  = 43
SPEC_ID_USE_DENSITY_DIFFUSION               = 44
SPEC_ID_USE_PST                             = 45
SPEC_ID_USE_PREFIX_SUM_DEFRAG               = 46
SPEC_ID_USE_NEIGHBOR_LIST                   = 47
SPEC_ID_MAX_PARTICLES_PER_VOXEL             = 50
SPEC_ID_WORKGROUP_SIZE                      = 51
SPEC_ID_MAX_INCOMING_PER_VOXEL              = 52
SPEC_ID_OWN_POOL_SIZE                       = 53
# Ghost pool / ghost voxel ids are still declared in common.glsl (V1
# merged-buffer layout). This branch is single-GPU only: they are pinned
# to 0 so every ghost branch in the kernels is dead-code-eliminated and
# the pid / voxel layout collapses to [1, POOL_SIZE] / [1, NX*NY*NZ].
SPEC_ID_LEADING_GHOST_POOL_SIZE             = 54
SPEC_ID_TRAILING_GHOST_POOL_SIZE            = 55
# Rotor (2026-09-25): axis + pivot of the prescribed rigid rotation. The
# time-dependent angle lives in the host-visible rotor_state buffer.
SPEC_ID_ROTOR_AXIS_X                        = 56
SPEC_ID_ROTOR_AXIS_Y                        = 57
SPEC_ID_ROTOR_AXIS_Z                        = 58
SPEC_ID_ROTOR_PIVOT_X                       = 59
SPEC_ID_ROTOR_PIVOT_Y                       = 60
SPEC_ID_ROTOR_PIVOT_Z                       = 61
SPEC_ID_MAX_NEIGHBORS                       = 62
SPEC_ID_LEADING_GHOST_VOXEL_COUNT           = 80
SPEC_ID_TRAILING_GHOST_VOXEL_COUNT          = 81


# ============================================================================
# Helpers / dataclasses
# ============================================================================


@dataclass
class Buffer:
    """One VkBuffer + VkDeviceMemory pair, plus byte size for debug."""
    handle: object
    memory: object
    size: int


@dataclass
class _BufferSpec:
    """Internal spec used by SphSimulatorV1._allocate_buffers."""
    name: str
    set_index: int
    binding: int
    size: int
    usage: int
    host_visible: bool = False      # True: HOST_VISIBLE | HOST_COHERENT, persistently mapped


# ============================================================================
# SphSimulatorV1
# ============================================================================


class SphSimulatorV1:
    """V1 SPH simulator — single GPU.

    Lifecycle:
        with VulkanContext.create() as ctx:
            sim = SphSimulatorV1(ctx, case)
            sim.bootstrap()
            sim.run_until(total_time=1.0)
            sim.destroy()
    """

    # Defrag set 4 mirrors set 0 except binding 2 (density_pressure_scratch
    # is transient — overwritten by next density.comp). Same convention as V0.
    DEFRAG_SET4_BINDINGS = (0, 1, 3, 4, 5, 6, 7, 8, 9)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, ctx: VulkanContext, case: Case):
        self.ctx = ctx
        self.case = case
        self._destroyed = False

        # ---- Grid dimensions + origin (straight from the case) --------------
        self.nx = int(case.grid["dimension"][0])
        self.ny = int(case.grid["dimension"][1])
        self.nz = int(case.grid["dimension"][2])
        case_origin = case.grid["origin"]
        self.own_origin_x = float(case_origin[0])
        self.own_origin_y = float(case_origin[1])
        self.own_origin_z = float(case_origin[2])

        # ---- Run state ------------------------------------------------------
        self.simulation_time = 0.0
        self.step_count = 0
        self.rotor_angle = 0.0          # theta written for the most recent step (rad)

        # ---- Pre-flight + setup pipeline -------------------------------------
        self._check_workgroup_limit()

        # Section 1: load shader modules (hot + defrag).
        self.shader_modules: dict = self._load_shader_modules()

        # Section 2 + 3: allocate buffers, then upload initial state.
        self.buffers: dict[str, Buffer] = self._allocate_buffers()
        self.scratch_buffers: dict[str, Buffer] = self._allocate_scratch_buffers()
        self._upload_initial_state()

        # Section 4: descriptor layouts + pool + sets, then wire buffers.
        self.descriptor_layouts: list = self._build_descriptor_layouts()
        self.defrag_set4_layout = self._build_defrag_set4_layout()
        self.descriptor_pool = self._create_descriptor_pool()
        self.descriptor_sets: dict = self._allocate_descriptor_sets()
        self.defrag_set4 = self._allocate_defrag_set4()
        self._wire_descriptor_sets()
        self._wire_defrag_set4()

        # Section 5: pipeline layout(s) + spec consts + pipelines.
        self.pipeline_layout = self._build_pipeline_layout()
        self.defrag_pipeline_layout = self._build_defrag_pipeline_layout()
        # Spec const data + entry list are kept alive on self for the lifetime
        # of all pipelines (Vulkan does NOT copy them).
        self._spec_keepalive: list = []
        self.spec_info_global = self._build_global_spec_info()
        self.pipelines: dict = self._build_compute_pipelines()
        self.defrag_pipeline = self._build_defrag_pipeline()

        # Section 6: pre-recorded command buffers. `step()` / `bootstrap()`
        # submit one combined cmd buffer per call — minimal fence overhead.
        self.bootstrap_cmd = self._record_bootstrap_cmd()
        self.step_cmd = self._record_step_cmd()
        self.defrag_cmd = self._record_defrag_cmd()

    # ==================================================================
    # Section 0: pre-flight
    # ==================================================================

    def _check_workgroup_limit(self) -> None:
        properties = vkGetPhysicalDeviceProperties(self.ctx.physical_device)
        limits = properties.limits
        requested = self.case.capacities.workgroup
        if requested > limits.maxComputeWorkGroupSize[0]:
            raise RuntimeError(
                f"workgroup={requested} exceeds maxComputeWorkGroupSize.x="
                f"{limits.maxComputeWorkGroupSize[0]}")
        if requested > limits.maxComputeWorkGroupInvocations:
            raise RuntimeError(
                f"workgroup={requested} exceeds maxComputeWorkGroupInvocations="
                f"{limits.maxComputeWorkGroupInvocations}")
        print(f"[SimV1] workgroup_size={requested} "
              f"(device max {limits.maxComputeWorkGroupSize[0]} / "
              f"{limits.maxComputeWorkGroupInvocations} invocations)")

    # ==================================================================
    # Section 1: SPV loading
    # ==================================================================

    def _load_shader_modules(self) -> dict:
        modules: dict = {}
        names = list(SHADER_NAMES_HOT) + [SHADER_NAME_DEFRAG]
        for name in names:
            spv_path = SHADER_DIR / f"{name}.comp.spv"
            if not spv_path.exists():
                raise FileNotFoundError(
                    f"compiled shader missing: {spv_path}. "
                    f"Compile experiment/v1/shaders/*.comp first.")
            spv_bytes = spv_path.read_bytes()
            create_info = VkShaderModuleCreateInfo(
                codeSize=len(spv_bytes),
                pCode=spv_bytes,
            )
            modules[name] = vkCreateShaderModule(self.ctx.device, create_info, None)
        print(f"[SimV1] loaded {len(modules)} shader modules from {SHADER_DIR}")
        return modules

    # ==================================================================
    # Section 2: Buffer allocation
    # ==================================================================

    def _build_buffer_specs(self) -> list[_BufferSpec]:
        case = self.case
        own_pool_size = int(case.capacities.pool_size)
        # Set 0: slot 0 unused + POOL_SIZE particles.
        pool_capacity = 1 + own_pool_size

        # Set 1: slot 0 unused + nx * ny * nz voxels.
        voxel_capacity = 1 + self.nx * self.ny * self.nz

        cap_inside = int(case.capacities.max_per_voxel)
        cap_incoming = int(case.capacities.max_incoming)
        cap_neighbors = int(case.capacities.max_neighbors) if case.numerics.use_neighbor_list else 1
        n_materials = len(case.materials)

        BSU = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        TRANSFER = (VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
        VERT = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT

        return [
            # ---- Set 0: particle SoA (10 bindings; binding 2 = scratch) ----
            _BufferSpec("position_voxel_id",            0, 0, 16 * pool_capacity, BSU | TRANSFER | VERT),
            _BufferSpec("density_pressure",             0, 1,  8 * pool_capacity, BSU | TRANSFER | VERT),
            _BufferSpec("density_pressure_scratch",     0, 2,  8 * pool_capacity, BSU | TRANSFER | VERT),
            _BufferSpec("velocity_mass",                0, 3, 16 * pool_capacity, BSU | TRANSFER | VERT),
            _BufferSpec("acceleration",                 0, 4, 16 * pool_capacity, BSU | TRANSFER),
            _BufferSpec("shift",                        0, 5, 16 * pool_capacity, BSU | TRANSFER),
            _BufferSpec("material",                     0, 6,  4 * pool_capacity, BSU | TRANSFER),
            _BufferSpec("correction_inverse",           0, 7, 32 * pool_capacity, BSU | TRANSFER),
            _BufferSpec("density_gradient_kernel_sum", 0, 8, 16 * pool_capacity, BSU | TRANSFER),
            _BufferSpec("extension_fields",             0, 9, 16 * pool_capacity, BSU | TRANSFER),

            # ---- Set 1: voxel cells -----------------------------------------
            _BufferSpec("inside_particle_count",        1, 0,  4 * voxel_capacity,                  BSU | TRANSFER),
            _BufferSpec("incoming_particle_count",      1, 1,  4 * voxel_capacity,                  BSU | TRANSFER),
            _BufferSpec("inside_particle_index",        1, 2,  4 * voxel_capacity * cap_inside,     BSU | TRANSFER),
            _BufferSpec("incoming_particle_index",      1, 3,  4 * voxel_capacity * cap_incoming,   BSU | TRANSFER),
            _BufferSpec("voxel_base_offset",            1, 4,  4 * voxel_capacity,                  BSU | TRANSFER),
            # Per-particle neighbour list (2026-09-25). neighbor_list is laid
            # out transposed, [k * pool_capacity + pid], so that at loop index
            # k a warp of consecutive pids reads one contiguous 128 B run.
            # Sized to 1 entry per particle when the list is disabled (the
            # kernels then never read it, but the binding must exist).
            _BufferSpec("neighbor_count",               1, 5,  4 * pool_capacity,                   BSU | TRANSFER),
            _BufferSpec("neighbor_list",                1, 6,  4 * pool_capacity * cap_neighbors,   BSU | TRANSFER),

            # ---- Set 2: UNUSED (no bindings) — placeholder layout
            # is built in _build_descriptor_layouts. No buffer specs.

            # ---- Set 3: global / transport / materials ---------------------
            _BufferSpec("global_status",                3, 0,  80,                       BSU | TRANSFER),
            _BufferSpec("overflow_log",                 3, 1,  64,                       BSU | TRANSFER),
            _BufferSpec("inlet_template",               3, 2,  32,                       BSU | TRANSFER),
            _BufferSpec("dispatch_indirect",            3, 3,  16,                       BSU | TRANSFER),
            _BufferSpec("ghost_out_packet",             3, 4,  16,                       BSU | TRANSFER),
            _BufferSpec("ghost_in_staging",             3, 5,  16,                       BSU | TRANSFER),
            _BufferSpec("diagnostic",                   3, 6,  16,                       BSU | TRANSFER),
            _BufferSpec("material_parameters",          3, 7,  48 * max(n_materials, 1), BSU | TRANSFER),
            _BufferSpec("defrag_scratch_counter",       3, 8,   4,                       BSU | TRANSFER),
            # Rotor state (cos, sin, omega, t): CPU-written every step, GPU-read
            # by predict.comp. Host-visible + coherent, mapped for the lifetime
            # of the simulator (see _rotor_write_state).
            _BufferSpec("rotor_state",                  3, 9,  16,                       BSU | TRANSFER, host_visible=True),
        ]

    def _allocate_buffer(self, size: int, usage: int, memory_properties: int) -> Buffer:
        if size == 0:
            raise ValueError("buffer size must be > 0")
        buffer_create_info = VkBufferCreateInfo(
            size=size,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        )
        handle = vkCreateBuffer(self.ctx.device, buffer_create_info, None)
        memory_requirements = vkGetBufferMemoryRequirements(self.ctx.device, handle)
        memory_type_index = self.ctx.find_memory_type(
            memory_requirements.memoryTypeBits, memory_properties)
        allocate_info = VkMemoryAllocateInfo(
            allocationSize=memory_requirements.size,
            memoryTypeIndex=memory_type_index,
        )
        memory = vkAllocateMemory(self.ctx.device, allocate_info, None)
        vkBindBufferMemory(self.ctx.device, handle, memory, 0)
        return Buffer(handle=handle, memory=memory, size=size)

    def _allocate_buffers(self) -> dict[str, Buffer]:
        specs = self._build_buffer_specs()
        buffers: dict[str, Buffer] = {}
        self._mapped: dict[str, object] = {}
        total_bytes = 0
        for spec in specs:
            if spec.host_visible:
                buffers[spec.name] = self._allocate_buffer(
                    size=spec.size,
                    usage=spec.usage,
                    memory_properties=(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                       | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
                )
                self._mapped[spec.name] = vkMapMemory(
                    self.ctx.device, buffers[spec.name].memory, 0, spec.size, 0)
            else:
                buffers[spec.name] = self._allocate_buffer(
                    size=spec.size,
                    usage=spec.usage,
                    memory_properties=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                )
            total_bytes += spec.size
        self._buffer_specs = specs
        print(f"[SimV1] allocated {len(buffers)} buffers, "
              f"{total_bytes / (1024*1024):.2f} MB total (device-local)")
        return buffers

    def _allocate_scratch_buffers(self) -> dict[str, Buffer]:
        usage = (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                 | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                 | VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
        scratch: dict[str, Buffer] = {}
        total_bytes = 0
        for spec in self._buffer_specs:
            if spec.set_index != 0:
                continue
            if spec.binding not in self.DEFRAG_SET4_BINDINGS:
                continue
            scratch[spec.name] = self._allocate_buffer(
                size=spec.size,
                usage=usage,
                memory_properties=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            )
            total_bytes += spec.size
        print(f"[SimV1] allocated {len(scratch)} defrag scratch buffers, "
              f"{total_bytes / (1024*1024):.2f} MB total")
        return scratch

    # ==================================================================
    # Section 3: Initial data upload
    # ==================================================================

    def own_first_pid(self) -> int:
        """Mirror helpers.glsl own_first_pid(): the pid range starts at 1
        (slot 0 is the dead / unallocated sentinel)."""
        return 1

    def own_last_pid(self) -> int:
        """Mirror helpers.glsl own_last_pid()."""
        return int(self.case.capacities.pool_size)

    def _build_initial_data(self) -> dict[str, bytes]:
        """Place particles starting at own_first_pid() (= 1; slot 0 unused)."""
        case = self.case
        own_pool_size = int(case.capacities.pool_size)
        pool_capacity = 1 + own_pool_size
        data: dict[str, bytes] = {}

        own_first = self.own_first_pid()

        # ---- positions (xyz, voxel_id_as_float = 0 initially) ---------------
        positions = np.zeros((pool_capacity, 4), dtype=np.float32)
        cursor = own_first
        for source in case.particle_sources:
            n = int(source.vertices.shape[0])
            positions[cursor:cursor + n, 0:3] = source.vertices
            cursor += n
        data["position_voxel_id"] = positions.tobytes()

        # ---- extension_fields.xyz = reference (initial) position --------------
        # Used by predict.comp's ROTOR branch (x_ref); harmless for other kinds.
        # Carried through defrag with the other set-0 fields.
        reference = positions.copy()
        reference[:, 3] = 0.0
        data["extension_fields"] = reference.tobytes()

        # ---- velocity_mass (initial_velocity, mass = ρ₀ * volume) -----------
        velocity_mass = np.zeros((pool_capacity, 4), dtype=np.float32)
        cursor = own_first
        for source in case.particle_sources:
            n = int(source.vertices.shape[0])
            material = case.materials[source.material_group_id]
            mass = material.rest_density * material.volume
            velocity_mass[cursor:cursor + n, 0] = material.initial_velocity[0]
            velocity_mass[cursor:cursor + n, 1] = material.initial_velocity[1]
            velocity_mass[cursor:cursor + n, 2] = material.initial_velocity[2]
            velocity_mass[cursor:cursor + n, 3] = mass
            cursor += n
        data["velocity_mass"] = velocity_mass.tobytes()

        # ---- density_pressure (ρ₀, 0) ---------------------------------------
        density_pressure = np.zeros((pool_capacity, 2), dtype=np.float32)
        cursor = own_first
        for source in case.particle_sources:
            n = int(source.vertices.shape[0])
            material = case.materials[source.material_group_id]
            density_pressure[cursor:cursor + n, 0] = material.rest_density
            cursor += n
        data["density_pressure"] = density_pressure.tobytes()

        # ---- material (group_id, 0 for unallocated slots) -------------------
        material_array = np.zeros(pool_capacity, dtype=np.uint32)
        cursor = own_first
        for source in case.particle_sources:
            n = int(source.vertices.shape[0])
            material_array[cursor:cursor + n] = source.material_group_id
            cursor += n
        data["material"] = material_array.tobytes()

        # ---- material_parameters (48 B per group) ---------------------------
        material_blob = bytearray()
        for material in case.materials:
            row = struct.pack(
                "I f f f f f f f f f I I",
                int(material.kind),
                float(material.rest_density),
                float(material.viscosity),
                float(material.eos_constant),
                float(material.smoothing_length),
                float(material.radius),
                float(material.volume),
                float(material.rotor_angular_velocity),
                float(material.viscosity_transfer),
                float(material.viscosity_rotation),
                int(material.reserved_material_0),
                int(material.reserved_material_1),
            )
            assert len(row) == 48
            material_blob.extend(row)
        if len(material_blob) == 0:
            material_blob = b"\x00" * 48
        data["material_parameters"] = bytes(material_blob)

        return data

    def _staging_upload(self, dest: Buffer, payload: bytes) -> None:
        if len(payload) > dest.size:
            raise ValueError(
                f"upload payload ({len(payload)} B) > dest buffer ({dest.size} B)")
        staging = self._allocate_buffer(
            size=len(payload),
            usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            memory_properties=(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                               | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
        )
        try:
            mapped = vkMapMemory(self.ctx.device, staging.memory, 0, len(payload), 0)
            mapped[:len(payload)] = payload
            vkUnmapMemory(self.ctx.device, staging.memory)

            cmd = self._allocate_oneshot_cmd()
            vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
                flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
            region = VkBufferCopy(srcOffset=0, dstOffset=0, size=len(payload))
            vkCmdCopyBuffer(cmd, staging.handle, dest.handle, 1, [region])
            vkEndCommandBuffer(cmd)
            self.ctx.submit_and_wait(cmd)
            vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])
        finally:
            vkDestroyBuffer(self.ctx.device, staging.handle, None)
            vkFreeMemory(self.ctx.device, staging.memory, None)

    def _zero_buffer(self, dest: Buffer) -> None:
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        vkCmdFillBuffer(cmd, dest.handle, 0, dest.size, 0)
        vkEndCommandBuffer(cmd)
        self.ctx.submit_and_wait(cmd)
        vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])

    def _upload_initial_state(self) -> None:
        initial_data = self._build_initial_data()
        for name, buffer in self.buffers.items():
            if name in self._mapped:
                continue                      # host-visible: written directly
            if name in initial_data:
                payload = initial_data[name]
                if len(payload) < buffer.size:
                    payload = payload + b"\x00" * (buffer.size - len(payload))
                self._staging_upload(buffer, payload)
            else:
                self._zero_buffer(buffer)
        print(f"[SimV1] uploaded initial state to {len(self.buffers)} buffers")
        self._rotor_write_state(0.0)

    # ==================================================================
    # Section 4: Descriptor sets
    # ==================================================================

    def _build_descriptor_layouts(self) -> list:
        """4 layouts: set 0/1/3 hold buffers; set 2 is empty (V1.0a unused)."""
        layouts = []
        for set_index in range(4):
            specs_in_set = [s for s in self._buffer_specs if s.set_index == set_index]
            bindings = [
                VkDescriptorSetLayoutBinding(
                    binding=spec.binding,
                    descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
                )
                for spec in specs_in_set
            ]
            create_info = VkDescriptorSetLayoutCreateInfo(
                bindingCount=len(bindings),
                pBindings=bindings if bindings else None,
            )
            layouts.append(vkCreateDescriptorSetLayout(self.ctx.device, create_info, None))
        return layouts

    def _build_defrag_set4_layout(self):
        bindings = [
            VkDescriptorSetLayoutBinding(
                binding=binding_index,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            )
            for binding_index in self.DEFRAG_SET4_BINDINGS
        ]
        return vkCreateDescriptorSetLayout(self.ctx.device,
            VkDescriptorSetLayoutCreateInfo(bindingCount=len(bindings), pBindings=bindings),
            None)

    def _create_descriptor_pool(self):
        per_set = {
            i: sum(1 for s in self._buffer_specs if s.set_index == i)
            for i in range(4)
        }
        # set 2 has zero bindings; allocate the descriptor set anyway (no buffer
        # writes against it). Pool size only counts STORAGE_BUFFER descriptors.
        total_descriptors = (per_set[0] + per_set[1] + per_set[2] + per_set[3]
                             + len(self.DEFRAG_SET4_BINDINGS))
        pool_size = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=max(total_descriptors, 1),
        )
        return vkCreateDescriptorPool(self.ctx.device,
            VkDescriptorPoolCreateInfo(maxSets=5, poolSizeCount=1, pPoolSizes=[pool_size]),
            None)

    def _allocate_descriptor_sets(self) -> dict:
        layouts = [self.descriptor_layouts[i] for i in range(4)]
        allocated = vkAllocateDescriptorSets(self.ctx.device,
            VkDescriptorSetAllocateInfo(
                descriptorPool=self.descriptor_pool,
                descriptorSetCount=len(layouts),
                pSetLayouts=layouts))
        return {
            "set_0": allocated[0],
            "set_1": allocated[1],
            "set_2": allocated[2],
            "set_3": allocated[3],
        }

    def _allocate_defrag_set4(self):
        return vkAllocateDescriptorSets(self.ctx.device,
            VkDescriptorSetAllocateInfo(
                descriptorPool=self.descriptor_pool,
                descriptorSetCount=1,
                pSetLayouts=[self.defrag_set4_layout]))[0]

    def _wire_defrag_set4(self) -> None:
        writes = []
        for spec in self._buffer_specs:
            if spec.set_index != 0:
                continue
            if spec.binding not in self.DEFRAG_SET4_BINDINGS:
                continue
            scratch = self.scratch_buffers[spec.name]
            writes.append(VkWriteDescriptorSet(
                dstSet=self.defrag_set4,
                dstBinding=spec.binding,
                dstArrayElement=0,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                pBufferInfo=[VkDescriptorBufferInfo(
                    buffer=scratch.handle, offset=0, range=scratch.size)],
            ))
        vkUpdateDescriptorSets(self.ctx.device, len(writes), writes, 0, None)

    def _wire_descriptor_sets(self) -> None:
        writes = []
        set_keys = ["set_0", "set_1", "set_2", "set_3"]
        for set_index, set_key in enumerate(set_keys):
            for spec in self._buffer_specs:
                if spec.set_index != set_index:
                    continue
                buffer = self.buffers[spec.name]
                writes.append(VkWriteDescriptorSet(
                    dstSet=self.descriptor_sets[set_key],
                    dstBinding=spec.binding,
                    dstArrayElement=0,
                    descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    pBufferInfo=[VkDescriptorBufferInfo(
                        buffer=buffer.handle, offset=0, range=buffer.size)],
                ))
        vkUpdateDescriptorSets(self.ctx.device, len(writes), writes, 0, None)

    # ==================================================================
    # Section 5: Pipelines
    # ==================================================================

    def _build_pipeline_layout(self):
        return vkCreatePipelineLayout(self.ctx.device,
            VkPipelineLayoutCreateInfo(
                setLayoutCount=len(self.descriptor_layouts),
                pSetLayouts=self.descriptor_layouts,
                pushConstantRangeCount=0,
                pPushConstantRanges=[]),
            None)

    def _build_defrag_pipeline_layout(self):
        layouts = self.descriptor_layouts + [self.defrag_set4_layout]
        return vkCreatePipelineLayout(self.ctx.device,
            VkPipelineLayoutCreateInfo(
                setLayoutCount=len(layouts),
                pSetLayouts=layouts,
                pushConstantRangeCount=0,
                pPushConstantRanges=[]),
            None)

    # ---- Spec const helpers --------------------------------------------------
    # We build VkSpecializationInfo manually (no Case-driven helper) because V1
    # adds spec const ids beyond V0's set. Each helper packs (id, value, type)
    # tuples into a binary blob + map entry list.

    def _pack_spec_blob(self, entries: list) -> tuple:
        """entries: list of (constant_id, value, fmt). fmt ∈ {'I', 'i', 'f', 'B'}.
        Returns (vk_specialization_info, keepalive_dict).
        """
        offsets = []
        blob = bytearray()
        map_entries = []
        for cid, value, fmt in entries:
            offset = len(blob)
            packed = struct.pack(fmt, value)
            blob.extend(packed)
            map_entries.append(VkSpecializationMapEntry(
                constantID=cid, offset=offset, size=len(packed)))
            offsets.append((cid, offset, len(packed)))
        cdata = ffi.new("uint8_t[]", bytes(blob))
        info = VkSpecializationInfo(
            mapEntryCount=len(map_entries),
            pMapEntries=map_entries,
            dataSize=len(blob),
            pData=cdata,
        )
        keepalive = {"blob": bytes(blob), "cdata": cdata, "entries": map_entries}
        self._spec_keepalive.append(keepalive)
        return info

    def _global_spec_entries(self) -> list:
        """Spec const values shared by ALL pipelines (field names mirror V0's
        utils/sph/case.py _SPEC_CONSTANT_MAPPING getter table)."""
        case = self.case
        physics = case.physics
        numerics = case.numerics
        capacities = case.capacities

        return [
            (SPEC_ID_SMOOTHING_LENGTH,             float(physics.h),                          'f'),
            (SPEC_ID_SPEED_OF_SOUND,               float(physics.speed_of_sound),             'f'),
            (SPEC_ID_DELTA_COEFFICIENT,            float(numerics.delta_coefficient),         'f'),
            (SPEC_ID_POWER_PARAMETER,              float(physics.power),                      'f'),
            (SPEC_ID_CFL_NUMBER,                   float(physics.cfl),                        'f'),
            (SPEC_ID_TIMESTEP,                     float(case.timestep),                      'f'),
            (SPEC_ID_OWN_ORIGIN_X,                 float(self.own_origin_x),                  'f'),
            (SPEC_ID_OWN_ORIGIN_Y,                 float(self.own_origin_y),                  'f'),
            (SPEC_ID_OWN_ORIGIN_Z,                 float(self.own_origin_z),                  'f'),
            # STRICT_BIT_EXACT only mattered for multi-GPU ghost integration; off.
            (SPEC_ID_STRICT_BIT_EXACT,             0,                                         'I'),
            (SPEC_ID_GRID_DIMENSION_X,             int(self.nx),                              'I'),
            (SPEC_ID_GRID_DIMENSION_Y,             int(self.ny),                              'I'),
            (SPEC_ID_GRID_DIMENSION_Z,             int(self.nz),                              'I'),
            (SPEC_ID_REGULARIZATION_XI,            float(numerics.regularization.xi),         'f'),
            (SPEC_ID_REGULARIZATION_DET_THRESHOLD, float(numerics.regularization.det_threshold),    'f'),
            (SPEC_ID_REGULARIZATION_MAX_FROBENIUS, float(numerics.regularization.frobenius_max),    'f'),
            (SPEC_ID_GRAVITY_X,                    float(physics.gravity[0]),                 'f'),
            (SPEC_ID_GRAVITY_Y,                    float(physics.gravity[1]),                 'f'),
            (SPEC_ID_GRAVITY_Z,                    float(physics.gravity[2]),                 'f'),
            (SPEC_ID_VOXEL_ORDER,                  0,                                         'I'),  # V0/V1: linear
            (SPEC_ID_MICROPOLAR_THETA,             2.0,                                       'f'),  # V0/V1: unused placeholder
            (SPEC_ID_DIMENSION,                    int(physics.dimension),                    'I'),
            (SPEC_ID_NEIGHBOR_Z_RANGE,             int(case.neighbor_z_range),                'I'),
            (SPEC_ID_KERNEL_COEFFICIENT,           float(case.kernel_coefficient),            'f'),
            (SPEC_ID_KERNEL_GRADIENT_COEFFICIENT,  float(case.kernel_gradient_coefficient),   'f'),
            (SPEC_ID_EPS_H_SQUARED,                float(case.eps_h_squared),                 'f'),
            (SPEC_ID_PST_MAIN_SHIFT_COEFFICIENT,   float(numerics.pst_main),                  'f'),
            (SPEC_ID_PST_ANTI_SHIFT_COEFFICIENT,   float(numerics.pst_anti),                  'f'),
            (SPEC_ID_USE_KCG_CORRECTION,           1 if numerics.use_kcg_correction else 0,   'I'),
            (SPEC_ID_USE_DENSITY_DIFFUSION,        1 if numerics.use_density_diffusion else 0, 'I'),
            (SPEC_ID_USE_PST,                      1 if numerics.use_pst else 0,              'I'),
            (SPEC_ID_USE_PREFIX_SUM_DEFRAG,        1 if numerics.use_prefix_sum_defrag else 0, 'I'),
            (SPEC_ID_USE_NEIGHBOR_LIST,            1 if numerics.use_neighbor_list else 0,    'I'),
            (SPEC_ID_MAX_PARTICLES_PER_VOXEL,      int(capacities.max_per_voxel),             'I'),
            (SPEC_ID_WORKGROUP_SIZE,               int(capacities.workgroup),                 'I'),
            (SPEC_ID_MAX_INCOMING_PER_VOXEL,       int(capacities.max_incoming),              'I'),
            (SPEC_ID_OWN_POOL_SIZE,                int(capacities.pool_size),                 'I'),
            # Single-GPU: no ghost pool / ghost voxels (see SPEC_ID_*_GHOST_* note).
            (SPEC_ID_LEADING_GHOST_POOL_SIZE,      0,                                         'I'),
            (SPEC_ID_TRAILING_GHOST_POOL_SIZE,     0,                                         'I'),
            # Rotor axis / pivot (defaults: z axis through the origin when no rotor block).
            (SPEC_ID_ROTOR_AXIS_X,                 float(case.rotor.axis[0]  if case.rotor else 0.0), 'f'),
            (SPEC_ID_ROTOR_AXIS_Y,                 float(case.rotor.axis[1]  if case.rotor else 0.0), 'f'),
            (SPEC_ID_ROTOR_AXIS_Z,                 float(case.rotor.axis[2]  if case.rotor else 1.0), 'f'),
            (SPEC_ID_ROTOR_PIVOT_X,                float(case.rotor.pivot[0] if case.rotor else 0.0), 'f'),
            (SPEC_ID_ROTOR_PIVOT_Y,                float(case.rotor.pivot[1] if case.rotor else 0.0), 'f'),
            (SPEC_ID_ROTOR_PIVOT_Z,                float(case.rotor.pivot[2] if case.rotor else 0.0), 'f'),
            (SPEC_ID_MAX_NEIGHBORS,                int(capacities.max_neighbors),             'I'),
            (SPEC_ID_LEADING_GHOST_VOXEL_COUNT,    0,                                         'I'),
            (SPEC_ID_TRAILING_GHOST_VOXEL_COUNT,   0,                                         'I'),
        ]

    def _build_global_spec_info(self):
        """Spec const for non-ghost pipelines (hot kernels + defrag)."""
        return self._pack_spec_blob(self._global_spec_entries())

    def _build_compute_pipelines(self) -> dict:
        """Hot kernels (all share the global spec consts)."""
        pipelines: dict = {}

        # ---- Hot kernels (use global spec consts) --------------------------
        # The stage structs are copied BY VALUE into each create info, but the
        # C string behind pName="main" is owned by the Python stage object. If
        # that object dies before vkCreateComputePipelines runs, the driver
        # reads a dangling entry-point pointer and fails with VK_ERROR_UNKNOWN
        # for a random subset of the kernels (seen on RTX 4070 Ti SUPER,
        # driver 616.92, python-vulkan 1.3.275.1). Keep every stage alive
        # until the call returns.
        create_infos = []
        stage_keepalive = []
        for name in SHADER_NAMES_HOT:
            stage = VkPipelineShaderStageCreateInfo(
                stage=VK_SHADER_STAGE_COMPUTE_BIT,
                module=self.shader_modules[name],
                pName="main",
                pSpecializationInfo=self.spec_info_global,
            )
            stage_keepalive.append(stage)
            create_infos.append(VkComputePipelineCreateInfo(
                stage=stage, layout=self.pipeline_layout))
        result = vkCreateComputePipelines(
            self.ctx.device, VK_NULL_HANDLE,
            len(create_infos), create_infos, None)
        pipelines.update(zip(SHADER_NAMES_HOT, result))

        return pipelines

    def _build_defrag_pipeline(self):
        stage = VkPipelineShaderStageCreateInfo(
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=self.shader_modules[SHADER_NAME_DEFRAG],
            pName="main",
            pSpecializationInfo=self.spec_info_global,
        )
        return vkCreateComputePipelines(
            self.ctx.device, VK_NULL_HANDLE, 1, [
                VkComputePipelineCreateInfo(stage=stage, layout=self.defrag_pipeline_layout),
            ], None)[0]

    # ==================================================================
    # Section 6: Command buffer recording
    # ==================================================================

    def _allocate_oneshot_cmd(self):
        return vkAllocateCommandBuffers(self.ctx.device,
            VkCommandBufferAllocateInfo(
                commandPool=self.ctx.command_pool,
                level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1))[0]

    def _record_compute_barrier(self, cmd) -> None:
        barrier = VkMemoryBarrier(
            sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        )
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, [barrier], 0, None, 0, None)

    def _record_density_scratch_to_primary_copy(self, cmd) -> None:
        compute_to_transfer = VkMemoryBarrier(
            sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT,
        )
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, [compute_to_transfer], 0, None, 0, None)
        # Copy the particle pid range [own_first_pid, own_last_pid]; slot 0 is
        # the unused sentinel and is left untouched.
        primary = self.buffers["density_pressure"]
        scratch = self.buffers["density_pressure_scratch"]
        density_stride = 8
        own_first = self.own_first_pid()
        own_pool = int(self.case.capacities.pool_size)
        own_byte_offset = own_first * density_stride
        own_byte_size = own_pool * density_stride
        vkCmdCopyBuffer(cmd, scratch.handle, primary.handle, 1,
            [VkBufferCopy(srcOffset=own_byte_offset,
                          dstOffset=own_byte_offset,
                          size=own_byte_size)])
        transfer_to_compute = VkMemoryBarrier(
            sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
            dstAccessMask=VK_ACCESS_SHADER_READ_BIT,
        )
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, [transfer_to_compute], 0, None, 0, None)

    def _bind_pipeline_and_sets(self, cmd, pipeline_key: str, *, defrag: bool = False) -> None:
        pipeline = self.defrag_pipeline if defrag else self.pipelines[pipeline_key]
        layout = self.defrag_pipeline_layout if defrag else self.pipeline_layout
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
        sets = [
            self.descriptor_sets["set_0"],
            self.descriptor_sets["set_1"],
            self.descriptor_sets["set_2"],
            self.descriptor_sets["set_3"],
        ]
        if defrag:
            sets = sets + [self.defrag_set4]
        vkCmdBindDescriptorSets(cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE, layout,
            0, len(sets), sets, 0, None)

    # ---- Dispatch counts -----------------------------------------------------

    def _per_own_particle_dispatch_count(self) -> int:
        """Hot kernels iterate own pid range only (own_first_pid .. own_last_pid)."""
        wg = int(self.case.capacities.workgroup)
        own = int(self.case.capacities.pool_size)
        return (own + wg - 1) // wg

    def _per_voxel_dispatch_count(self) -> int:
        """Per-voxel kernels (update_voxel, defrag) iterate the whole grid."""
        wg = int(self.case.capacities.workgroup)
        voxel_count = self.nx * self.ny * self.nz
        return (voxel_count + wg - 1) // wg

    # ---- Step / bootstrap / defrag --------------------------------------------

    def _record_bootstrap_cmd(self):
        """initialize_voxelization → correction → density → density_copy →
        force → bootstrap_half_kick."""
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
        self._record_compute_barrier(cmd)

        per_p = self._per_own_particle_dispatch_count()

        self._bind_pipeline_and_sets(cmd, "initialize_voxelization")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_compute_barrier(cmd)

        self._record_build_neighbor_list(cmd, per_p)

        self._bind_pipeline_and_sets(cmd, "correction")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_compute_barrier(cmd)

        self._bind_pipeline_and_sets(cmd, "density")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_density_scratch_to_primary_copy(cmd)

        self._bind_pipeline_and_sets(cmd, "force")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_compute_barrier(cmd)

        self._bind_pipeline_and_sets(cmd, "bootstrap_half_kick")
        vkCmdDispatch(cmd, per_p, 1, 1)

        vkEndCommandBuffer(cmd)
        return cmd

    # ---- Per-step cmd buffer ------------------------------------------------

    def _record_step_cmd(self):
        """predict → update_voxel → correction → density → density_copy → force."""
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
        self._record_compute_barrier(cmd)

        per_p = self._per_own_particle_dispatch_count()
        per_v = self._per_voxel_dispatch_count()

        self._bind_pipeline_and_sets(cmd, "predict")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_compute_barrier(cmd)

        self._bind_pipeline_and_sets(cmd, "update_voxel")
        vkCmdDispatch(cmd, per_v, 1, 1)
        self._record_compute_barrier(cmd)

        self._record_build_neighbor_list(cmd, per_p)

        self._bind_pipeline_and_sets(cmd, "correction")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_compute_barrier(cmd)

        self._bind_pipeline_and_sets(cmd, "density")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_density_scratch_to_primary_copy(cmd)

        self._bind_pipeline_and_sets(cmd, "force")
        vkCmdDispatch(cmd, per_p, 1, 1)

        vkEndCommandBuffer(cmd)
        return cmd

    def _record_build_neighbor_list(self, cmd, per_p: int) -> None:
        """Neighbour-list build (2026-09-25): one dispatch per own particle
        right after the voxel lists are final (update_voxel in a step,
        initialize_voxelization in bootstrap). No-op when the case runs the
        legacy 27-voxel scan (numerics.use_neighbor_list = false). Defrag
        renumbers particles, but it runs after force and the next step
        rebuilds the list before anything reads it, so nothing is carried."""
        if not self.case.numerics.use_neighbor_list:
            return
        self._bind_pipeline_and_sets(cmd, "build_neighbor_list")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_compute_barrier(cmd)

    def _record_defrag_cmd(self):
        """Defrag with copy-back (scratch set 4 → set 0) + alive count refresh."""
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
        self._record_compute_barrier(cmd)

        # 1. Reset defrag_scratch_counter (path B atomic counter).
        scratch_counter = self.buffers["defrag_scratch_counter"]
        vkCmdFillBuffer(cmd, scratch_counter.handle, 0, scratch_counter.size, 0)

        # transfer (fill) → compute (defrag.comp atomic read/write)
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, [VkMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
                dstAccessMask=VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)],
            0, None, 0, None)

        # 2. Defrag dispatch (per voxel).
        self._bind_pipeline_and_sets(cmd, "defrag", defrag=True)
        vkCmdDispatch(cmd, self._per_voxel_dispatch_count(), 1, 1)

        # compute → transfer (read scratch for copy-back)
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, [VkMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT)],
            0, None, 0, None)

        # 3. Copy each scratch SoA back to set 0 primary.
        for spec in self._buffer_specs:
            if spec.set_index != 0:
                continue
            if spec.binding not in self.DEFRAG_SET4_BINDINGS:
                continue
            scratch = self.scratch_buffers[spec.name]
            primary = self.buffers[spec.name]
            vkCmdCopyBuffer(cmd, scratch.handle, primary.handle, 1,
                [VkBufferCopy(srcOffset=0, dstOffset=0, size=spec.size)])

        # 4. Refresh alive_particle_count from defrag_scratch_counter.
        # GlobalStatusBuffer V1 layout: alive_particle_count at offset 0 (uint).
        # defrag_scratch_counter is its own 4 B buffer; copy it into global_status[0..4).
        vkCmdCopyBuffer(cmd,
            scratch_counter.handle,
            self.buffers["global_status"].handle, 1,
            [VkBufferCopy(srcOffset=0, dstOffset=0, size=4)])

        # 5. transfer (copy-back) → compute (next step's predict reads set 0).
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, [VkMemoryBarrier(
                sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
                dstAccessMask=VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)],
            0, None, 0, None)

        vkEndCommandBuffer(cmd)
        return cmd

    # ==================================================================
    # Section 7: Public run API
    # ==================================================================

    def bootstrap(self) -> None:
        self.ctx.submit_and_wait(self.bootstrap_cmd)
        status = self.readback_global_status()
        if status["overflow_inside_count"] != 0 or status["overflow_incoming_count"] != 0:
            raise RuntimeError(
                f"GPU initial voxelization overflowed: "
                f"inside={status['overflow_inside_count']} "
                f"incoming={status['overflow_incoming_count']} "
                f"first_inside_vid={status['first_overflow_voxel_inside']}")
        if status["overflow_neighbor_count"] != 0:
            raise RuntimeError(
                f"neighbour list overflowed at bootstrap: "
                f"{status['overflow_neighbor_count']} particles have more than "
                f"capacities.max_neighbors={self.case.capacities.max_neighbors} neighbours "
                f"(first pid {status['first_overflow_neighbor_pid']})")
        print(f"[SimV1] bootstrap ok: alive={status['alive_particle_count']}"
              + (f" max_neighbors={self.case.capacities.max_neighbors}"
                 if self.case.numerics.use_neighbor_list else " (27-voxel scan)"))

        if self.case.numerics.defrag_enabled:
            self.ctx.submit_and_wait(self.defrag_cmd)
            print(f"[SimV1] init defrag complete "
                  f"(cadence={self.case.numerics.defrag_cadence})")

    # ==================================================================
    # Rotor: prescribed rigid rotation (2026-09-25)
    # ==================================================================

    def rotor_angle_and_rate(self, time: float) -> tuple[float, float]:
        """(theta, omega) at absolute time `time` for the case's rotor,
        computed in float64. omega ramps linearly from 0 over ramp_time (if
        > 0) to the signed material value; theta is its exact integral:
            t <  T:  omega = w t / T,       theta = w t^2 / (2 T)
            t >= T:  omega = w,             theta = w (t - T / 2)
        Returns (0, 0) when the case has no rotor."""
        if self.case.rotor is None:
            return 0.0, 0.0
        w = float(self.case.rotor_angular_velocity)
        T = float(self.case.rotor.ramp_time)
        if T <= 0.0 or time >= T:
            return w * (time - 0.5 * T), w
        return w * time * time / (2.0 * T), w * time / T

    def _rotor_write_state(self, time_next: float) -> None:
        """Write (cos theta, sin theta, omega, t) for t_{n+1} into the
        host-visible rotor_state buffer read by predict.comp. Must be called
        only when no submitted step is still executing (step(wait=True)
        guarantees that; step(wait=False) waits for the queue first)."""
        if "rotor_state" not in self._mapped:
            return
        theta, omega = self.rotor_angle_and_rate(time_next)
        payload = struct.pack("ffff", math.cos(theta), math.sin(theta), omega, time_next)
        self._mapped["rotor_state"][:16] = payload
        self.rotor_angle = theta

    def step(self, *, wait: bool = True) -> None:
        cmd = self.step_cmd
        if self.case.rotor is not None:
            if not wait:
                # predict.comp of the previous (unwaited) step may still be
                # reading rotor_state; finish it before overwriting.
                vkQueueWaitIdle(self.ctx.compute_queue)
            self._rotor_write_state(self.simulation_time + self.case.timestep)
        if wait:
            self.ctx.submit_and_wait(cmd)
        else:
            vkQueueSubmit(self.ctx.compute_queue, 1,
                VkSubmitInfo(
                    sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    commandBufferCount=1,
                    pCommandBuffers=[cmd]),
                VK_NULL_HANDLE)
        self.simulation_time += self.case.timestep
        self.step_count += 1

        numerics = self.case.numerics
        if (numerics.defrag_enabled
                and self.step_count > 0
                and self.step_count % numerics.defrag_cadence == 0):
            if wait:
                self.ctx.submit_and_wait(self.defrag_cmd)
            else:
                vkQueueSubmit(self.ctx.compute_queue, 1,
                    VkSubmitInfo(
                        sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        commandBufferCount=1,
                        pCommandBuffers=[self.defrag_cmd]),
                    VK_NULL_HANDLE)

    def run_until(
        self,
        total_time: Optional[float] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        time_budget = self.case.time
        while True:
            if total_time is not None and self.simulation_time >= total_time:
                return
            if max_steps is not None and self.step_count >= max_steps:
                return
            if time_budget.is_time_exceeded(self.simulation_time):
                return
            if time_budget.is_step_exceeded(self.step_count):
                return
            self.step()

    # ==================================================================
    # Section 8: Readback + render hook
    # ==================================================================

    def _readback_buffer(self, buffer: Buffer) -> bytes:
        staging = self._allocate_buffer(
            size=buffer.size,
            usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            memory_properties=(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                               | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
        try:
            cmd = self._allocate_oneshot_cmd()
            vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
                flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
            vkCmdCopyBuffer(cmd, buffer.handle, staging.handle, 1,
                [VkBufferCopy(srcOffset=0, dstOffset=0, size=buffer.size)])
            vkEndCommandBuffer(cmd)
            self.ctx.submit_and_wait(cmd)
            vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])
            mapped = vkMapMemory(self.ctx.device, staging.memory, 0, buffer.size, 0)
            data = bytes(mapped[0:buffer.size])
            vkUnmapMemory(self.ctx.device, staging.memory)
            return data
        finally:
            vkDestroyBuffer(self.ctx.device, staging.handle, None)
            vkFreeMemory(self.ctx.device, staging.memory, None)

    def readback_global_status(self) -> dict:
        """V1 GlobalStatusBuffer layout (20 × 4 B = 80 B). Field order matches
        experiment/v1/shaders/common.glsl. The ghost_* / migration_* fields
        are always 0 on this single-GPU branch."""
        raw = self._readback_buffer(self.buffers["global_status"])
        unpacked = struct.unpack("I f I I I I I I I I I I I I I I I I I I", raw)
        return {
            "alive_particle_count":          unpacked[0],
            "maximum_velocity":              unpacked[1],
            "overflow_inside_count":         unpacked[2],
            "overflow_incoming_count":       unpacked[3],
            "first_overflow_voxel_inside":   unpacked[4],
            "first_overflow_voxel_incoming": unpacked[5],
            "correction_fallback_count":     unpacked[6],
            "overflow_ghost_count":          unpacked[7],
            "ghost_send_leading_count":      unpacked[8],
            "ghost_send_trailing_count":     unpacked[9],
            "ghost_recv_leading_count":      unpacked[10],
            "ghost_recv_trailing_count":     unpacked[11],
            "migration_install_count":       unpacked[12],
            "overflow_install_tail":         unpacked[13],
            "overflow_install_inside":       unpacked[14],
            "first_overflow_voxel_install":  unpacked[15],
            # Neighbour-list build (2026-09-25): particles whose true
            # neighbour count exceeded MAX_NEIGHBORS (extra neighbours dropped).
            "overflow_neighbor_count":       unpacked[16],
            "first_overflow_neighbor_pid":   unpacked[17],
        }

    def readback_positions(self) -> np.ndarray:
        raw = self._readback_buffer(self.buffers["position_voxel_id"])
        return np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)

    def readback_velocity_mass(self) -> np.ndarray:
        raw = self._readback_buffer(self.buffers["velocity_mass"])
        return np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)

    def readback_acceleration(self) -> np.ndarray:
        raw = self._readback_buffer(self.buffers["acceleration"])
        return np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)

    def readback_material(self) -> np.ndarray:
        raw = self._readback_buffer(self.buffers["material"])
        return np.frombuffer(raw, dtype=np.uint32)

    def readback_extension_fields(self) -> np.ndarray:
        raw = self._readback_buffer(self.buffers["extension_fields"])
        return np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)

    def live_slot_mask(self, positions: np.ndarray) -> np.ndarray:
        """Boolean mask of the slots that hold live particles.

        Defrag packs the live particles into slots [1, alive_particle_count]
        and only that range is copied back; slots beyond the packed tail keep
        STALE copies from before the defrag (voxel_id != 0, not referenced by
        any voxel list, invisible to the kernels). A plain `voxel_id > 0` test
        therefore over-counts after any particle has been killed. Between two
        defrags no particle is created, so live == (slot <= alive count at the
        last defrag) and (voxel_id > 0). Found 2026-09-25 during the power-
        number campaign (stale rotor copies contaminated the torque sum once
        a run had lost particles)."""
        alive_count = int(self.readback_global_status()["alive_particle_count"])
        mask = np.zeros(positions.shape[0], dtype=bool)
        mask[1:alive_count + 1] = True
        mask &= positions[:, 3] > 0
        return mask

    def rotor_group_ids(self) -> list[int]:
        return [m.group_id for m in self.case.materials if m.kind == KIND_ROTOR]

    def readback_rotor_torque(self, split_height: Optional[float] = None,
                              shaft_radius: Optional[float] = None) -> dict:
        """Hydrodynamic force and torque on the rotor from the last force.comp.

        force.comp writes, for every ROTOR particle, its acceleration from the
        pressure + viscous interaction with all neighbours plus gravity (the
        PST shift is a separate buffer and not part of it). The fluid force on
        particle i is therefore f_i = m_i (a_i - g); the rigid body's force is
        F = sum_i f_i and its torque about the rotor axis is
            tau = axis . sum_i (x_i - pivot) x f_i.
        Rotor-rotor pair forces are antisymmetric and cancel in both sums.
        Returns force (3,), torque (3,), torque_axis (scalar), the rotor
        particle count, the rotor angle, time and step.

        Per-impeller split (2026-09-26): when split_height and shaft_radius
        are given, rotor particles are classified by their distance r from
        the rotor axis and their height h along it (both measured from the
        pivot): r < shaft_radius -> "shaft", otherwise h < split_height ->
        "lower" impeller, h >= split_height -> "upper" impeller. The result
        then also holds torque_axis_lower / _upper / _shaft and the three
        particle counts; the three torques sum to torque_axis exactly."""
        if self.case.rotor is None:
            raise RuntimeError("case has no rotor")
        groups = np.asarray(self.rotor_group_ids(), dtype=np.uint32)
        material = self.readback_material()
        is_rotor = np.isin(material, groups)
        is_rotor[0] = False
        positions = self.readback_positions()
        alive = self.live_slot_mask(positions)
        sel = is_rotor & alive
        x = positions[sel, :3].astype(np.float64)
        a = self.readback_acceleration()[sel, :3].astype(np.float64)
        m = self.readback_velocity_mass()[sel, 3].astype(np.float64)
        g = np.asarray(self.case.physics.gravity, dtype=np.float64)
        force_per_particle = (a - g) * m[:, None]
        axis = np.asarray(self.case.rotor.axis, dtype=np.float64)
        pivot = np.asarray(self.case.rotor.pivot, dtype=np.float64)
        arm = x - pivot
        torque_per_particle = np.cross(arm, force_per_particle)
        torque = torque_per_particle.sum(axis=0)
        result = {
            "force": force_per_particle.sum(axis=0),
            "torque": torque,
            "torque_axis": float(np.dot(torque, axis)),
            "rotor_particle_count": int(sel.sum()),
            "rotor_angle": float(self.rotor_angle),
            "time": float(self.simulation_time),
            "step": int(self.step_count),
        }
        if split_height is not None and shaft_radius is not None:
            height = arm @ axis
            radial = np.linalg.norm(arm - np.outer(height, axis), axis=1)
            axial_torque = torque_per_particle @ axis
            is_shaft = radial < shaft_radius
            is_lower = ~is_shaft & (height < split_height)
            is_upper = ~is_shaft & (height >= split_height)
            for name, mask in (("lower", is_lower), ("upper", is_upper), ("shaft", is_shaft)):
                result[f"torque_axis_{name}"] = float(axial_torque[mask].sum())
                result[f"rotor_particle_count_{name}"] = int(mask.sum())
        return result

    def get_render_buffers(self) -> dict:
        return {
            "position_voxel_id": self.buffers["position_voxel_id"].handle,
            "velocity_mass":     self.buffers["velocity_mass"].handle,
            "density_pressure":  self.buffers["density_pressure"].handle,
            "global_status":     self.buffers["global_status"].handle,
        }

    # ==================================================================
    # Section 9: Cleanup
    # ==================================================================

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        device = self.ctx.device

        # Pre-recorded cmds are freed when the pool is destroyed (ctx owns it).

        # Pipelines
        for pipeline in self.pipelines.values():
            vkDestroyPipeline(device, pipeline, None)
        if hasattr(self, "defrag_pipeline"):
            vkDestroyPipeline(device, self.defrag_pipeline, None)
        vkDestroyPipelineLayout(device, self.pipeline_layout, None)
        vkDestroyPipelineLayout(device, self.defrag_pipeline_layout, None)

        # Descriptor sets / pool / layouts
        vkDestroyDescriptorPool(device, self.descriptor_pool, None)
        for layout in self.descriptor_layouts:
            vkDestroyDescriptorSetLayout(device, layout, None)
        vkDestroyDescriptorSetLayout(device, self.defrag_set4_layout, None)

        # Buffers (unmap the persistently mapped ones first)
        for name in list(self._mapped):
            vkUnmapMemory(device, self.buffers[name].memory)
        self._mapped.clear()
        for buffer in self.buffers.values():
            vkDestroyBuffer(device, buffer.handle, None)
            vkFreeMemory(device, buffer.memory, None)
        for buffer in self.scratch_buffers.values():
            vkDestroyBuffer(device, buffer.handle, None)
            vkFreeMemory(device, buffer.memory, None)

        # Shaders
        for module in self.shader_modules.values():
            vkDestroyShaderModule(device, module, None)

    def __enter__(self) -> "SphSimulatorV1":
        return self

    def __exit__(self, *_) -> None:
        self.destroy()
