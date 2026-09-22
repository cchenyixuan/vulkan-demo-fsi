"""
renderer_v1.py — Vulkan + GLFW point-sprite renderer for SphSimulatorV1.

Draws the simulator's pid range [own_first_pid, own_last_pid] by passing
firstVertex = own_first_pid - 1 to vkCmdDraw (single-GPU: own_first_pid = 1,
so the whole pool is drawn). Owns its render shaders
(experiment/v1/shaders/render/) so color modes (e.g. vorticity) can evolve
together with the compute kernels.

Attaches to an SphSimulatorV1 without modifying it.
Reads particle SSBOs by binding them as descriptor inputs to a graphics
pipeline; particles are drawn as round point sprites.

Hotkeys:
    SPACE       pause/resume the simulation
    0-4         color mode: 0=speed, 1=accel, 2=density dev, 3=voxel id, 4=kernel_sum
    P           toggle perspective ↔ orthogonal projection
    F           re-frame to fit the case bbox
    +/-         steps_per_frame ± 1
    [/]         steps_per_frame /÷ 2
    ESC         quit

Mouse:
    left drag   orbit (rotate around lookat)
    middle drag pan
    scroll      zoom

V0 sync model: render and compute share one queue (graphics+compute family).
``simulator.step()`` waits its compute fence, then the renderer records and
submits a graphics frame. No async overlap.
"""

import ctypes
import pathlib
import time
from typing import Optional

import glfw
import numpy as np
from vulkan import *
from vulkan._vulkancache import ffi

from utils.sph.camera import Camera
from utils.sph.vulkan_context import VulkanContext

from experiment.v1.utils.simulator_v1 import SphSimulatorV1


# ============================================================================
# Constants
# ============================================================================

# V1 owns its render SPV (experiment/v1/shaders/render/particle.{vert,frag}) so
# it can add the vorticity color mode without perturbing V0's shared shader.
# parents[1] from experiment/v1/utils/renderer_v1.py = experiment/v1/.
SHADER_RENDER_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "shaders" / "spv" / "render"
)
MAX_FRAMES_IN_FLIGHT = 2

# Push constant block — must match experiment/v1/shaders/render/particle.vert.
# Layout (96 B, std430 / push_constant):
#   mat4  view_proj                64 B
#   uint  color_mode                4 B
#   float velocity_scale            4 B
#   float acceleration_scale        4 B
#   float density_deviation_scale   4 B
#   float rest_density              4 B
#   float point_size                4 B
#   float kernel_sum_scale          4 B
#   float vorticity_scale           4 B   ω_z normalizer for color mode 5
PUSH_CONSTANT_SIZE = 64 + 4 * 8


# ============================================================================
# SphRenderer
# ============================================================================


class SphRendererV1:
    """Point-sprite renderer for SphSimulatorV1; draws the simulator's pid
    range by passing firstVertex = own_first_pid - 1 to vkCmdDraw.

    Use:
        sim = SphSimulatorV1(ctx, case)
        sim.bootstrap()
        viewer = SphRendererV1(sim, window_width=1280, window_height=720)
        viewer.run()
        viewer.destroy()
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        simulator: SphSimulatorV1,
        window_width: int = 1280,
        window_height: int = 720,
    ):
        self.simulator = simulator
        self.ctx = simulator.ctx
        self.case = simulator.case

        # Window state
        self.window = None
        self.window_width = window_width
        self.window_height = window_height
        self.framebuffer_resized = False

        # Render config (live-tunable via hotkeys)
        self.color_mode = 0
        self.steps_per_frame = 1
        self.paused = False  # start running; SPACE to pause
        self.point_size = 5.0
        # Default scales (tunable live via ',' '.' hotkeys; current value
        # printed to stderr on each change).
        # velocity_scale=1 → saturates at 1 m/s (matches lid-driven cavity's
        #   driving lid speed; retune via hotkey for other characteristic speeds).
        # acceleration_scale=0.01 → saturates at 100 m/s² (gravity = 9.81;
        #   at impact, pressure forces are 1-2 orders larger).
        # density_deviation_scale=100 → saturates at 1% deviation. Typical
        #   WCSPH stays well under 1% for stable runs.
        self.velocity_scale = 1.0
        self.acceleration_scale = 1e-2
        self.density_deviation_scale = 100.0
        # kernel_sum_scale=20 → saturate at ±0.05 from 1.0 (= 0.95 to 1.05).
        self.kernel_sum_scale = 20.0
        # vorticity_scale=0.02 → saturate at |ω_z| = 50 1/s. force.comp's curl
        # uses the M-corrected gradient so ω_z is in physical units. Measured on
        # the lid-driven cavity: the lid shear layer reaches ~300/s while the
        # interior vortex builds up slowly (O(1-30)/s), so 50 shows both. Retune
        # live ( , / . in mode 5 ) for other flows.
        self.vorticity_scale = 0.02

        # Camera
        self.camera = Camera(projection_type="orthogonal")
        self.camera.update_aspect(window_width, window_height)
        self._frame_camera_to_case()

        # Mouse state
        self._mouse_left_down = False
        self._mouse_middle_down = False
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0

        # Vulkan handles (set during init)
        self.surface = None
        self.swapchain = None
        self.swapchain_format = None
        self.swapchain_extent = None
        self.image_views: list = []
        self.framebuffers: list = []
        self.render_pass = None
        self.descriptor_layout = None
        self.descriptor_pool = None
        self.descriptor_sets: list = []      # single set; binding 1 = canonical density_pressure
        self.pipeline_layout = None
        self.pipeline = None
        self.vert_module = None
        self.frag_module = None
        self.command_buffers: list = []      # MAX_FRAMES_IN_FLIGHT
        self.image_available_semaphores: list = []
        self.render_finished_semaphores: list = []
        self.in_flight_fences: list = []
        self.current_frame = 0
        self._destroyed = False

        # Per-frame loaded extension funcs (KHR swapchain etc.)
        self._fn_create_swapchain = None
        self._fn_destroy_swapchain = None
        self._fn_get_swapchain_images = None
        self._fn_acquire_next_image = None
        self._fn_queue_present = None
        self._fn_get_surface_capabilities = None
        self._fn_get_surface_formats = None
        self._fn_get_surface_present_modes = None
        self._fn_get_surface_support = None
        self._fn_destroy_surface = None

        # Init sequence
        self._load_extension_functions()
        self._init_window()
        self._create_surface()
        self._verify_present_support()
        self._load_shaders()
        self._create_swapchain()
        self._create_image_views()
        self._create_render_pass()
        self._create_framebuffers()
        self._create_descriptors()
        self._create_pipeline()
        self._create_command_buffers()
        self._create_sync_objects()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_extension_functions(self) -> None:
        instance = self.ctx.instance
        device = self.ctx.device
        self._fn_create_swapchain   = vkGetDeviceProcAddr(device,   "vkCreateSwapchainKHR")
        self._fn_destroy_swapchain  = vkGetDeviceProcAddr(device,   "vkDestroySwapchainKHR")
        self._fn_get_swapchain_images = vkGetDeviceProcAddr(device, "vkGetSwapchainImagesKHR")
        self._fn_acquire_next_image = vkGetDeviceProcAddr(device,   "vkAcquireNextImageKHR")
        self._fn_queue_present      = vkGetDeviceProcAddr(device,   "vkQueuePresentKHR")
        self._fn_get_surface_capabilities = vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR")
        self._fn_get_surface_formats      = vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceSurfaceFormatsKHR")
        self._fn_get_surface_present_modes = vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceSurfacePresentModesKHR")
        self._fn_get_surface_support      = vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceSurfaceSupportKHR")
        self._fn_destroy_surface          = vkGetInstanceProcAddr(instance, "vkDestroySurfaceKHR")

    def _init_window(self) -> None:
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)              # not OpenGL
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        title = f"SPH V1 - {self.case.case_dir.name}"
        self.window = glfw.create_window(
            self.window_width, self.window_height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("glfw.create_window failed")
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor_pos)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_key_callback(self.window, self._on_key)

    def _create_surface(self) -> None:
        # GLFW returns a VkSurfaceKHR via a void**; we need to cast it.
        surface_ptr = ctypes.c_void_p()
        result = glfw.create_window_surface(
            self.ctx.instance, self.window, None, ctypes.pointer(surface_ptr))
        if result != 0:
            raise RuntimeError(f"glfw.create_window_surface failed: {result}")
        self.surface = ffi.cast("VkSurfaceKHR", surface_ptr.value)

    def _verify_present_support(self) -> None:
        supported = self._fn_get_surface_support(
            self.ctx.physical_device,
            self.ctx.compute_queue_family_index,
            self.surface,
        )
        if not supported:
            raise RuntimeError(
                f"Queue family {self.ctx.compute_queue_family_index} does not "
                f"support presentation on this surface. V0 expects the chosen "
                f"compute queue family to also be present-capable.")

    def _load_shaders(self) -> None:
        for name in ("particle.vert", "particle.frag"):
            spv = SHADER_RENDER_DIR / f"{name}.spv"
            if not spv.exists():
                raise FileNotFoundError(f"compiled render shader missing: {spv}")
        vert_bytes = (SHADER_RENDER_DIR / "particle.vert.spv").read_bytes()
        frag_bytes = (SHADER_RENDER_DIR / "particle.frag.spv").read_bytes()
        self.vert_module = vkCreateShaderModule(self.ctx.device,
            VkShaderModuleCreateInfo(codeSize=len(vert_bytes), pCode=vert_bytes), None)
        self.frag_module = vkCreateShaderModule(self.ctx.device,
            VkShaderModuleCreateInfo(codeSize=len(frag_bytes), pCode=frag_bytes), None)

    def _query_swapchain_support(self):
        capabilities = self._fn_get_surface_capabilities(
            self.ctx.physical_device, self.surface)
        formats = self._fn_get_surface_formats(self.ctx.physical_device, self.surface)
        present_modes = self._fn_get_surface_present_modes(
            self.ctx.physical_device, self.surface)
        return capabilities, formats, present_modes

    def _choose_surface_format(self, formats):
        for f in formats:
            if (f.format == VK_FORMAT_B8G8R8A8_SRGB
                    and f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR):
                return f
        return formats[0]

    def _choose_present_mode(self, present_modes):
        # MAILBOX = low-latency triple-buffer (preferred). FIFO = vsync.
        for preferred in (VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR):
            if preferred in present_modes:
                return preferred
        return VK_PRESENT_MODE_FIFO_KHR

    def _choose_extent(self, capabilities):
        if capabilities.currentExtent.width != 0xFFFFFFFF:
            return VkExtent2D(
                width=capabilities.currentExtent.width,
                height=capabilities.currentExtent.height)
        w, h = glfw.get_framebuffer_size(self.window)
        w = max(capabilities.minImageExtent.width,
                min(capabilities.maxImageExtent.width, w))
        h = max(capabilities.minImageExtent.height,
                min(capabilities.maxImageExtent.height, h))
        return VkExtent2D(width=w, height=h)

    def _create_swapchain(self) -> None:
        capabilities, formats, present_modes = self._query_swapchain_support()
        surface_format = self._choose_surface_format(formats)
        present_mode = self._choose_present_mode(present_modes)
        extent = self._choose_extent(capabilities)

        image_count = capabilities.minImageCount + 1
        if capabilities.maxImageCount > 0 and image_count > capabilities.maxImageCount:
            image_count = capabilities.maxImageCount

        create_info = VkSwapchainCreateInfoKHR(
            surface=self.surface,
            minImageCount=image_count,
            imageFormat=surface_format.format,
            imageColorSpace=surface_format.colorSpace,
            imageExtent=extent,
            imageArrayLayers=1,
            imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            imageSharingMode=VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount=0,
            pQueueFamilyIndices=[],
            preTransform=capabilities.currentTransform,
            compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            presentMode=present_mode,
            clipped=VK_TRUE,
            oldSwapchain=None,
        )
        self.swapchain = self._fn_create_swapchain(self.ctx.device, create_info, None)
        self.swapchain_format = surface_format.format
        self.swapchain_extent = extent
        self.swapchain_images = self._fn_get_swapchain_images(
            self.ctx.device, self.swapchain)

    def _create_image_views(self) -> None:
        self.image_views = []
        for image in self.swapchain_images:
            create_info = VkImageViewCreateInfo(
                image=image,
                viewType=VK_IMAGE_VIEW_TYPE_2D,
                format=self.swapchain_format,
                components=VkComponentMapping(
                    r=VK_COMPONENT_SWIZZLE_IDENTITY,
                    g=VK_COMPONENT_SWIZZLE_IDENTITY,
                    b=VK_COMPONENT_SWIZZLE_IDENTITY,
                    a=VK_COMPONENT_SWIZZLE_IDENTITY,
                ),
                subresourceRange=VkImageSubresourceRange(
                    aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0, levelCount=1,
                    baseArrayLayer=0, layerCount=1,
                ),
            )
            self.image_views.append(
                vkCreateImageView(self.ctx.device, create_info, None))

    def _create_render_pass(self) -> None:
        color_attachment = VkAttachmentDescription(
            format=self.swapchain_format,
            samples=VK_SAMPLE_COUNT_1_BIT,
            loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        )
        color_ref = VkAttachmentReference(
            attachment=0, layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        subpass = VkSubpassDescription(
            pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1, pColorAttachments=[color_ref],
        )
        dependency = VkSubpassDependency(
            srcSubpass=VK_SUBPASS_EXTERNAL,
            dstSubpass=0,
            srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            srcAccessMask=0,
            dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        )
        create_info = VkRenderPassCreateInfo(
            attachmentCount=1, pAttachments=[color_attachment],
            subpassCount=1, pSubpasses=[subpass],
            dependencyCount=1, pDependencies=[dependency],
        )
        self.render_pass = vkCreateRenderPass(self.ctx.device, create_info, None)

    def _create_framebuffers(self) -> None:
        self.framebuffers = []
        for view in self.image_views:
            create_info = VkFramebufferCreateInfo(
                renderPass=self.render_pass,
                attachmentCount=1, pAttachments=[view],
                width=self.swapchain_extent.width,
                height=self.swapchain_extent.height,
                layers=1,
            )
            self.framebuffers.append(
                vkCreateFramebuffer(self.ctx.device, create_info, None))

    def _create_descriptors(self) -> None:
        # Layout: 5 SSBO bindings for vertex stage. Binding numbers match the
        # simulator's set 0 layout in shaders/sph/common.glsl exactly (sparse
        # set: render only needs a subset). This lets particle.vert
        # `#include "common.glsl"` and read sim's buffers under their canonical
        # binding numbers, eliminating any renderer-local mapping.
        #
        # Binding 1 points at density_pressure, the canonical buffer. The
        # simulator's step cmd has already copied scratch → primary by the
        # time we render, so this is always the freshly-written ρ.
        #
        #   binding 0 = position_voxel_id
        #   binding 1 = density_pressure (canonical; post-copy each step)
        #   binding 3 = velocity_mass
        #   binding 4 = acceleration
        #   binding 8 = density_gradient_kernel_sum
        render_binding_indices = [0, 1, 3, 4, 8]
        bindings = [
            VkDescriptorSetLayoutBinding(
                binding=binding_index,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_VERTEX_BIT,
            )
            for binding_index in render_binding_indices
        ]
        self.descriptor_layout = vkCreateDescriptorSetLayout(
            self.ctx.device,
            VkDescriptorSetLayoutCreateInfo(
                bindingCount=len(bindings), pBindings=bindings),
            None,
        )

        # Pool: 1 set × 5 SSBOs.
        pool_size = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=5)
        self.descriptor_pool = vkCreateDescriptorPool(
            self.ctx.device,
            VkDescriptorPoolCreateInfo(
                maxSets=1, poolSizeCount=1, pPoolSizes=[pool_size]),
            None,
        )

        allocate_info = VkDescriptorSetAllocateInfo(
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[self.descriptor_layout],
        )
        # `descriptor_sets` is a single-element list; index [0] is the only
        # used set. (Kept as a list so existing call sites that subscript
        # don't need to be reshaped.)
        self.descriptor_sets = vkAllocateDescriptorSets(
            self.ctx.device, allocate_info)

        sim_buffers = self.simulator.buffers
        buffer_pairs = [
            (0, sim_buffers["position_voxel_id"]),
            (1, sim_buffers["density_pressure"]),
            (3, sim_buffers["velocity_mass"]),
            (4, sim_buffers["acceleration"]),
            (8, sim_buffers["density_gradient_kernel_sum"]),
        ]
        writes = []
        for binding, buffer in buffer_pairs:
            buffer_info = VkDescriptorBufferInfo(
                buffer=buffer.handle, offset=0, range=buffer.size)
            writes.append(VkWriteDescriptorSet(
                dstSet=self.descriptor_sets[0],
                dstBinding=binding,
                dstArrayElement=0,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                pBufferInfo=[buffer_info],
            ))
            vkUpdateDescriptorSets(self.ctx.device, len(writes), writes, 0, None)

    def _create_pipeline(self) -> None:
        # Pipeline layout: 1 descriptor set + 1 push constant block.
        push_range = VkPushConstantRange(
            stageFlags=VK_SHADER_STAGE_VERTEX_BIT,
            offset=0, size=PUSH_CONSTANT_SIZE,
        )
        self.pipeline_layout = vkCreatePipelineLayout(
            self.ctx.device,
            VkPipelineLayoutCreateInfo(
                setLayoutCount=1, pSetLayouts=[self.descriptor_layout],
                pushConstantRangeCount=1, pPushConstantRanges=[push_range],
            ),
            None,
        )

        # Stages
        stages = [
            VkPipelineShaderStageCreateInfo(
                stage=VK_SHADER_STAGE_VERTEX_BIT,
                module=self.vert_module, pName="main"),
            VkPipelineShaderStageCreateInfo(
                stage=VK_SHADER_STAGE_FRAGMENT_BIT,
                module=self.frag_module, pName="main"),
        ]

        # Vertex input: NONE (vertex shader reads SSBO via gl_VertexIndex).
        vertex_input = VkPipelineVertexInputStateCreateInfo(
            vertexBindingDescriptionCount=0, pVertexBindingDescriptions=[],
            vertexAttributeDescriptionCount=0, pVertexAttributeDescriptions=[],
        )

        input_assembly = VkPipelineInputAssemblyStateCreateInfo(
            topology=VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
            primitiveRestartEnable=VK_FALSE,
        )

        # Dynamic viewport + scissor so we don't have to rebuild on resize.
        viewport_state = VkPipelineViewportStateCreateInfo(
            viewportCount=1, scissorCount=1)
        dynamic_state = VkPipelineDynamicStateCreateInfo(
            dynamicStateCount=2,
            pDynamicStates=[VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR],
        )

        rasterizer = VkPipelineRasterizationStateCreateInfo(
            depthClampEnable=VK_FALSE,
            rasterizerDiscardEnable=VK_FALSE,
            polygonMode=VK_POLYGON_MODE_FILL,
            cullMode=VK_CULL_MODE_NONE,
            frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE,
            depthBiasEnable=VK_FALSE,
            lineWidth=1.0,
        )
        multisampling = VkPipelineMultisampleStateCreateInfo(
            rasterizationSamples=VK_SAMPLE_COUNT_1_BIT,
            sampleShadingEnable=VK_FALSE,
        )

        # Alpha blending so point sprite edges fade smoothly.
        color_blend_attachment = VkPipelineColorBlendAttachmentState(
            blendEnable=VK_TRUE,
            srcColorBlendFactor=VK_BLEND_FACTOR_SRC_ALPHA,
            dstColorBlendFactor=VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            colorBlendOp=VK_BLEND_OP_ADD,
            srcAlphaBlendFactor=VK_BLEND_FACTOR_ONE,
            dstAlphaBlendFactor=VK_BLEND_FACTOR_ZERO,
            alphaBlendOp=VK_BLEND_OP_ADD,
            colorWriteMask=(VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                          | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT),
        )
        color_blending = VkPipelineColorBlendStateCreateInfo(
            logicOpEnable=VK_FALSE,
            attachmentCount=1, pAttachments=[color_blend_attachment],
        )

        create_info = VkGraphicsPipelineCreateInfo(
            stageCount=2, pStages=stages,
            pVertexInputState=vertex_input,
            pInputAssemblyState=input_assembly,
            pViewportState=viewport_state,
            pRasterizationState=rasterizer,
            pMultisampleState=multisampling,
            pColorBlendState=color_blending,
            pDynamicState=dynamic_state,
            layout=self.pipeline_layout,
            renderPass=self.render_pass,
            subpass=0,
        )
        result = vkCreateGraphicsPipelines(
            self.ctx.device, VK_NULL_HANDLE, 1, [create_info], None)
        self.pipeline = result[0]

    def _create_command_buffers(self) -> None:
        allocate_info = VkCommandBufferAllocateInfo(
            commandPool=self.ctx.command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=MAX_FRAMES_IN_FLIGHT,
        )
        self.command_buffers = vkAllocateCommandBuffers(
            self.ctx.device, allocate_info)

    def _create_sync_objects(self) -> None:
        # in_flight_fence is per in-flight frame; created once and survives
        # swapchain rebuild (not signaled by acquire/present, so resize-safe).
        for _ in range(MAX_FRAMES_IN_FLIGHT):
            self.in_flight_fences.append(
                vkCreateFence(self.ctx.device,
                              VkFenceCreateInfo(flags=VK_FENCE_CREATE_SIGNALED_BIT),
                              None))
        # image_available is per in-flight frame; recreated on swapchain
        # rebuild because acquire signals it on VK_SUBOPTIMAL_KHR (AMD's
        # typical resize signal). On SUBOPTIMAL python-vulkan raises and we
        # bail to _recreate_swapchain without ever consuming the signal.
        # Recreating ensures no stale signaled state survives.
        self._create_image_available_semaphores()
        # render_finished must be per swapchain image (not per in-flight frame):
        # the present operation may still hold the binary semaphore when we
        # cycle frame_index back, so reusing a frame-indexed signal semaphore
        # races with the swapchain. Indexing by image_index makes each
        # acquire-render-present cycle reuse only its own semaphore.
        self._create_render_finished_semaphores()

    def _create_image_available_semaphores(self) -> None:
        for sem in self.image_available_semaphores:
            vkDestroySemaphore(self.ctx.device, sem, None)
        self.image_available_semaphores = [
            vkCreateSemaphore(self.ctx.device, VkSemaphoreCreateInfo(), None)
            for _ in range(MAX_FRAMES_IN_FLIGHT)
        ]

    def _create_render_finished_semaphores(self) -> None:
        for sem in self.render_finished_semaphores:
            vkDestroySemaphore(self.ctx.device, sem, None)
        self.render_finished_semaphores = [
            vkCreateSemaphore(self.ctx.device, VkSemaphoreCreateInfo(), None)
            for _ in range(len(self.swapchain_images))
        ]

    def _frame_camera_to_case(self) -> None:
        h = self.case.physics.h
        origin = np.array(self.case.grid["origin"], dtype=np.float32)
        dim = np.array(self.case.grid["dimension"], dtype=np.float32)
        bbox_min = origin + 0.5 * h
        bbox_max = origin + (dim - 0.5) * h
        self.camera.frame_bbox(bbox_min, bbox_max, margin=1.4)

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self, log_fps_path: Optional[str] = None,
            auto_quit_seconds: Optional[float] = None) -> None:
        """Main event loop. If log_fps_path is set, append per-window samples
        to a CSV (header `wallclock_s,step_count,sim_time,fps_window_avg`) for
        benchmark comparison runs.

        - `wallclock_s` is relative to run() entry.
        - `step_count` is the simulator's step counter at sample time.
        - `sim_time` is the simulation's physical-time clock (== step_count·dt
          while dt is constant; logged explicitly for cross-case comparisons).
        - `fps_window_avg` is renderer frame rate over the last ~0.5 s window.

        Rows are only written when `step_count` strictly advanced since the
        previous log entry — paused / pre-SPACE intervals are skipped to keep
        the CSV compact and to ensure each row has a unique sim_time."""
        last_t = time.perf_counter()
        run_start_t = last_t
        frame_counter = 0
        title_base = f"SPH V1 - {self.case.case_dir.name}"

        log_file = None
        last_logged_step = 0    # step 0 == nothing has run; first row will be skipped.
        if log_fps_path is not None:
            log_path_obj = pathlib.Path(log_fps_path)
            log_path_obj.parent.mkdir(parents=True, exist_ok=True)
            log_file = open(log_path_obj, "w")
            log_file.write("wallclock_s,step_count,sim_time,fps_window_avg\n")

        try:
            while not glfw.window_should_close(self.window):
                glfw.poll_events()

                if (auto_quit_seconds is not None
                        and time.perf_counter() - run_start_t >= auto_quit_seconds):
                    break

                if not self.paused:
                    # Fire-and-forget compute submits — the per-frame render fence
                    # (vkWaitForFences at the top of _draw_frame) is the only CPU
                    # sync point; all SPH steps and the render submit share the
                    # same queue and serialize naturally. step()'s leading barrier
                    # handles cross-submission memory dependencies.
                    for _ in range(self.steps_per_frame):
                        self.simulator.step(wait=False)

                self._draw_frame()

                frame_counter += 1
                now = time.perf_counter()
                if now - last_t > 0.5:
                    fps = frame_counter / (now - last_t)
                    mode_name = ["speed", "accel", "density", "voxel_id", "kernel_sum", "vorticity"][self.color_mode]
                    pause_tag = "  [PAUSED]" if self.paused else ""
                    current_step = self.simulator.step_count
                    current_sim_time = self.simulator.simulation_time
                    glfw.set_window_title(
                        self.window,
                        f"{title_base}   step={current_step}   "
                        f"t={current_sim_time:.4e}s   "
                        f"{fps:.0f} fps   spf={self.steps_per_frame}   "
                        f"color={mode_name}{pause_tag}",
                    )
                    if log_file is not None and current_step > last_logged_step:
                        # Flush each row so SIGINT / window-close preserves data.
                        log_file.write(
                            f"{now - run_start_t:.3f},{current_step},"
                            f"{current_sim_time:.6e},{fps:.2f}\n")
                        log_file.flush()
                        last_logged_step = current_step
                    frame_counter = 0
                    last_t = now

            vkDeviceWaitIdle(self.ctx.device)
        finally:
            if log_file is not None:
                log_file.close()

    def _draw_frame(self) -> None:
        device = self.ctx.device
        frame_index = self.current_frame
        fence = self.in_flight_fences[frame_index]
        image_available = self.image_available_semaphores[frame_index]
        cmd = self.command_buffers[frame_index]

        vkWaitForFences(device, 1, [fence], VK_TRUE, 0xFFFFFFFFFFFFFFFF)

        # Acquire next image (handle resize / out-of-date)
        try:
            image_index = self._fn_acquire_next_image(
                device, self.swapchain, 0xFFFFFFFFFFFFFFFF,
                image_available, VK_NULL_HANDLE)
        except Exception as exc:
            # python-vulkan raises on OUT_OF_DATE/SUBOPTIMAL; recreate swapchain
            self._recreate_swapchain()
            return

        # render_finished is indexed by the acquired image (not by in-flight
        # frame) so present operations don't race when frame_index cycles.
        render_finished = self.render_finished_semaphores[image_index]

        vkResetFences(device, 1, [fence])
        vkResetCommandBuffer(cmd, 0)
        self._record_render_cmd(cmd, image_index)

        wait_stages = [VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT]
        submit_info = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            waitSemaphoreCount=1, pWaitSemaphores=[image_available],
            pWaitDstStageMask=wait_stages,
            commandBufferCount=1, pCommandBuffers=[cmd],
            signalSemaphoreCount=1, pSignalSemaphores=[render_finished],
        )
        vkQueueSubmit(self.ctx.compute_queue, 1, submit_info, fence)

        present_info = VkPresentInfoKHR(
            sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            waitSemaphoreCount=1, pWaitSemaphores=[render_finished],
            swapchainCount=1, pSwapchains=[self.swapchain],
            pImageIndices=[image_index],
        )
        try:
            self._fn_queue_present(self.ctx.compute_queue, present_info)
        except Exception:
            self._recreate_swapchain()

        if self.framebuffer_resized:
            self.framebuffer_resized = False
            self._recreate_swapchain()

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT

    def _record_render_cmd(self, cmd, image_index: int) -> None:
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(flags=0))

        # Compute → graphics handover barrier. Particle SSBOs were last
        # written by simulator.step()'s force.comp (compute stage); the
        # particle vertex shader reads them. Same queue + same fence handles
        # ordering, but we still need an explicit memory dependency to
        # propagate writes from the compute caches to the vertex stage.
        compute_to_vertex = VkMemoryBarrier(
            sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=VK_ACCESS_SHADER_READ_BIT,
        )
        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,    # src
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,     # dst
            0,
            1, [compute_to_vertex],
            0, None,
            0, None,
        )

        clear_color = VkClearValue(color=VkClearColorValue(float32=[0.05, 0.05, 0.07, 1.0]))
        render_pass_begin = VkRenderPassBeginInfo(
            renderPass=self.render_pass,
            framebuffer=self.framebuffers[image_index],
            renderArea=VkRect2D(
                offset=VkOffset2D(x=0, y=0), extent=self.swapchain_extent),
            clearValueCount=1, pClearValues=[clear_color],
        )
        vkCmdBeginRenderPass(cmd, render_pass_begin, VK_SUBPASS_CONTENTS_INLINE)

        # Vulkan Y-flip via negative-height viewport (origin at bottom-left).
        viewport = VkViewport(
            x=0.0, y=float(self.swapchain_extent.height),
            width=float(self.swapchain_extent.width),
            height=-float(self.swapchain_extent.height),
            minDepth=0.0, maxDepth=1.0,
        )
        scissor = VkRect2D(
            offset=VkOffset2D(x=0, y=0), extent=self.swapchain_extent)
        vkCmdSetViewport(cmd, 0, 1, [viewport])
        vkCmdSetScissor(cmd, 0, 1, [scissor])

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline)

        # Single descriptor set; binding 1 always points at the canonical
        # density_pressure (post scratch→primary copy in the simulator's
        # step cmd).
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            self.pipeline_layout, 0,
            1, [self.descriptor_sets[0]],
            0, None,
        )

        # Push constants — view_proj + color params.
        view_projection = self.camera.view_projection()
        rest_density = (
            self.case.materials[0].rest_density if self.case.materials else 1000.0
        )
        push_data = bytearray()
        push_data.extend(view_projection.tobytes())                  # 64 B
        push_data.extend(np.uint32(self.color_mode).tobytes())       #  4 B
        push_data.extend(np.float32(self.velocity_scale).tobytes())  #  4 B
        push_data.extend(np.float32(self.acceleration_scale).tobytes())       # 4 B
        push_data.extend(np.float32(self.density_deviation_scale).tobytes())  # 4 B
        push_data.extend(np.float32(rest_density).tobytes())         #  4 B
        push_data.extend(np.float32(self.point_size).tobytes())      #  4 B
        push_data.extend(np.float32(self.kernel_sum_scale).tobytes())         # 4 B
        push_data.extend(np.float32(self.vorticity_scale).tobytes())          # 4 B
        assert len(push_data) == PUSH_CONSTANT_SIZE
        push_cdata = ffi.new("uint8_t[]", bytes(push_data))
        vkCmdPushConstants(
            cmd, self.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
            0, PUSH_CONSTANT_SIZE, push_cdata,
        )

        # firstVertex = own_first_pid - 1 shifts gl_VertexIndex so that the
        # vertex shader's `gl_VertexIndex + 1u` lands at own_first_pid for
        # vertex 0 and own_last_pid for vertex (POOL_SIZE - 1).
        own_pool = self.case.capacities.pool_size
        first_vertex = self.simulator.own_first_pid() - 1
        vkCmdDraw(cmd, own_pool, 1, first_vertex, 0)

        vkCmdEndRenderPass(cmd)
        vkEndCommandBuffer(cmd)

    # ------------------------------------------------------------------
    # Resize handling
    # ------------------------------------------------------------------

    def _recreate_swapchain(self) -> None:
        # Block until window has non-zero area (minimization)
        width, height = glfw.get_framebuffer_size(self.window)
        while width == 0 or height == 0:
            glfw.wait_events()
            width, height = glfw.get_framebuffer_size(self.window)

        vkDeviceWaitIdle(self.ctx.device)

        # Tear down swapchain-dependent resources
        for fb in self.framebuffers:
            vkDestroyFramebuffer(self.ctx.device, fb, None)
        self.framebuffers = []
        for view in self.image_views:
            vkDestroyImageView(self.ctx.device, view, None)
        self.image_views = []
        if self.swapchain is not None:
            self._fn_destroy_swapchain(self.ctx.device, self.swapchain, None)
            self.swapchain = None

        self._create_swapchain()
        self._create_image_views()
        self._create_framebuffers()
        # Image count may differ after resize → re-create the per-image
        # render_finished semaphores so the indexing stays valid.
        self._create_render_finished_semaphores()
        # Also recreate image_available: AMD-style SUBOPTIMAL acquire signals
        # this semaphore but our exception handler returns before any submit
        # consumes it. Without recreation, the next frame's acquire on the
        # same per-frame slot trips "Semaphore must not be currently signaled".
        self._create_image_available_semaphores()

        self.window_width = self.swapchain_extent.width
        self.window_height = self.swapchain_extent.height
        self.camera.update_aspect(self.window_width, self.window_height)

    # ------------------------------------------------------------------
    # GLFW callbacks
    # ------------------------------------------------------------------

    def _on_resize(self, window, width, height):
        self.framebuffer_resized = True

    def _on_mouse_button(self, window, button, action, mods):
        pressed = (action == glfw.PRESS)
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._mouse_left_down = pressed
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self._mouse_middle_down = pressed
        if pressed:
            x, y = glfw.get_cursor_pos(window)
            self._last_mouse_x, self._last_mouse_y = x, y

    def _on_cursor_pos(self, window, xpos, ypos):
        dx = xpos - self._last_mouse_x
        dy = ypos - self._last_mouse_y
        self._last_mouse_x, self._last_mouse_y = xpos, ypos
        if self._mouse_left_down:
            self.camera.rotate(dx, dy)
        elif self._mouse_middle_down:
            self.camera.translate(dx, dy)

    def _on_scroll(self, window, dx, dy):
        self.camera.zoom(dy)

    def _on_key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            self.paused = not self.paused
        elif key == glfw.KEY_P:
            self.camera.switch_projection()
        elif key == glfw.KEY_F:
            self._frame_camera_to_case()
        elif key in (glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3,
                     glfw.KEY_4, glfw.KEY_5):
            self.color_mode = key - glfw.KEY_0
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):       # '+' / numpad +
            self.steps_per_frame += 1
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):  # '-' / numpad -
            self.steps_per_frame = max(1, self.steps_per_frame - 1)
        elif key == glfw.KEY_LEFT_BRACKET:
            self.steps_per_frame = max(1, self.steps_per_frame // 2)
        elif key == glfw.KEY_RIGHT_BRACKET:
            self.steps_per_frame *= 2
        # ';' / ''' shrink / grow the rendered point size (live).
        elif key == glfw.KEY_SEMICOLON:
            self.point_size = max(0.5, self.point_size / 1.25)
            print(f"[viewer] point_size = {self.point_size:.3g} px")
        elif key == glfw.KEY_APOSTROPHE:
            self.point_size = min(64.0, self.point_size * 1.25)
            print(f"[viewer] point_size = {self.point_size:.3g} px")
        # ',' / '.' adjust whichever scale the current color_mode uses.
        elif key == glfw.KEY_COMMA:
            self._scale_current_mode(1.0 / 1.5)
        elif key == glfw.KEY_PERIOD:
            self._scale_current_mode(1.5)

    def _scale_current_mode(self, factor: float) -> None:
        """Multiply the active color mode's scale by `factor` (live tuning)."""
        if self.color_mode == 0:
            self.velocity_scale *= factor
            print(f"[viewer] velocity_scale = {self.velocity_scale:.4g} "
                  f"(saturates at speed = {1.0/self.velocity_scale:.4g} m/s)")
        elif self.color_mode == 1:
            self.acceleration_scale *= factor
            print(f"[viewer] acceleration_scale = {self.acceleration_scale:.4g} "
                  f"(saturates at |a| = {1.0/self.acceleration_scale:.4g} m/s²)")
        elif self.color_mode == 2:
            self.density_deviation_scale *= factor
            print(f"[viewer] density_deviation_scale = {self.density_deviation_scale:.4g} "
                  f"(saturates at (rho-rho0)/rho0 = +-{1.0/self.density_deviation_scale:.4g})")
        elif self.color_mode == 4:
            self.kernel_sum_scale *= factor
            print(f"[viewer] kernel_sum_scale = {self.kernel_sum_scale:.4g} "
                  f"(saturates at kernel_sum = 1.0 +- {1.0/self.kernel_sum_scale:.4g})")
        elif self.color_mode == 5:
            self.vorticity_scale *= factor
            print(f"[viewer] vorticity_scale = {self.vorticity_scale:.4g} "
                  f"(saturates at |omega_z| = {1.0/self.vorticity_scale:.4g} 1/s)")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        if self._destroyed:
            return
        device = self.ctx.device
        if device is not None:
            vkDeviceWaitIdle(device)

        for sem in self.image_available_semaphores:
            vkDestroySemaphore(device, sem, None)
        for sem in self.render_finished_semaphores:
            vkDestroySemaphore(device, sem, None)
        for fence in self.in_flight_fences:
            vkDestroyFence(device, fence, None)

        if self.command_buffers:
            vkFreeCommandBuffers(
                device, self.ctx.command_pool,
                len(self.command_buffers), self.command_buffers)

        if self.pipeline is not None:
            vkDestroyPipeline(device, self.pipeline, None)
        if self.pipeline_layout is not None:
            vkDestroyPipelineLayout(device, self.pipeline_layout, None)
        if self.descriptor_pool is not None:
            vkDestroyDescriptorPool(device, self.descriptor_pool, None)
        if self.descriptor_layout is not None:
            vkDestroyDescriptorSetLayout(device, self.descriptor_layout, None)

        for fb in self.framebuffers:
            vkDestroyFramebuffer(device, fb, None)
        for view in self.image_views:
            vkDestroyImageView(device, view, None)
        if self.swapchain is not None:
            self._fn_destroy_swapchain(device, self.swapchain, None)
        if self.render_pass is not None:
            vkDestroyRenderPass(device, self.render_pass, None)

        if self.vert_module is not None:
            vkDestroyShaderModule(device, self.vert_module, None)
        if self.frag_module is not None:
            vkDestroyShaderModule(device, self.frag_module, None)

        if self.surface is not None:
            self._fn_destroy_surface(self.ctx.instance, self.surface, None)

        if self.window is not None:
            glfw.destroy_window(self.window)
        glfw.terminate()
        self._destroyed = True

    def __enter__(self) -> "SphRendererV1":
        return self

    def __exit__(self, *_) -> None:
        self.destroy()
