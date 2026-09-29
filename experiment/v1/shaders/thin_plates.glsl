// ============================================================================
// thin_plates.glsl  (2026-09-30)
//
// Thin plates wetted on both sides (impeller blades, the Rushton disk, the
// baffles). Included only by the builds compiled with -DWITH_THIN_PLATES=1
// (density_plates, force_plates, force_scalar_plates); the default builds do
// not contain any of this.
//
// What is wrong with a plate made of one layer of ordinary solid particles
//   1. A solid particle has ONE pressure, the fluid on both sides reads the
//      same value. With a shared pressure a pressure difference across the
//      plate has no state of rest (the balance of the two fluid layers next to
//      the plate has the only solution P_front = P_back).
//   2. The support radius is 3 dx, the plate is 1 dx thick: fluid particles on
//      opposite sides are neighbours of each other.
//
// Treatment (side-aware mirror; the splitting into sides and the cut of the
// pairs across the plate are those of the normal flux method, Gao and Fu 2024,
// CMAME 429:117179, the wall state is the pairwise mirror of Adami et al. 2012
// with the acoustic term of De Leffe et al. / Chiron et al. 2019)
//   A plate is a plane rectangle or annulus, described analytically
//   (ThinPlateBuffer). For a fluid particle i, every neighbour j BEHIND a plate
//   - a particle of the plate itself, or any particle whose connecting segment
//   with i crosses the mid-plane of the plate inside its outline - is a dummy
//   particle of the wall:
//       velocity   v_j := velocity of the plate at x_j (0, or omega x r)
//       pressure   particle i over the plate:
//                  P_j := P_i + G_i . d_t + rho_i [(g - a_w) . n] (n . d)
//                         + beta rho_i c0 (v_i - v_j) . n_out
//                  particle i beside the plate (its projection outside the outline):
//                  P_j := P_i + G_i . d
//                  d = x_j - x_i, d_t its part along the plate
//       no density diffusion, no scalar flux, the matrix of particle i.
//   G_i is the pressure gradient of the fluid on the side of particle i (from
//   the density gradient of the neighbours that are not behind a plate,
//   correction.comp): along the wall the pressure of the fluid continues into
//   the dummies (as it does in the wall particles of Adami et al. 2012, whose
//   pressure is a mean over the fluid next to them); normal to the wall the
//   gradient is the one a fluid at the wall has, dp/dn = rho (g - a_w) . n.
//   a_w is the acceleration of the plate at x_j (rotor: centripetal plus the
//   angular acceleration of the ramp), n_out the normal of the plate pointing
//   away from the side of particle i. beta = numerics.thin_plate_dashpot,
//   default 0: with beta = 1 a fluid that approaches a plate at 5 % of the tip
//   speed at the distance of one spacing already gives rho U^2.
//   Everything else is untouched: the correction matrix, the kernel sum and
//   the particle shift see all particles as they are, so the support of every
//   particle is complete also next to a plate edge.
//
//   The surface integral form of the literature was implemented first and is
//   in the history (commit 441deb4, log/2026-09-30_thin-plates-normal-flux.md):
//   cutting the pairs without filling the region behind the plate leaves the
//   support incomplete at the surfaces through the plate edges; under an
//   absolute pressure of 1 kPa the symmetric form then stirs the fluid at rest
//   to 0.3 m/s and the difference form is linearly unstable.
//
// A plate particle is marked in the PRESSURE slot,
//     density_pressure[k] = ( rho0,  -(plate index + 1) * THIN_PLATE_MARKER ),
// so that every neighbour loop recognises it with the load it already makes;
// its density stays rho0 (it is a particle of the volume V_p like any other).
//
// Load on the plates (read back): the reaction. force.comp stores for every
// fluid particle the force it received from the wall dummies of thin plates
// (ThinPlateReactionBuffer); the load on the plates is minus their sum, the
// torque about the rotor axis minus the sum of x_i x f_i. A pressure
// integration over the plate particles (Shepard mean per face times the area)
// was used first; it was 10 - 30 % below what the fluid received (rotating
// paddle test, log/2026-09-30_thin-plates-mirror.md).
// ============================================================================

#ifndef SPH_THIN_PLATES_GLSL_INCLUDED
#define SPH_THIN_PLATES_GLSL_INCLUDED

const uint  MAX_NEAR_THIN_PLATES       = 4u;
const uint  THIN_PLATE_SHAPE_RECTANGLE = 0u;
const uint  THIN_PLATE_SHAPE_ANNULUS   = 1u;
const uint  THIN_PLATE_FRAME_STATIC    = 0u;
const uint  THIN_PLATE_FRAME_ROTOR     = 1u;
const float THIN_PLATE_MARKER          = 1.0e9;   // Pa; must match simulator_v1.py

// A plate close to the particle this invocation works on, in world
// coordinates at the current rotor angle.
struct NearThinPlate {
    vec3  centre;
    vec3  normal;
    vec3  axis_a;
    vec3  axis_b;
    float extent_a;          // rectangle: half length along axis_a; annulus: outer radius
    float extent_b;          // rectangle: half length along axis_b; annulus: inner radius
    float half_thickness;
    float point_measure;     // area (3D) or length (2D) of one plate particle
    float self_distance;     // normal . (x_self - centre)
    bool  self_over_plate;   // the projection of x_self lies inside the outline
    uint  shape;
    uint  frame;
    uint  index;
};

NearThinPlate near_thin_plate[MAX_NEAR_THIN_PLATES];
uint near_thin_plate_count = 0u;

bool thin_plate_outline_contains(uint shape, float extent_a, float extent_b,
                                 float coordinate_a, float coordinate_b) {
    if (shape == THIN_PLATE_SHAPE_ANNULUS) {
        float radius_squared = coordinate_a * coordinate_a + coordinate_b * coordinate_b;
        return radius_squared <= extent_a * extent_a && radius_squared >= extent_b * extent_b;
    }
    return abs(coordinate_a) <= extent_a && abs(coordinate_b) <= extent_b;
}

// Squared in-plane distance from a point to the outline (0 inside).
float thin_plate_outline_gap_squared(uint shape, float extent_a, float extent_b,
                                     float coordinate_a, float coordinate_b) {
    if (shape == THIN_PLATE_SHAPE_ANNULUS) {
        float radius = sqrt(coordinate_a * coordinate_a + coordinate_b * coordinate_b);
        float gap = max(max(radius - extent_a, extent_b - radius), 0.0);
        return gap * gap;
    }
    vec2 gap = max(abs(vec2(coordinate_a, coordinate_b)) - vec2(extent_a, extent_b), vec2(0.0));
    return dot(gap, gap);
}

// Fill near_thin_plate[] with the plates that come closer than the support
// radius to `position`.
void gather_near_thin_plates(vec3 position) {
    near_thin_plate_count = 0u;
    vec3 rotor_axis  = normalize(vec3(ROTOR_AXIS_X, ROTOR_AXIS_Y, ROTOR_AXIS_Z));
    vec3 rotor_pivot = vec3(ROTOR_PIVOT_X, ROTOR_PIVOT_Y, ROTOR_PIVOT_Z);
    for (uint group_index = 0u; group_index < THIN_PLATE_GROUP_COUNT; group_index++) {
        vec4 bounding_sphere = thin_plate_group[group_index].centre_radius;
        vec3 to_centre = position - bounding_sphere.xyz;
        if (dot(to_centre, to_centre) >= bounding_sphere.w * bounding_sphere.w) continue;

        uvec4 plate_range = thin_plate_group[group_index].range;
        bool  rotor_frame = (plate_range.z == THIN_PLATE_FRAME_ROTOR);
        // the plates of a rotor group are stored at the rotor angle 0
        vec3 frame_position = position;
        if (rotor_frame) {
            frame_position = rotor_pivot + rotate_about_axis(position - rotor_pivot, rotor_axis,
                                                             rotor_cos_theta, -rotor_sin_theta);
        }
        for (uint plate_index = plate_range.x; plate_index < plate_range.x + plate_range.y; plate_index++) {
            ThinPlate plate = thin_plate[plate_index];
            vec3  relative        = frame_position - plate.centre_extent_a.xyz;
            float distance_normal = dot(relative, plate.normal_extent_b.xyz);
            float coordinate_a    = dot(relative, plate.axis_a_half_thickness.xyz);
            float coordinate_b    = dot(relative, plate.axis_b_measure.xyz);
            float outline_gap_squared = thin_plate_outline_gap_squared(
                plate.flags.x, plate.centre_extent_a.w, plate.normal_extent_b.w, coordinate_a, coordinate_b);
            if (distance_normal * distance_normal + outline_gap_squared >= smoothing_length_squared()) continue;
            if (near_thin_plate_count >= MAX_NEAR_THIN_PLATES) return;

            NearThinPlate entry;
            entry.centre = plate.centre_extent_a.xyz;
            entry.normal = plate.normal_extent_b.xyz;
            entry.axis_a = plate.axis_a_half_thickness.xyz;
            entry.axis_b = plate.axis_b_measure.xyz;
            if (rotor_frame) {
                entry.centre = rotor_pivot + rotate_about_axis(entry.centre - rotor_pivot, rotor_axis,
                                                               rotor_cos_theta, rotor_sin_theta);
                entry.normal = rotate_about_axis(entry.normal, rotor_axis, rotor_cos_theta, rotor_sin_theta);
                entry.axis_a = rotate_about_axis(entry.axis_a, rotor_axis, rotor_cos_theta, rotor_sin_theta);
                entry.axis_b = rotate_about_axis(entry.axis_b, rotor_axis, rotor_cos_theta, rotor_sin_theta);
            }
            entry.extent_a        = plate.centre_extent_a.w;
            entry.extent_b        = plate.normal_extent_b.w;
            entry.half_thickness  = plate.axis_a_half_thickness.w;
            entry.point_measure   = plate.axis_b_measure.w;
            entry.self_distance   = distance_normal;
            entry.self_over_plate = (outline_gap_squared == 0.0);
            entry.shape           = plate.flags.x;
            entry.frame           = plate.flags.y;
            entry.index           = plate_index;
            near_thin_plate[near_thin_plate_count] = entry;
            near_thin_plate_count++;
        }
    }
}

// Does the segment cross the mid-plane of the near plate `slot` inside its outline?
bool thin_plate_slot_blocks_segment(uint slot, vec3 from_position, vec3 to_position) {
    vec3  normal        = near_thin_plate[slot].normal;
    vec3  centre        = near_thin_plate[slot].centre;
    float distance_from = dot(normal, from_position - centre);
    float distance_to   = dot(normal, to_position - centre);
    if (distance_from * distance_to >= 0.0) return false;
    float fraction = distance_from / (distance_from - distance_to);
    vec3  crossing = from_position + fraction * (to_position - from_position) - centre;
    return thin_plate_outline_contains(near_thin_plate[slot].shape,
                                       near_thin_plate[slot].extent_a, near_thin_plate[slot].extent_b,
                                       dot(crossing, near_thin_plate[slot].axis_a),
                                       dot(crossing, near_thin_plate[slot].axis_b));
}

// First near plate that separates the pair; MAX_NEAR_THIN_PLATES if none.
uint thin_plate_blocking_slot(vec3 from_position, vec3 to_position) {
    for (uint slot = 0u; slot < near_thin_plate_count; slot++) {
        if (thin_plate_slot_blocks_segment(slot, from_position, to_position)) return slot;
    }
    return MAX_NEAR_THIN_PLATES;
}

// A plate particle stores -(plate index + 1) * THIN_PLATE_MARKER as its pressure.
bool is_thin_plate_particle(float stored_pressure) {
    return stored_pressure < -0.5 * THIN_PLATE_MARKER;
}

// Slot of the plate a plate particle belongs to; MAX_NEAR_THIN_PLATES if that
// plate is not in the near list.
uint thin_plate_slot_of_particle(float stored_pressure) {
    uint plate_index = uint(-stored_pressure / THIN_PLATE_MARKER - 0.5);
    for (uint slot = 0u; slot < near_thin_plate_count; slot++) {
        if (near_thin_plate[slot].index == plate_index) return slot;
    }
    return MAX_NEAR_THIN_PLATES;
}

// For the particle at `position` (the one the near list was gathered for) and
// its neighbour: the plate the neighbour is a wall dummy of, or
// MAX_NEAR_THIN_PLATES if it is an ordinary neighbour.
uint thin_plate_wall_slot(vec3 position, vec3 neighbor_position, float neighbor_stored_pressure) {
    if (is_thin_plate_particle(neighbor_stored_pressure)) {
        return thin_plate_slot_of_particle(neighbor_stored_pressure);
    }
    return thin_plate_blocking_slot(position, neighbor_position);
}

// Velocity and acceleration of the plate `slot` at a point (rigid rotation for
// a rotor plate: centripetal part plus the angular acceleration of the ramp).
vec3 thin_plate_velocity(uint slot, vec3 position) {
    if (near_thin_plate[slot].frame != THIN_PLATE_FRAME_ROTOR) return vec3(0.0);
    vec3 rotor_axis = normalize(vec3(ROTOR_AXIS_X, ROTOR_AXIS_Y, ROTOR_AXIS_Z));
    vec3 arm        = position - vec3(ROTOR_PIVOT_X, ROTOR_PIVOT_Y, ROTOR_PIVOT_Z);
    return rotor_angular_velocity_now * cross(rotor_axis, arm);
}

vec3 thin_plate_acceleration(uint slot, vec3 position) {
    if (near_thin_plate[slot].frame != THIN_PLATE_FRAME_ROTOR) return vec3(0.0);
    vec3 rotor_axis = normalize(vec3(ROTOR_AXIS_X, ROTOR_AXIS_Y, ROTOR_AXIS_Z));
    vec3 arm        = position - vec3(ROTOR_PIVOT_X, ROTOR_PIVOT_Y, ROTOR_PIVOT_Z);
    return solid_particle_acceleration(MATERIAL_ROTOR, position)
         + rotor_angular_acceleration_now * cross(rotor_axis, arm);
}

// Pressure of a wall dummy at `dummy_position` as the fluid particle sees it.
// `fluid_distance` is the signed distance of the fluid particle from the
// mid-plane, `over_plate` whether it stands over the plate,
// `fluid_pressure_gradient` the pressure gradient of the fluid on its side.
float thin_plate_mirror_pressure(uint slot, float fluid_pressure, float fluid_density,
                                 vec3 fluid_position, vec3 fluid_velocity,
                                 float fluid_distance, bool over_plate, vec3 dummy_position,
                                 vec3 fluid_pressure_gradient) {
    vec3 offset = dummy_position - fluid_position;
    if (!over_plate) {
        // beside the plate: the field of this side continued into the shadow
        return fluid_pressure + dot(fluid_pressure_gradient, offset);
    }
    // Over the plate. Along the wall: the gradient of the fluid. Normal to it:
    // dp/dn = rho (g - a_w) . n. (With the full vector (g - a_w) the dummies of
    // a rotor plate would carry the gradient rho omega^2 r of a fluid turning
    // rigidly with the plate, which is not the state of the fluid a blade
    // throws outward.)
    vec3  plate_normal  = near_thin_plate[slot].normal;
    float offset_normal = dot(plate_normal, offset);
    vec3  relative_body_force = vec3(GRAVITY_X, GRAVITY_Y, GRAVITY_Z)
                              - thin_plate_acceleration(slot, dummy_position);
    float mirror = fluid_pressure
                 + dot(fluid_pressure_gradient, offset - offset_normal * plate_normal)
                 + fluid_density * dot(relative_body_force, plate_normal) * offset_normal;
    // acoustic term (off by default): the fluid approaches the plate
    vec3 normal_out = ((fluid_distance >= 0.0) ? -1.0 : 1.0) * plate_normal;
    mirror += THIN_PLATE_DASHPOT * fluid_density * SPEED_OF_SOUND
            * dot(fluid_velocity - thin_plate_velocity(slot, dummy_position), normal_out);
    return mirror;
}

#endif  // SPH_THIN_PLATES_GLSL_INCLUDED
