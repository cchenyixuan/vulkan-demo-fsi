// ============================================================================
// thin_plates.glsl  (2026-09-30)
//
// Thin plates wetted on both sides (impeller blades, the Rushton disk, the
// baffles), treated with the normal flux (boundary integral) method. Included
// only by the builds compiled with -DWITH_THIN_PLATES=1 (correction, density,
// force, defrag); the default builds do not contain any of this.
//
// Literature: Gao and Fu 2024 (CMAME 429:117179, 3D, the method followed here),
// Bao et al. 2024 (CMAME 431:117255), Peng et al. 2021 (JFS 102:103254),
// Li et al. 2022 (JCP 464:111328), Chiron et al. 2019 (CPC 234:93).
// docs/thin_wall_normal_flux_study_2026-09-30.md, log/2026-09-30_thin-plates.md.
//
// Representation
//   A plate is a plane rectangle or annulus with a thickness t, described
//   analytically (ThinPlateBuffer). Its particles lie on the mid-plane and
//   serve as quadrature points only: they are no volume neighbours of anybody.
//   A quadrature point k has the two face points
//       x_s = x_k + s (t / 2) n,      s = +1, -1
//   and the particle i receives from the face s the surface terms with the
//   normal n_out = -s n (pointing from the fluid into the plate), if it SEES
//   the face: the segment from x_i to x_s crosses no plate.
//   A plate particle is marked by a negative density,
//       density_pressure[k] = ( -(plate index + 1),  measure of the point ),
//   measure = area (3D) or length (2D) the point stands for, so that every
//   neighbour loop recognises it with the load it already makes.
//
// Pairs across a plate
//   A pair (i, j) whose segment crosses a plate inside its outline does not
//   interact: pressure, viscosity, continuity, diffusion, scalars, correction
//   matrix. The PARTICLE SHIFT and the KERNEL SUM ignore the cut: they see
//   every neighbour, and a plate point as one particle of a layer of the
//   spacing dx. The shift only regularises the arrangement of the particles;
//   computed on the cut neighbourhood it has no state of rest next to a plate
//   edge (the neighbourhood ends at the surfaces through the edge, where there
//   is fluid) and pumps particles around the edge (free edge test,
//   2026-09-30: 222 particles through the plate in 0.54 s).
//
// Surface terms (B_i = inverse of the correction matrix of particle i, which
// itself contains the surface term, w_k = measure * V_p / dx^d):
//   matrix      N_i  += w_k W_is  n_out (x) (x_s - x_i)
//   continuity  d rho_i / dt += rho_i w_k W_is (v_i - v_k) . B_i n_out
//   pressure    a_i  -= w_k W_is / rho_i  (P_i + P_ik)  B_i n_out      (or P_ik - P_i)
//   wall pressure    P_ik = P_i + rho_i (g - a_k) . (x_s - x_i)
//                           + beta rho_i c0 (v_i - v_k) . n_out
//   viscosity   a_i  += 2 nu w_k W_is (v_i - v_k) (x_i - x_s) . B_i n_out / (r^2 + eta^2)
//   shift       not a surface term, see above; its part pointing into a plate
//               is removed closer than half a spacing to the face
//   load on the plate (read back): pressure integration without B_i,
//       F_k = sum_i (m_i / rho_i) measure [ (P_i + P_ik) n_out W_is - viscous term ]
// ============================================================================

#ifndef SPH_THIN_PLATES_GLSL_INCLUDED
#define SPH_THIN_PLATES_GLSL_INCLUDED

const uint MAX_NEAR_THIN_PLATES       = 4u;
const uint THIN_PLATE_SHAPE_RECTANGLE = 0u;
const uint THIN_PLATE_SHAPE_ANNULUS   = 1u;
const uint THIN_PLATE_FRAME_STATIC    = 0u;
const uint THIN_PLATE_FRAME_ROTOR     = 1u;

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
    float point_measure;     // area (3D) or length (2D) of one quadrature point
    float self_distance;     // normal . (x_self - centre)
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

// Fill near_thin_plate[] with the plates whose slab comes closer than the
// support radius to `position`.
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
            float half_thickness  = plate.axis_a_half_thickness.w;
            float gap_normal      = max(abs(distance_normal) - half_thickness, 0.0);
            float gap_squared     = gap_normal * gap_normal
                + thin_plate_outline_gap_squared(plate.flags.x, plate.centre_extent_a.w,
                                                 plate.normal_extent_b.w, coordinate_a, coordinate_b);
            if (gap_squared >= smoothing_length_squared()) continue;
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
            entry.extent_a       = plate.centre_extent_a.w;
            entry.extent_b       = plate.normal_extent_b.w;
            entry.half_thickness = half_thickness;
            entry.point_measure  = plate.axis_b_measure.w;
            entry.self_distance  = distance_normal;
            entry.shape          = plate.flags.x;
            entry.frame          = plate.flags.y;
            entry.index          = plate_index;
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

// Is the pair separated by one of the near plates?
bool thin_plate_blocks_segment(vec3 from_position, vec3 to_position) {
    for (uint slot = 0u; slot < near_thin_plate_count; slot++) {
        if (thin_plate_slot_blocks_segment(slot, from_position, to_position)) return true;
    }
    return false;
}

// A plate particle stores -(plate index + 1) as its density.
bool is_thin_plate_point(float stored_density) {
    return stored_density < 0.0;
}

// Slot of the plate a quadrature point belongs to; MAX_NEAR_THIN_PLATES if the
// plate is not in the near list.
uint thin_plate_slot_of_point(float stored_density) {
    uint plate_index = uint(-stored_density - 0.5);
    for (uint slot = 0u; slot < near_thin_plate_count; slot++) {
        if (near_thin_plate[slot].index == plate_index) return slot;
    }
    return MAX_NEAR_THIN_PLATES;
}

// Faces of the quadrature point `point_position` (on the mid-plane of the near
// plate `slot`) that the particle at `viewer` sees. face 0 is the face on the
// viewer's own side, if it is seen at all.
uint thin_plate_visible_faces(uint slot, vec3 point_position, vec3 viewer,
                              out vec3 face_position[2], out vec3 face_normal_out[2],
                              out bool face_is_own_side[2]) {
    vec3  normal          = near_thin_plate[slot].normal;
    vec3  centre          = near_thin_plate[slot].centre;
    float half_thickness  = near_thin_plate[slot].half_thickness;
    float viewer_distance = dot(normal, viewer - centre);
    float own_side        = (viewer_distance >= 0.0) ? 1.0 : -1.0;
    uint  face_count      = 0u;
    for (uint candidate = 0u; candidate < 2u; candidate++) {
        float side     = (candidate == 0u) ? own_side : -own_side;
        vec3  position = point_position + side * half_thickness * normal;
        vec3  offset   = position - viewer;
        float distance_squared = dot(offset, offset);
        if (distance_squared >= smoothing_length_squared() || distance_squared < 1e-24) continue;

        bool visible = true;
        if (viewer_distance * side <= 0.0) {
            // the viewer is on the other side of the mid-plane (or on it): the
            // face is seen only past the edge of the plate
            float face_distance = side * half_thickness;
            float denominator   = viewer_distance - face_distance;
            float fraction      = (abs(denominator) > 1e-12) ? viewer_distance / denominator : 0.0;
            vec3  crossing      = viewer + fraction * offset - centre;
            visible = !thin_plate_outline_contains(near_thin_plate[slot].shape,
                                                   near_thin_plate[slot].extent_a,
                                                   near_thin_plate[slot].extent_b,
                                                   dot(crossing, near_thin_plate[slot].axis_a),
                                                   dot(crossing, near_thin_plate[slot].axis_b));
        }
        for (uint other = 0u; visible && other < near_thin_plate_count; other++) {
            if (other == slot) continue;
            if (thin_plate_slot_blocks_segment(other, viewer, position)) visible = false;
        }
        if (!visible) continue;
        face_position[face_count]    = position;
        face_normal_out[face_count]  = -side * normal;
        face_is_own_side[face_count] = (viewer_distance * side > 0.0);
        face_count++;
    }
    return face_count;
}

// Weight of a quadrature point in the sums of a particle with the calibrated
// volume V_p: measure * V_p / dx^d. The correction matrix carries the inverse
// factor dx^d / V_p, so that B_i n W w is the surface term of the true area.
float thin_plate_point_weight(uint slot, float particle_volume, float particle_radius) {
    float spacing = 2.0 * particle_radius;
    float cell    = (DIMENSION == 3u) ? spacing * spacing * spacing : spacing * spacing;
    return near_thin_plate[slot].point_measure * particle_volume / cell;
}

// Acceleration of a point of the plate (centripetal for a rotor plate).
vec3 thin_plate_point_acceleration(uint slot, vec3 position) {
    if (near_thin_plate[slot].frame != THIN_PLATE_FRAME_ROTOR) return vec3(0.0);
    return solid_particle_acceleration(MATERIAL_ROTOR, position);
}

// Pressure on the face as the fluid particle sees it.
float thin_plate_wall_pressure(uint slot, float fluid_pressure, float fluid_density,
                               vec3 fluid_position, vec3 fluid_velocity,
                               vec3 face_position, vec3 face_normal_out, vec3 point_velocity,
                               bool face_is_own_side, float rest_density, float particle_radius) {
    vec3 relative_body_force = vec3(GRAVITY_X, GRAVITY_Y, GRAVITY_Z)
                             - thin_plate_point_acceleration(slot, face_position);
    // penalty: the particle stands over the plate, closer than a quarter spacing
    // to the face on its own side (or between that face and the mid-plane)
    float approach = 0.0;
    if (face_is_own_side) {
        vec3 relative = fluid_position - near_thin_plate[slot].centre;
        if (thin_plate_outline_contains(near_thin_plate[slot].shape,
                                        near_thin_plate[slot].extent_a, near_thin_plate[slot].extent_b,
                                        dot(relative, near_thin_plate[slot].axis_a),
                                        dot(relative, near_thin_plate[slot].axis_b))) {
            float gap       = dot(fluid_position - face_position, -face_normal_out);
            float gap_limit = 0.5 * particle_radius;
            approach = clamp(1.0 - gap / gap_limit, 0.0, 4.0);
        }
    }
    return fluid_pressure
         + fluid_density * dot(relative_body_force, face_position - fluid_position)
         + THIN_PLATE_DASHPOT * fluid_density * SPEED_OF_SOUND
           * dot(fluid_velocity - point_velocity, face_normal_out)
         + THIN_PLATE_PENALTY * rest_density * SPEED_OF_SOUND * SPEED_OF_SOUND * approach * approach;
}

// Shift of a particle next to a plate: remove the part that points into the
// plate, for every near plate the particle stands over at less than half a
// spacing from the face.
vec3 thin_plate_limit_shift(vec3 shift, vec3 position, float particle_radius) {
    for (uint slot = 0u; slot < near_thin_plate_count; slot++) {
        vec3  relative = position - near_thin_plate[slot].centre;
        float distance_normal = dot(relative, near_thin_plate[slot].normal);
        if (abs(distance_normal) - near_thin_plate[slot].half_thickness >= particle_radius) continue;
        if (!thin_plate_outline_contains(near_thin_plate[slot].shape,
                                         near_thin_plate[slot].extent_a, near_thin_plate[slot].extent_b,
                                         dot(relative, near_thin_plate[slot].axis_a),
                                         dot(relative, near_thin_plate[slot].axis_b))) continue;
        vec3  away = ((distance_normal >= 0.0) ? 1.0 : -1.0) * near_thin_plate[slot].normal;
        float toward_plate = -dot(shift, away);
        if (toward_plate > 0.0) shift += toward_plate * away;
    }
    return shift;
}

#endif  // SPH_THIN_PLATES_GLSL_INCLUDED
