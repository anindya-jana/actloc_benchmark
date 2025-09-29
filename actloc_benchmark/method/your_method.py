import numpy as np
from typing import Dict, List
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree

def vector_to_level_quaternion(
    vec: np.ndarray,
    world_up: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float64),
) -> np.ndarray:
    """
    Convert a 3D direction vector into a level (horizontal) camera orientation quaternion (x, y, z, w).
    """
    direction = np.array([vec[0], vec[1], 0.0], dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    fwd = direction / norm
    up = world_up / np.linalg.norm(world_up)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up_cam = np.cross(right, fwd)
    R_mat = np.stack([right, up_cam, fwd], axis=1)
    if np.linalg.det(R_mat) < 0:
        R_mat[:, 1] *= -1.0
    return R.from_matrix(R_mat).as_quat()


def generate_smooth_and_aware_orientations(
    waypoints: Dict[str, np.ndarray],
    points3D: Dict[int, object],
    path_clearance: float = 0.5,
    vertical_tolerance: float = 0.75,
    max_tilt_angle: float = 30.0,
    smoothing_window: int = 4,
    smoothing_strength: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Generates ultra-smooth, predictive camera orientations. It looks several
    waypoints ahead to anticipate turns smoothly. Obstacle detection overrides
    this behavior with a dynamic downward tilt.
    """
    if not waypoints:
        return {}

    ids = list(waypoints.keys())
    n = len(ids)
    final_orientations = {}

    # Step 1: Calculate new ultra-smooth default orientations using a predictive window
    weights = np.array([smoothing_strength**i for i in range(smoothing_window)])
    
    for i, wid in enumerate(ids):
        # --- NEW SMOOTHING LOGIC ---
        weighted_vectors = []
        for j in range(smoothing_window):
            # Look ahead from the current point i
            if i + j + 1 < n:
                p1 = waypoints[ids[i + j]]
                p2 = waypoints[ids[i + j + 1]]
                direction_vec = p2 - p1
                norm = np.linalg.norm(direction_vec)
                if norm > 1e-6:
                    # Add the weighted, normalized direction vector
                    weighted_vectors.append(direction_vec / norm * weights[j])

        if weighted_vectors:
            # Average the vectors to get a smooth, forward-looking direction
            smooth_vec = np.sum(weighted_vectors, axis=0)
        else:
            # Fallback for the last few waypoints: look back from the previous one
            if i > 0:
                smooth_vec = waypoints[wid] - waypoints[ids[i - 1]]
            else: # Fallback for a single waypoint
                smooth_vec = np.array([1.0, 0.0, 0.0])

        final_orientations[wid] = vector_to_level_quaternion(smooth_vec)
        # --- END OF NEW SMOOTHING LOGIC ---

    # Step 2: Build KD-Tree (no change)
    point_xyz_list = [points3D[pid].xyz for pid in points3D if points3D[pid]]
    if not point_xyz_list:
        return final_orientations
    all_points_xyz = np.array(point_xyz_list)
    kdtree = KDTree(all_points_xyz)
    path_clearance_sq = path_clearance ** 2

    # Step 3: Check for obstacles and apply dynamic tilt override (no change)
    for i, wid in enumerate(ids):
        if i >= n - 1: continue
        start_node, end_node = waypoints[wid], waypoints[ids[i + 1]]
        vec_path = end_node - start_node
        path_len_sq = np.dot(vec_path, vec_path)
        if path_len_sq < 1e-8: continue
        search_radius = np.sqrt(path_len_sq) + path_clearance
        nearby_indices = kdtree.query_ball_point(start_node, r=search_radius)
        if not nearby_indices: continue
            
        obstacle_on_path, detected_obstacle_pos = False, None
        for idx in nearby_indices:
            point_pos = all_points_xyz[idx]
            vec_to_point = point_pos - start_node
            t = np.dot(vec_to_point, vec_path) / path_len_sq
            if not (0.0 <= t <= 1.0): continue
            if abs(vec_to_point[2]) > vertical_tolerance: continue
            projection_point = start_node + t * vec_path
            dist_vector = point_pos - projection_point
            dist_vector[2] = 0
            if np.dot(dist_vector, dist_vector) < path_clearance_sq:
                obstacle_on_path, detected_obstacle_pos = True, point_pos
                break
        
        if obstacle_on_path:
            # The override behavior remains the same
            if i < n - 2:
                print(f"INFO: Obstacle near '{wid}'. Looking ahead and tilting down.")
                next_next_pos = waypoints[ids[i + 2]]
                vec_to_next_next = next_next_pos - start_node
                base_yaw_quat = vector_to_level_quaternion(vec_to_next_next)
                distance_to_obstacle = np.linalg.norm(detected_obstacle_pos - start_node)
                clamped_dist = np.clip(distance_to_obstacle, 0, path_clearance)
                tilt_ratio = 1.0 - (clamped_dist / path_clearance)
                dynamic_tilt_angle_deg = max_tilt_angle * tilt_ratio
                base_yaw_rotation = R.from_quat(base_yaw_quat)
                tilt_rotation = R.from_euler('x', -dynamic_tilt_angle_deg, degrees=True)
                final_orientations[wid] = (base_yaw_rotation * tilt_rotation).as_quat()

    return final_orientations