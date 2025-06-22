import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import collections
State = collections.namedtuple("State", ["tsdf", "pc"])
import open3d as o3d
import copy
import time
from scipy.spatial.transform import Rotation as R


def get_perpendicular_distance(A, B, P):
    # Vector from A to B
    AB = B - A
    # Vector from A to P
    AP = P - A
    # Cross product of AB and AP for each point
    cross_product = np.cross(AB, AP)

    # Magnitude of the cross product (norm across the last axis)
    cross_product_magnitude = np.linalg.norm(cross_product, axis=1)  # Shape: (N,)

    # Magnitude of the vector AB (norm across the last axis)
    AB_magnitude = np.linalg.norm(AB, axis=1)  # Shape: (N,)

    # Perpendicular distance from point P to the line AB for each point
    perpendicular_distance = cross_product_magnitude / AB_magnitude  # Shape: (N,)

    return perpendicular_distance


def calculate_stability_score(box, grasps, alpha=0.5):
    bc = box.get_center()
    grasps_transformation_matrix = grasps[:, :16].reshape(grasps.shape[0], 4, 4)
    grasps_pos = grasps_transformation_matrix[:, :3, 3]
    grasps_width = grasps[:, -2]
    grasps_depth = grasps[:, -1]

    # initialize the gripper points
    left_gripper_points = np.zeros((grasps.shape[0], 4))
    right_gripper_points = np.zeros((grasps.shape[0], 4))
    left_gripper_points[:, 0] = -grasps_width/2
    right_gripper_points[:, 0] = +grasps_width/2
    left_gripper_points[:, 2] = grasps_depth
    right_gripper_points[:, 2] = grasps_depth
    left_gripper_points[:, 3] = 1
    right_gripper_points[:, 3] = 1

    transformed_left_gripper_points = np.einsum('ijk,ik->ij', grasps_transformation_matrix, left_gripper_points)
    transformed_right_gripper_points = np.einsum('ijk,ik->ij', grasps_transformation_matrix, right_gripper_points)

    d1_list = get_perpendicular_distance(transformed_left_gripper_points[:, :3], transformed_right_gripper_points[:, :3], bc)
    d2_list = np.linalg.norm(grasps_pos - bc, axis=1)
    l_diag = np.linalg.norm(box.get_max_bound() - box.get_min_bound())

    p_dists = alpha * (1-(d1_list/(l_diag/2))) + (1-alpha) * (1-(d2_list/(l_diag/2)))
    return p_dists

def vizualize_grasps_o3d(Grasps, pc, transformed_box=None, original_box=None):
    if transformed_box is None:
        transformed_box = []
    else:
        transformed_box = [transformed_box]
    if original_box is None:
        original_box = []
    else:
        original_box.color = [1, 0, 0]
        original_box = [original_box]
    debug_box = []
    debug_ball = []
    debug_coord = []
    for i in range(Grasps.shape[0]):
        trans_matrix = Grasps[i,:16].reshape(4, 4)
        width, depth = Grasps[i, -2], Grasps[i, -1]

        ball, bar, coord = get_o3d_gripper(trans_matrix, width)
        debug_coord.append(coord)
        debug_box.append(bar)
        debug_ball.append(ball)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([origin, pc]+debug_box+debug_ball+debug_coord+original_box+transformed_box, window_name='Point Cloud', width=800, height=600, left=50)

def get_o3d_gripper(transform, width):
    bar = o3d.geometry.TriangleMesh.create_box(width=0.01, height=width, depth=0.01)
    bar.translate(-bar.get_center())
    bar.transform(transform)
    bar.paint_uniform_color([1., 0., 0.])  # RGB values between 0 and 1

    ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    ball.transform(transform)
    ball.paint_uniform_color([0., 1., 0.])

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    coord.transform(transform)

    return bar, ball, coord

def normalize_array(arr):
    return arr / np.linalg.norm(arr)



def transform_to_origin(box):
    transformed_box = o3d.geometry.OrientedBoundingBox(box)
    rotation = transformed_box.R.copy()
    rot_center = transformed_box.get_center().copy()
    transformed_box = transformed_box.rotate(np.linalg.inv(rotation), center=transformed_box.get_center())
    translation = transformed_box.get_min_bound()
    transformed_box = transformed_box.translate(-translation)
    transformation_rot = np.eye(4)
    transformation_rot_center = np.eye(4)
    transformation_trans = np.eye(4)
    transformation_rot[:3, :3] = rotation
    transformation_rot_center[:3, 3] = rot_center
    transformation_trans[:3, 3] = translation
    transformation_matrix = transformation_rot_center @ transformation_rot @ np.linalg.inv(transformation_rot_center) @ transformation_trans
    return transformed_box, transformation_matrix, translation, rotation


# def GoalGrasp_algorithm(box, pc,  sample_num=5, debug=True):
#     G = []
#     #store original bounding box
#     original_box = o3d.geometry.OrientedBoundingBox(box)
#     original_box.color = [0, 0, 1]
#
#     #transform the box to the origin for easier calculations
#     transformed_box, original_transformation_matrix, original_translation, original_rotation = transform_to_origin(box)
#     transformed_box.color = [0, 1, 0]
#     max_bounds = transformed_box.get_max_bound()
#     length = max_bounds
#     CT = np.tile((length/2).reshape(1, -1), (4, 1))
#     gd = 0.06
#     st_idex = ["z", "z", "z", "z", "y", "y", "y", "y", "x", "x", "x", "x"]
#     ST = np.array([[CT[0, 0], 0, length[2]],
#                    [CT[0, 0], max_bounds[1], length[2]],
#                    [0, CT[0, 1], length[2]],
#                    [max_bounds[0], CT[0, 1], length[2]],
#                    [CT[0, 0], length[1], 0],
#                    [CT[0, 0], length[1], max_bounds[2]],
#                    [0, length[1], CT[0, 2]],
#                    [max_bounds[0], length[1], CT[0, 2]],
#                    [length[0], CT[0, 1], 0],
#                    [length[0], CT[0, 1], max_bounds[2]],
#                    [length[0], 0, CT[0, 2]],
#                    [length[0], max_bounds[1], CT[0, 2]]])
#
#     ST_x = np.array([[length[0], CT[0, 1], 0],
#                    [length[0], CT[0, 1], max_bounds[2]],
#                    [length[0], 0, CT[0, 2]],
#                    [length[0], max_bounds[1], CT[0, 2]]])
#
#     ST_y = np.array([[CT[0, 0], length[1], 0],
#                     [CT[0, 0], length[1], max_bounds[2]],
#                     [0, length[1], CT[0, 2]],
#                     [max_bounds[0], length[1], CT[0, 2]]])
#
#     ST_z = np.array([[CT[0, 0], 0, length[2]],
#                    [CT[0, 0], max_bounds[1], length[2]],
#                    [0, CT[0, 1], length[2]],
#                    [max_bounds[0], CT[0, 1], length[2]]])
#
#     unit_vectors = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
#                              [0, -1, 0], [0, 0, 1], [0, 0, -1]])
#
#     p_x = np.hstack((np.random.uniform(0, length[0], (4, 1)), ST_x[:, 1:3]))
#     p_y = np.hstack((np.random.uniform(0, length[1], (4, 1)), ST_y[:, :2]))
#     p_z = np.hstack((np.random.uniform(0, length[2], (4, 1)), ST_z[:, :2]))
#
#     X_x = np.stack([p_x[:, 0], CT[:, 1], CT[:, 2]], axis=-1)
#     X_y = np.stack([CT[:, 0], p_y[:, 0], CT[:, 2]], axis=-1)
#     X_z = np.stack([CT[:, 0], CT[:, 1], p_z[:, 0]], axis=-1)
#
#     X_x = normalize_array(X_x)
#     X_y = normalize_array(X_y)
#     X_z = normalize_array(X_z)
#     print("X_x", X_x)
#
#     Y_x = np.cross(X_x, np.array([1, 0, 0]).T)
#     Y_y = np.cross(X_y, np.array([0, 1, 0]).T)
#     Y_z = np.cross(X_z, np.array([0, 0, 1]).T)
#
#     width_map = np.array([length[0], length[0], length[1], length[1], length[2], length[2]])
#     print("y_x", Y_x)
#     matches_x = (Y_x[:, None] == unit_vectors).all(-1)
#     w_x = np.dot(matches_x, width_map)
#     print(w_x)
#     print(p_x.shape, X_x.shape, CT.shape)
#
#     max_height = 0
#     for i, st in enumerate(ST):
#         for n in range(sample_num):
#             # generate rotation and translation for n in ST
#             if st_idex[i] == "x":
#                 p = np.array([np.random.uniform(0, length[0]), st[1], st[2]])
#                 X = np.array([p[0], CT[1], CT[2]]) - p
#                 X = normalize_array(X.T)
#                 Y = np.cross(X, np.array([1, 0, 0]).T)
#             elif st_idex[i] == "y":
#                 p = np.array([st[0], np.random.uniform(0, length[1]), st[2]])
#                 X = np.array([CT[0], p[1], CT[2]]) - p
#                 X = normalize_array(X.T)
#                 Y = np.cross(X, np.array([0, 1, 0]).T)
#             else:
#                 p = np.array([st[0], st[1],  np.random.uniform(0, length[2])])
#                 X = np.array([CT[0], CT[1], p[2]]) - p
#                 X = normalize_array(X.T)
#                 Y = np.cross(X, np.array([0, 0, 1]).T)
#
#             #get width of grasp
#             if np.array_equal(Y, np.array([1, 0, 0])) or np.array_equal(Y, np.array([-1, 0, 0])):
#                 w = length[0]
#             elif np.array_equal(Y, np.array([0, 1, 0])) or np.array_equal(Y, np.array([0, -1, 0])):
#                 w = length[1]
#             else:
#                 w = length[2]
#
#             # get depth of grasp
#             d = min(length[0]/2, length[1]/2, length[2]/2, gd)
#
#             #calculate random pitch angle
#             pitch_angle = np.random.uniform(-np.pi/4, np.pi/4)
#             pitch_rotation = np.array([
#                 [np.cos(pitch_angle), 0, np.sin(pitch_angle)],
#                 [0, 1, 0],
#                 [-np.sin(pitch_angle), 0, np.cos(pitch_angle)]
#             ])
#
#             #calculate transformation matrix
#             transformation_matrix = np.eye(4)
#             transformation_matrix[:3, 3] = p
#             transformation_matrix[:3, :3] = np.array([X, Y, np.cross(X, Y)]).T @ pitch_rotation
#             transformation_matrix = original_transformation_matrix @ transformation_matrix
#
#             if transformation_matrix[2,3] > max_height:
#                 max_height = transformation_matrix[2,3]
#             G.append(np.append(transformation_matrix.flatten(), np.array([w, d])))
#     if debug:
#         vizualize_grasps_o3d(G, pc, transformed_box, original_box)
#     return np.array(G), max_height


def GoalGrasp_algorithm(box, pc,  sample_num=5, debug=True):
    G = []
    #store original bounding box
    original_box = o3d.geometry.OrientedBoundingBox(box)
    original_box.color = [0, 0, 1]

    #transform the box to the origin for easier calculations
    transformed_box, original_transformation_matrix, original_translation, original_rotation = transform_to_origin(box)
    transformed_box.color = [0, 1, 0]
    max_bounds = transformed_box.get_max_bound()
    length = max_bounds
    CT = length/2
    gd = 0.06
    st_idex = ["z", "z", "z", "z", "y", "y", "y", "y", "x", "x", "x", "x"]
    ST = np.array([[CT[0], 0, length[2]],
                   [CT[0], max_bounds[1], length[2]],
                   [0, CT[1], length[2]],
                   [max_bounds[0], CT[1], length[2]],
                   [CT[0], length[1], 0],
                   [CT[0], length[1], max_bounds[2]],
                   [0, length[1], CT[2]],
                   [max_bounds[0], length[1], CT[2]],
                   [length[0], CT[1], 0],
                   [length[0], CT[1], max_bounds[2]],
                   [length[0], 0, CT[2]],
                   [length[0], max_bounds[1], CT[2]]])
    max_height = 0
    for i, st in enumerate(ST):
        for n in range(sample_num):
            # generate rotation and translation for n in ST
            if st_idex[i] == "x":
                p = np.array([np.random.uniform(0, length[0]), st[1], st[2]])
                X = np.array([p[0], CT[1], CT[2]]) - p
                X = normalize_array(X.T)
                Y = np.cross(X, np.array([1, 0, 0]).T)
            elif st_idex[i] == "y":
                p = np.array([st[0], np.random.uniform(0, length[1]), st[2]])
                X = np.array([CT[0], p[1], CT[2]]) - p
                X = normalize_array(X.T)
                Y = np.cross(X, np.array([0, 1, 0]).T)
            else:
                p = np.array([st[0], st[1],  np.random.uniform(0, length[2])])
                X = np.array([CT[0], CT[1], p[2]]) - p
                X = normalize_array(X.T)
                Y = np.cross(X, np.array([0, 0, 1]).T)

            #get width of grasp
            if np.array_equal(Y, np.array([1, 0, 0])) or np.array_equal(Y, np.array([-1, 0, 0])):
                w = length[0]
            elif np.array_equal(Y, np.array([0, 1, 0])) or np.array_equal(Y, np.array([0, -1, 0])):
                w = length[1]
            else:
                w = length[2]

            # get depth of grasp
            d = min(length[0]/2, length[1]/2, length[2]/2, gd)

            #calculate random pitch angle
            pitch_angle = np.random.uniform(-np.pi/4, np.pi/4)
            pitch_rotation = np.array([
                [np.cos(pitch_angle), 0, np.sin(pitch_angle)],
                [0, 1, 0],
                [-np.sin(pitch_angle), 0, np.cos(pitch_angle)]
            ])

            #calculate transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, 3] = p
            transformation_matrix[:3, :3] = np.array([X, Y, np.cross(X, Y)]).T @ pitch_rotation
            transformation_matrix = original_transformation_matrix @ transformation_matrix

            if transformation_matrix[2,3] > max_height:
                max_height = transformation_matrix[2,3]
            G.append(np.append(transformation_matrix.flatten(), np.array([w, d])))
    if debug:
        vizualize_grasps_o3d(G, pc, transformed_box, original_box)
    return np.array(G), max_height


def point_to_obb_distance(point, obb):
    # Transform the point to the local coordinate system of the OBB
    local_point = np.dot(obb.R.T, point - obb.center)
    # Clamp the local point to the bounds of the OBB
    clamped_local_point = np.maximum(-obb.extent / 2, np.minimum(local_point, obb.extent / 2))
    # Transform the clamped point back to the global coordinate system
    clamped_point = np.dot(obb.R, clamped_local_point) + obb.center
    # Compute the Euclidean distance
    distance = np.linalg.norm(clamped_point - point)
    return distance

def get_grasps(obb_list, pc_list, frontier_list, frontier_map, env):
    final_grasps = np.zeros((0, 18))
    final_scores = np.zeros((0, 1))

    for i in range(len(obb_list)):
        grasps, max_height = GoalGrasp_algorithm(obb_list[i], pc_list[i], sample_num=10, debug=False)
        print(grasps.shape)
        gripper_width_check = grasps[:,-2] > 0.085
        object_height_check = grasps[:,11] < 0.95
        object_height_check_2 = grasps[:,11] + 0.03 >= max_height
        wall_check = grasps[:, 3] > 0.32
        wall_check_2 = grasps[:, 3] < -0.32

        map_grasps = env.mapping.hg.world_point_to_map_point(grasps[:, :16].reshape(grasps.shape[0], 4, 4)[:, :3, 3])
        map_grasps[:, 0] = frontier_map.shape[1] - map_grasps[:, 0]

        # calculate y condition
        y_map = map_grasps[:, 0].reshape(-1, 1)
        y_frontier = frontier_list[:, 1].reshape(1, -1)
        # calculate z condition
        z_map = map_grasps[:, 2].reshape(-1, 1)
        z_frontier = frontier_list[:, 2].reshape(1, -1)
        y_z_condition = np.logical_and(np.logical_and(y_map < y_frontier + 5, y_map > y_frontier - 5),
                                       z_map < z_frontier)
        x_map = map_grasps[:, 1].reshape(-1, 1)
        x_frontier = frontier_list[:, 0].reshape(1, -1)
        x_y_z_condition = np.logical_and(y_z_condition, x_map > x_frontier)
        frontier_conditions = np.any(x_y_z_condition, axis=1)

        mask = np.logical_or(np.logical_or(np.logical_or(gripper_width_check, frontier_conditions), np.logical_or(object_height_check, object_height_check_2)), np.logical_or(wall_check, wall_check_2))
        filtered_grasps = grasps[np.invert(mask)]
        final_grasps = np.concatenate((final_grasps, filtered_grasps), axis=0)

        scores = calculate_stability_score(obb_list[i], final_grasps).reshape(-1, 1)
        final_scores = np.concatenate((final_scores, scores), axis=0)

    scores = calculate_stability_score(obb_list[i], final_grasps)
    return final_grasps, scores

def test_grasp_feasability(env, start_config, target_pos, target_ori=None):
    if target_ori is None:
        target_ori = env.init_ori
    start_goal_array = np.zeros((2, len(start_config)))
    env.klampt_utils.set_joint_positions(start_config)
    if len(target_pos) == 3:
        target_arm_joint_pos = env.get_ik_joints(target_pos[:3], target_ori, link=env.tool_tip_id)
        target_joint_config = copy.deepcopy(start_config)
        target_joint_config[:6] = target_arm_joint_pos
    else:
        target_joint_config = target_pos

    path = env.klampt_utils.plan_to_joint_goal(target_joint_config, verbose=False)

    if path is not None:
        start_goal_array[0] = start_config
        start_goal_array[1] = target_joint_config
        return target_joint_config, path, target_pos, start_goal_array
    return None, None, None, None

def get_feasable_grasps(env, grasps, scores):
    feasible_grasps = np.zeros((0, 18))
    feasible_scores = np.zeros((0, 1))
    current_joint_config = env.get_current_arm_and_gripper_joint_config()
    feasible_start_goal_configs = np.zeros((0, 2, len(current_joint_config)))
    feasible_paths = []
    feasible_homing_paths = []
    for i in range(grasps.shape[0]):
        current_joint_config = env.get_current_arm_and_gripper_joint_config()
        grasp_pos = grasps[i, :16].reshape(4, 4)[:3, 3]
        grasp_ori = R.from_matrix(grasps[i, :16].reshape(4, 4)[:3, :3]).as_quat()
        start_arm_joint_config, path, start_pose, start_goal_configs = test_grasp_feasability(env, current_joint_config, grasp_pos)
        if path is not None:
            feasible_start_goal_configs = np.concatenate((feasible_start_goal_configs, np.expand_dims(start_goal_configs, axis=0)), axis=0)
            grasp_pos[1] += 0.06
            end_arm_joint_config, end_path, end_pose, _ = test_grasp_feasability(env, start_arm_joint_config, grasp_pos)
            if end_path is not None:
                path.pop()
                path.extend(end_path)
                grasp_pos[2] += 0.03
                lift_arm_joint_config, lift_path, lift_pose, _ = test_grasp_feasability(env, end_arm_joint_config, grasp_pos)
                if lift_path is not None:
                    home_arm_joint_config, home_path, home_pose, _ = test_grasp_feasability(env, lift_arm_joint_config, current_joint_config)
                    if home_path is not None:
                        lift_path.pop()
                        lift_path.extend(home_path)

                feasible_grasps = np.concatenate((feasible_grasps, grasps[i].reshape(1, -1)), axis=0)
                feasible_scores = np.concatenate((feasible_scores, scores[i].reshape(1, -1)), axis=0)
                feasible_paths.append(path)
                feasible_homing_paths.append(lift_path)
    np.save("feasible_start_goal_configs.npy", feasible_start_goal_configs)
    return feasible_grasps, feasible_scores, feasible_paths, feasible_homing_paths

def execute_grasp(env, path, homing_path, grasp_width):
    #open gripper
    env.move_gripper(0.085)

    #execute path
    moved_positions = []
    mps = env.linear_interpolate_motion_klampt_joint_traj(path, verbose=False)
    moved_positions.extend(mps)

    #close gripper
    env.move_gripper(grasp_width)

    #find closest object in Gripper and add as constraint for grasping
    closest_obj = None
    closest_dist = np.inf
    for o in env.current_obj_ids:
        closest_points = env._p.getClosestPoints(bodyA=env.robot_id,
                                                 bodyB=o,
                                                 distance=.3,
                                                 linkIndexA=env.grasp_link_id,
                                                 linkIndexB=-1)
        if closest_points:
            c = closest_points[0]
            if c[8] < closest_dist:
                closest_obj = c
                closest_dist = c[8]

    env.constraint_list.append(env.add_constraint_to_gripper(closest_obj[2]))

    #retrieve object
    #reversed_path = path.copy()
    #reversed_path.reverse()
    mps = env.linear_interpolate_motion_klampt_joint_traj(homing_path, verbose=False)
    moved_positions.extend(mps)
    return moved_positions
