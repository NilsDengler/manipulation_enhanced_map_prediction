import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cupy as cp
import cv2


class FrontierFinder:
    def __init__(self):
        self.idx = None

    def find_frontiers(self, voxel_map, force_resample_idx=False, pushing_method="MEM"):
        # Ensure the index array is initialized or resampled
        if self.idx is None or force_resample_idx:
            self.idx = cp.indices(voxel_map.shape, dtype=cp.uint16)

        # Create binary height map threshold
        voxel_map = cp.asarray(voxel_map)
        voxel_map_binary = voxel_map > 0.7
        cumulative_height = (cp.cumsum(voxel_map_binary, axis=0) > 0).astype(cp.uint8)

        # Compute frontier voxels
        frontier_voxels = cp.zeros(voxel_map.shape, dtype=cp.uint8)
        frontier_voxels[1:] = cumulative_height[1:] - cumulative_height[:-1]

        # Propagate values along depth axis
        frontier_voxels = (cp.cumsum(frontier_voxels[:, :, ::-1], axis=2)[:, :, ::-1] > 0).astype(cp.uint8)

        # Compute final frontier map
        frontier_voxels[:, :, :-1] = frontier_voxels[:, :, 1:] - frontier_voxels[:, :, :-1]
        frontier_voxels = frontier_voxels > 0

        # Apply boundary restrictions based on pushing method
        if pushing_method == "MEM":
            frontier_voxels[:, :, :5] = False
            frontier_voxels[115:, :, :] = False
            frontier_voxels[:, :23, :] = False
            frontier_voxels[:, -22:, :] = False
        else:
            frontier_voxels[:, :, :13] = False
            frontier_voxels[:, :54, :] = False
            frontier_voxels[:, -55:, :] = False
            frontier_voxels[425:, :, :] = False
            frontier_voxels[:142, :, :] = False

        # Extract frontier coordinates
        frontier_coords = self.idx[:, frontier_voxels]

        cp.cuda.runtime.deviceSynchronize()
        return frontier_coords.transpose()


    def filter_tall_frontiers(self, occupied_indices):
        """
        Filters out short frontiers, keeping only tall objects.
        """
        max_occupied_height = occupied_indices.max(axis=0)[-1]
        return occupied_indices[occupied_indices[:, 1] >= max_occupied_height / 2]


class PushSampler:
    def __init__(self):
        self.a = None
        self.no = None
        self.idxs = None
        self.ff = FrontierFinder()
        self.rng = np.random.default_rng()
        self.max_push_distance = 20 # in map voxels i.e. 20*0.005 = 10 cm
        self.min_push_distance = 5 # in map voxels i.e. 5*0.005 = 2.5 cm
        self.z_in_pix = (cp.floor((0.97 - 0.91) / 0.005)).astype(cp.int32)


    def perform_single_pushing(self, env, occ_map, samples=None, execute=False, verbose=False):
        '''
        Perform pushing on the given occupancy map
        Args:
            env: Environment object for klampt usage
            occ_map: Occupancy map
            execute: Execute the pushing or not
        Returns:
            path_data: Dictionary containing the paths, path annotations, motion parametrization and link poses
        '''

        push_data = self.get_samples(env, occ_map, samples, 100, just_endpoints=True, verbose=verbose)
        motion_parametrization = np.array([int(x) if isinstance(x, cp.ndarray) else x for x in push_data['motion_parametrization']]).reshape(-1, 6)

        if motion_parametrization.shape[0] == 0:
            return None, None

        # get motion parameters in form of cartesian joint positions
        mps = env.linear_interpolate_motion_klampt_joint_traj(push_data['paths'][0],
                                                              traj_annotation=push_data['path_annotations'][0],
                                                              imagined=True, verbose=False)

        # Calculate push parametrization in form of Swept volume
        sv = np.moveaxis(env.smg.get_swept_map(mps[::2])[10:-10], [2, 0, 1], [0, 1, 2])

        if execute:
            success, mp = execute_push(env, push_data['paths'][0], path_annotations=push_data['path_annotations'][0])
            print(f'Pushing success: {not bool(success)}')

        return sv, mps



    def get_samples(self, env, occ_map, samples=None, num_points=2, force_resample_idx=False, just_endpoints=False,  verbose=False):
        """
        Generate random samples for pushing.
        Args:
            env: Environment object for klampt usage
            occ_map: Occupancy map
            num_points: Number of points to sample
            force_resample_idx:
            just_endpoints: return just endpoints in path (no interpolation)
        Return:
            dict:
                'paths': list of feasible paths
                'path_annotations': list of annotations for each path ("free", "occupied", "pushing")
                'motion_parametrization': list of motion parametrization for each path (start, end points)
        """
        #preprocess the occupancy map
        voxel_map = occ_map.copy()
        voxel_map[voxel_map < 0.85] = 0
        heighest_cell = np.max(np.argwhere(voxel_map >= 0.85)[:,2])
        voxel_map[:,:,int(heighest_cell/2)] = 0

        if samples is not None:
            sampled_indices = samples[:, 0, :]
            sampled_indices = np.hstack([sampled_indices, np.full((samples.shape[0], 1), 44)])
            new_sampling = True
        else:
            #find frontiers in map
            occupied_indices = self.ff.find_frontiers(voxel_map, force_resample_idx)
            occupied_indices = self.ff.filter_tall_frontiers(occupied_indices)

            sampled_indices = cp.random.choice(occupied_indices.shape[0], num_points, replace=False)
            sampled_indices = occupied_indices[sampled_indices]
            new_sampling = False

        current_joint_config = env.get_current_arm_and_gripper_joint_config()

        #add heightmap to klampt for feasibility check
        if env.klampt_utils.tm_rb_name is None:
            env.klampt_utils.add_heightmap_to_klampt(cp.asnumpy(voxel_map[:,:,5:]))
        else:
            env.klampt_utils.delete_heightmap_from_klampt()
            env.klampt_utils.add_heightmap_to_klampt(voxel_map[:,:,5:])

        #sample start points
        start_data = {'start_arm_joint_configs': [], 'path_to_start_positions': [], 'start_poses': [], 'all_start_indices': []}
        start_data = self.find_start_points(start_data, current_joint_config, env, sampled_indices, just_endpoints=just_endpoints, verbose=verbose,new_sampling = new_sampling)
        env.klampt_utils.delete_heightmap_from_klampt()

        #return feasible start-end point combos
        final_data = {'paths': [], 'path_annotations': [], 'motion_parametrization': []}
        return self.find_end_points(final_data, start_data, env, voxel_map, samples=samples, just_endpoints=just_endpoints, verbose=verbose)


    def  find_start_points(self, data, initial_joint_config, env, frontier_indices, just_endpoints=False, verbose=False, new_sampling=False):
        ''' Sample start poses from given frontier indices
        Args:
            data: Empty dictionary containing the start poses, arm joint configurations, path to start positions and all start indices
            initial_joint_config: Initial joint configuration of the robot
            env: Environment object for klampt usage
            frontier_indices: List of frontier indices
            just_endpoints: return just endpoints in path (no interpolation)
        Return:
            dict: Dictionary containing the start poses, arm joint configurations, path to start positions and all start indices
        '''

        for i in tqdm(range(len(frontier_indices)), disable=True):
            start_indices = frontier_indices[i]
            to_transform_indices = map_index_to_world_voxel_index(start_indices)

            #start_pose = envtarget.mapping.hg.map_point_to_world_point(np.append(to_transform_indices[:2], self.z_in_pix))
            if new_sampling:
                start_indices[1] = start_indices[1] + 10
                start_indices[1] = 140 - start_indices[1]
                start_pose = env.mapping.hg.map_point_to_world_point(start_indices)
                start_indices[1] = start_indices[1] + 10
            else:
                start_pose = env.mapping.hg.map_point_to_world_point(np.append(to_transform_indices[:2], self.z_in_pix))
                #start_indices = np.append(to_transform_indices[:2], self.z_in_pix)

            start_pose[1] -= 0.02
            start_pose[2] = 0.975
            start_arm_joint_config, path_to_start_position, _ = env.klampt_utils.test_feasibility(initial_joint_config,
                                                                                                  start_pose,
                                                                                                  just_endpoints=just_endpoints,
                                                                                                  verbose=verbose,
                                                                                                  is_pybullet_config=True,
                                                                                                  free_yaw=False)
            if (start_arm_joint_config is not None):
                data['all_start_indices'].append(start_indices)
                data['start_arm_joint_configs'].append(start_arm_joint_config)
                data['path_to_start_positions'].append(path_to_start_position)
                data['start_poses'].append(start_pose)
        return data


    def find_end_points(self, data, start_data, env, occ_map, samples=None,  just_endpoints=False, verbose=False):
        ''' Sample goal poses from given start poses
        Args:
            data: Empty dictionary containing the paths, path annotations and motion parametrization
            start_data: Dictionary containing the start poses, arm joint configurations, path to start positions and all start indices
            env: Environment object for klampt usage
            occ_map: Occupancy map
            just_endpoints: return just endpoints in path (no interpolation)
        Return:
            dict: Dictionary containing the paths, path annotations and motion parametrization
        '''
        for start_arm_joint_config, path_to_start_position, start_indices, start_pose \
                in zip(start_data['start_arm_joint_configs'], start_data['path_to_start_positions'],
                       start_data['all_start_indices'], start_data['start_poses']):
            # sampling goal points
            reach_annotations = len(path_to_start_position) * ['pushing']

            if samples is not None:
                start_indices[1] = 140 - start_indices[1]
                idx = np.where((samples[:, 0, :] == start_indices[:2]).all(axis=1))[0]
                start_indices[1] = 140 - start_indices[1] - 10
                end_indices = samples[idx, 1, :]
                target_indices = np.hstack([end_indices, np.full((idx.shape[0], 1), 44)])

            else:
                push_directions = cp.asarray(self.find_push_direction_points(cp.asnumpy(occ_map).copy()[np.newaxis,:,:,5:].max(axis = 3), cp.asnumpy(start_indices[:2]).copy()))
                num_points = min(push_directions.shape[0], 10)
                if (num_points > 0):
                    # Sample push directions
                    sampled_indices = np.random.choice(push_directions.shape[0], num_points, replace=False)
                    sampled_push_points = push_directions[sampled_indices].astype(float)

                    # Compute push directions from start_indices
                    start_xy = start_indices[:2].astype(float)
                    push_directions = sampled_push_points - start_xy  # shape: (num_points, 2)
                    norms = np.linalg.norm(push_directions, axis=1, keepdims=True)
                    push_directions /= norms  # Normalize

                    # Sample random distances and compute target positions
                    random_distances = cp.asarray(self.rng.uniform(10, 50, size=(num_points, 1)))
                    target_indices = start_xy + push_directions * random_distances  # shape: (num_points, 2)
                    to_transform_indices = map_index_to_world_voxel_index(start_indices)

                    start_indices = np.append(to_transform_indices[:2], self.z_in_pix)
                else:
                    continue

            # Clip and convert to uint16
            target_indices = np.clip(np.around(target_indices, decimals=0), 0, 200).astype(np.uint16)

            for j in range(target_indices.shape[0]):
                #to_transform_indices = map_index_to_world_voxel_index(target_indices)
                #to_transform_indices[j, 0] = np.clip(to_transform_indices[j, 0], 0, 200)
                #to_transform_indices[j, 1] = np.clip(to_transform_indices[j, 1], 0, 140)
                #target_pose = env.mapping.hg.map_point_to_world_point(np.append(to_transform_indices[j, :2], self.z_in_pix))
                if samples is not None:

                    target_indices[j,1] = 140 - target_indices[j,1] - 10
                    target_pose = env.mapping.hg.map_point_to_world_point(target_indices[j])
                    #target_indices[j,1] = 140 - target_indices[j,1] - 10

                else:
                    to_transform_indices = map_index_to_world_voxel_index(target_indices)
                    #to_transform_indices = target_indices
                    to_transform_indices[j, 0] = np.clip(to_transform_indices[j, 0], 0, 200)
                    to_transform_indices[j, 1] = np.clip(to_transform_indices[j, 1], 0, 140)
                    target_pose = env.mapping.hg.map_point_to_world_point(np.append(to_transform_indices[j, :2], self.z_in_pix))
                    target_indices[j] = to_transform_indices[j, :2]
                target_pose[1] -= 0.02
                target_pose[2] = 0.975
                _, path_to_goal_position, _ = env.klampt_utils.test_feasibility(start_arm_joint_config,
                                                                                     target_pose,
                                                                                     just_endpoints=just_endpoints,
                                                                                     verbose=verbose, free_yaw=False)

                start_indices[:2] = start_indices[:2][::-1]
                target_indices[j][:2] = target_indices[j][:2][::-1]
                start_indices[1] = 200 - start_indices[1]
                target_indices[j][1] =200- target_indices[j][1]
                if path_to_goal_position is not None:
                    path_to_start_position.pop()
                    path_to_start_position.extend(path_to_goal_position)
                    push_annotation = len(path_to_goal_position) * ['pushing']
                    reach_annotations.extend(push_annotation[1:])
                    data['paths'].append(path_to_start_position)
                    data['path_annotations'].append(reach_annotations)
                    data['motion_parametrization'].extend(start_indices.tolist())
                    data['motion_parametrization'].extend([target_indices[j].flatten()[-2]])
                    data['motion_parametrization'].extend(target_indices[j, :2].flatten().tolist())
                    break

        return data


    def find_push_direction_points(self, voxel_map, pt, radius=12, push_method="MEM"):
        ''' Find push direction points
        Args:
            voxel_map: Voxel map
            pt: start Point to push
            radius: Radius of the circle for neighborhood
            push_method: Push method
        Return:
            valid_push_directions: Valid push directions
        '''

        if (self.a is None):
            self.a = np.zeros(voxel_map[0].shape, dtype=np.uint8)
            self.no = np.zeros(voxel_map[0].shape, dtype=np.uint8)
            self.idxs = np.indices(self.a.shape, dtype=np.uint16)
        else:
            self.a[:, :] = 0
            self.no[:,:] = 0
            self.a = self.a.astype(np.uint8)

        occupied = voxel_map[0] > 0.5
        # eliminating walls from push direction consideration
        if push_method == "MEM":
            occupied[:, :23] = False
            occupied[:, -23:] = False
            occupied[115:, :] = False
        else:
            occupied[:, :54] = False
            occupied[:, -55:] = False
            occupied[425:, :] = False
            occupied[:142, :] = False
        kernel = np.ones((3, 3), np.uint8)
        occupied = cv2.erode(occupied.astype(np.uint8), kernel, iterations=2)

        cv2.circle(self.a, pt[::-1], radius, [True], thickness=-1)
        cv2.circle(self.no,pt[::-1],2*radius//3,[True],thickness = -1)

        b = np.logical_and(occupied, self.a)
        valid_push_directions = self.idxs[:, b].transpose()
        return valid_push_directions


def execute_push(env, path, path_annotations = None, verbose=False):
    env.reset_robot(env.initial_parameters)

    if (path_annotations is None):
        path_annotations = len(path) * ['free']
    if (verbose):
        print(f'[execute_push] Executing the path')

    # invert path to go back from push
    reversed_path = path.copy()
    reversed_annotations = path_annotations.copy()
    reversed_annotations.reverse()
    reversed_path.reverse()

    complete_path = path.copy()
    complete_path.extend(reversed_path)
    complete_annotations = path_annotations.copy()
    complete_annotations.extend(reversed_annotations)

    mps = env.linear_interpolate_motion_klampt_joint_traj(complete_path, traj_annotation=complete_annotations, verbose=verbose)

    env.reset_robot(env.initial_parameters)

    for i, obj_id in enumerate(list(env.current_obj_ids) + [env.shelf_id]):
        if obj_id != env.shelf_id:
            collision = env.check_for_tilted(obj_id, i)

    if collision.any():
        return 1, mps
    return 0, mps


def visualize_push_point(pt, occupancy_height_map, show=True):
    a = np.zeros((140, 200))
    image = np.zeros((140, 200, 3))
    image[:, :, 0] = occupancy_height_map[0] > 0.5
    cv2.circle(a, pt[::-1], 1, [True], thickness=-1)
    image[:, :, 1] = a
    if (show):
        plt.imshow(image)
        plt.show()
    return image


def visualize_full_push(pt, occupancy_height_map, sampled_push):
    image = visualize_push_point(pt, occupancy_height_map, False)
    a = np.zeros((140, 200))
    cv2.circle(a, sampled_push[::-1], 1, [True], thickness=-1)
    image[:, :, 2] = a
    plt.imshow(image[::-1, ::-1, :])
    plt.title('sampled push')
    plt.show()


def map_index_to_world_voxel_index(indices, map_size=(200, 140)):
    original_shape = indices.shape
    if (len(original_shape) == 1):
        to_align = indices[np.newaxis, :].copy()
    else:
        to_align = indices.copy()
    to_align[:, [0, 1]] = to_align[:, [1, 0]]
    to_align[:, 0] = map_size[0] - to_align[:, 0]
    return to_align.reshape(original_shape)