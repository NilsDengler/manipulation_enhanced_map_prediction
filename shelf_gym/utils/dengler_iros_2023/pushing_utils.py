import matplotlib.pyplot as plt
import numpy as np
import cv2
from shelf_gym.utils.pushing_utils import PushSampler, FrontierFinder, map_index_to_world_voxel_index
import imutils
import torch
import cupy as cp

class OldPushSampler(PushSampler):
    def __init__(self):
        super().__init__()
        self.ff = FrontierFinder()

    def get_samples(self, occ_map, num_points=100, force_resample_idx=False, verbose=False):
        '''Input:
                    height_map: Voxel height map
                Returns:
                    radom samples for pushing
                '''
        if (verbose):
            print(f'[GET_SAMPLES] Getting samples for pushing')

        #preprocess the occupancy map
        voxel_map = occ_map.copy()
        voxel_map[voxel_map < 0.85] = 0
        heighest_cell = np.max(np.argwhere(voxel_map >= 0.85)[:,2])
        voxel_map[:,:,int(heighest_cell/2)] = 0

        #find frontiers in map
        occupied_indices = self.ff.find_frontiers(voxel_map, force_resample_idx,  pushing_method="old")
        occupied_indices = self.ff.filter_tall_frontiers(occupied_indices)

        if (verbose):
            print(f'[GET_SAMPLES] Occupied indices: {occupied_indices.shape}')

        equally_sampled_indices = cp.random.choice(occupied_indices.shape[0], num_points, replace=False)
        equally_sampled_indices = occupied_indices[equally_sampled_indices]


        # occupied_indices = self.ff.find_frontiers(height_map, pushing_method="old")
        # if (verbose):
        #     print(f'[GET_SAMPLES] Occupied indices: {occupied_indices.shape}')
        # # Sample 100 random points from the occupied indices
        # equally_sampled_indices = self.rng.choice(occupied_indices.get(), num_points, replace=True)
        # print(equally_sampled_indices.shape)

        rows = equally_sampled_indices[:, 0].astype(int).get()
        cols = equally_sampled_indices[:, 1].astype(int).get()

        test = np.zeros(occ_map.shape[:2], dtype=float)
        test[rows, cols] = 1

        height_map_max = occ_map[np.newaxis,:,:,5:].max(axis = 3).get()[0] # shape = (H, W)
        occupied = height_map_max > 0.5
        print("shape occupied_1", occupied.shape)

        fig, axes = plt.subplots(1, 3, figsize=(10, 5))

        # Left: the binary “test” mask
        axes[0].imshow(test, cmap='gray', origin='lower')
        axes[0].set_title('Sampled Frontier Mask')
        axes[0].set_xlabel('X (column)')
        axes[0].set_ylabel('Y (row)')

        # Right: the height_map (max‐projection)
        axes[1].imshow(height_map_max, cmap='viridis', origin='lower')
        axes[1].set_title('Height Map (max over channels)')
        axes[1].set_xlabel('X (column)')
        axes[1].set_ylabel('Y (row)')

        # Right: the height_map (max‐projection)
        axes[2].imshow(occupied, cmap='viridis', origin='lower')
        axes[2].set_title('Height Map (max over channels)')
        axes[2].set_xlabel('X (column)')
        axes[2].set_ylabel('Y (row)')

        plt.tight_layout()
        plt.show()


        if (verbose):
            print(f'[GET_SAMPLES] Equally sampled indices: {equally_sampled_indices.shape}')

        start_pushes = []
        end_pushes = []
        for i in range(len(equally_sampled_indices)):
            start_indices = equally_sampled_indices[i]
            push_directions = cp.asarray(self.find_push_direction_points(cp.asnumpy(occ_map).copy()[np.newaxis,:,:,5:].max(axis = 3), cp.asnumpy(start_indices[:2]).copy(), radius=32))

            #push_directions = self.find_push_direction_points(height_map.get()[np.newaxis,:,:,:], start_indices[:2].copy(), push_method="old", radius=32)
            print(push_directions)
            if (push_directions.shape[0] > 0):
                sampled_push_points = self.rng.choice(push_directions, 10)
                sampled_push_point = sampled_push_points[0]
                push_direction = sampled_push_point.astype(float) - start_indices[:2].astype(float)
                push_direction /= np.linalg.norm(push_direction, keepdims=True)
                target_indices = start_indices[:2].copy().astype(float)
                target_indices[:2] += self.rng.uniform(10, 90) * push_direction
                target_indices = np.clip(np.around(target_indices,decimals = 0),0,480).astype(np.uint16)
                start_pushes.append(start_indices)
                end_pushes.append(target_indices)
            else:
                continue
        print(start_pushes, end_pushes)
        return np.asarray(start_pushes), np.asarray(end_pushes)


    def preprocess_images_np(self, image):
        current_min, current_max = np.amin(image), np.amax(image)
        if current_min == current_max:
            return (image * 0).astype(np.float16)
        normed_min, normed_max = 0, 1
        x_normed = (image - current_min) / (current_max - current_min)
        x_normed = x_normed * (normed_max - normed_min) + normed_min
        return x_normed.astype(np.float16)


    def get_push_images(self, m, start_point, end_point):
        start_point = start_point.astype(float)
        end_point = end_point.astype(float)

        old_shape = np.array(m.shape[:2])
        # push_points, directions = shuffle(push_points, directions, random_state=1)
        push_sample_batch = np.zeros((len(start_point), 400, 400))
        for idx in range(len(start_point)):
            dst = cv2.copyMakeBorder(m, int(old_shape[0] / 2), int(old_shape[0] / 2),
                                     int(old_shape[1] / 2), int(old_shape[1] / 2),
                                     cv2.BORDER_CONSTANT, None, value=0)
            new_width, new_height = dst.shape[:2]
            new_center = np.array(dst.shape[:2]) / 2
            transformed_point = start_point[idx] + old_shape / 2
            translation = new_center - transformed_point
            translated_image = imutils.translate(dst, translation[1], translation[0])

            rot_angle = np.degrees(
                np.arctan2(end_point[idx][1] - start_point[idx][1], end_point[idx][0] - start_point[idx][0]))
            rotated_image = imutils.rotate(translated_image, rot_angle + 180)
            cropped_image = rotated_image[int((new_width / 2) - 200):int((new_width / 2) + 200),
                            int((new_height / 2) - 200):int((new_height / 2) + 200)]
            preprocessed_img = self.preprocess_images_np(cropped_image)
            push_sample_batch[idx] = preprocessed_img
        return push_sample_batch


    def network_inference(self, model, data, device="cuda"):
        model.eval()
        final_prediction = np.zeros(0)
        for i in range(int(np.ceil(data.shape[0] / 32))):
            n, w, h = data[i * 32:i * 32 + 32].shape
            torch_data = torch.from_numpy(data[i * 32:i * 32 + 32].reshape(n, 1, w, h)).float().to(device)
            with torch.no_grad():
                x_pred_tag = torch.sigmoid(model(torch_data))
                predictions = x_pred_tag.cpu().detach().numpy().reshape(n)
            final_prediction = np.concatenate((final_prediction, predictions))
        return np.argsort(final_prediction)[::-1]


    def perform_best_push(self, env, start_pose, target_pose, Debug=False):
        env.reset_robot(env.initial_parameters)

        if Debug:
            vs_id_start = env._p.createVisualShape(shapeType=env._p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1])
            m_id_start = env._p.createMultiBody(baseVisualShapeIndex=vs_id_start, basePosition=start_pose,
                                                baseOrientation=[0, 0, 0, 1])
            vs_id_end = env._p.createVisualShape(shapeType=env._p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 1, 0, 1])
            m_id_end = env._p.createMultiBody(baseVisualShapeIndex=vs_id_end, basePosition=target_pose,
                                              baseOrientation=[0, 0, 0, 1])
        verbose = False
        env.linear_interpolate_motion_klampt(start_pose, env.init_ori, verbose=verbose)
        env.linear_interpolate_motion_klampt(target_pose, env.init_ori, verbose=verbose)
        env.linear_interpolate_motion_klampt(start_pose, env.init_ori, verbose=verbose)
        env.linear_interpolate_motion_klampt(env.init_pos, env.init_ori, verbose=verbose)

        if Debug:
            env._p.removeBody(m_id_start)
            env._p.removeBody(m_id_end)


    def perform_pushing(self, env, push_model, height_map):
        '''Input:
            height_map: Voxel height map
        '''
        start_pushes, end_pushes = self.get_samples(height_map, 100)
        if start_pushes.shape[0] == 0 or end_pushes.shape[0] == 0:
            print("No push points were found")
            return None, None
        border_map = env.old_mapping.hg.draw_borders(occupancy_height_map[1].copy())
        push_images_batch = self.get_push_images(border_map, start_pushes[:, :2], end_pushes[:, :2])
        best_prediction_idxs = self.network_inference(push_model, push_images_batch)
        current_joint_config = env.init_arm_and_gripper_joint_config
        start_pose, target_pose = None, None
        for i in range(len(best_prediction_idxs)):
            map_size = np.flip(height_map.shape[:2])
            pz = (np.floor((0.97 - 0.91) / 0.005)).astype(np.int32)
            to_transform_indices = map_index_to_world_voxel_index(start_pushes[best_prediction_idxs[i]][:2], map_size=map_size)
            start_pose = env.old_mapping.hg.map_point_to_world_point(np.append(to_transform_indices, pz))
            start_pose[2] = 0.975
            start_arm_joint_config, path_to_start_position, start_pose= env.klampt_utils.test_feasibility(current_joint_config,
                                                                                                          start_pose,
                                                                                                          just_endpoints=True,
                                                                                                          verbose=False,
                                                                                                          is_pybullet_config=True)

            if (start_arm_joint_config is not None):
                to_transform_indices = map_index_to_world_voxel_index(end_pushes[best_prediction_idxs[i]][:2], map_size=map_size)
                to_transform_indices[0] = np.clip(to_transform_indices[0], 0, map_size[0])
                to_transform_indices[1] = np.clip(to_transform_indices[1], 0, map_size[1])
                target_pose = env.old_mapping.hg.map_point_to_world_point(np.append(to_transform_indices, pz))
                target_pose[2] = 0.975
                target_arm_joint_config, path_to_target_position, target_pose = env.klampt_utils.test_feasibility(
                    start_arm_joint_config,
                    target_pose,
                    just_endpoints=True,
                    verbose=False,
                    is_pybullet_config=True)

                if (target_arm_joint_config is not None):
                    break
            else:
                continue

        if start_pose is None or target_pose is None:
            print("No feasible push was found")
            return None, None

        self.perform_best_push(env, start_pose, target_pose)
        return start_pose, target_pose
