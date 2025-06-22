import os
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=6
import numpy as np

from shelf_gym.utils.mapping_utils import BEVMapping
from glob import glob
from tqdm import tqdm
from stable_baselines3.common.env_util import make_vec_env
import pybullet as pb
import matplotlib.pyplot as plt
import multiprocessing
import cupy as cp
multiprocessing.set_start_method('spawn', force=True)
from shelf_gym.scripts.data_generation.map_collection import MapCollection
from shelf_gym.utils.dengler_iros_2023.viewpoint_push_planning_2023_utils import get_action_in_world_coord, get_information_gain, get_observation, \
    find_center_point, load_vae, load_vpp_agent, load_push_agent
from shelf_gym.utils.dengler_iros_2023.pushing_utils import OldPushSampler
from collections import deque
import pickle

VAE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../utils/dengler_iros_2023/models/vae_model")
VPP_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../utils/dengler_iros_2023/models/rl_model.zip")
PUSH_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../utils/dengler_iros_2023/models/push_model.pth.tar")

class MapComparison(MapCollection):
    def __init__(self, render=False, shared_memory=False, hz=240, use_egl=True, all_background = True, debug = False,
                 save_dir = '../data/old_approach_data', max_dataset_size = 1000,max_obj_num = 20,
                 use_occupancy_for_placing = False, use_ycb=False, show_vis = False,job_id = 1):
        super().__init__(render=render, shared_memory=shared_memory, hz=hz, max_obj_num = max_obj_num,
                         use_occupancy_for_placing=use_occupancy_for_placing, use_ycb=use_ycb, show_vis = show_vis)
        self.old_mapping = BEVMapping(self.hand_camera.get_cam_in_hand(self.robot_id, self.camera_link, remove_gripper=True, no_conversion = True), mapping_version="old", n_classes=15)

        self.save_dir = save_dir + '/{}/{:09d}/{}'
        self.job_id = job_id
        self.max_dataset_size = max_dataset_size
        self.all_background = all_background
        self.debug = debug

        self.vae = load_vae(VAE_PATH)
        self.rl_model = load_vpp_agent(VPP_MODEL_PATH)
        self.push_model = load_push_agent(PUSH_MODEL_PATH)

        self.ps = OldPushSampler()
        self.obsv_list = []

    def collect_old_mapping_approach_data(self, image_data, hms, dilated_hms, semantics, depths):
        # calculate heightmap from the point cloud
        data = self.old_mapping.postprocess_cam_data(image_data)

        hms.append(list(cp.asnumpy(data[3])))
        dilated_hms.append(cp.asnumpy(data[5]))
        semantics.append(cp.asnumpy(data[6]))
        depths.append(cp.asnumpy(data[7]))

        # pdb.set_trace()
        if (self.debug):
            # mapping the heightmap to the occupancy map for debugging
            self.old_mapping.mapping(data)
        return hms, dilated_hms, semantics, depths

    def get_camera_array_heightmaps(self):
        '''
        process the camera_array heightmaps and save them to disk
        Returns:
            iterations: number of iterations of mapping done so far
        '''
        # reset mapping to empty map
        self.mapping.reset_mapping()
        self.old_mapping.reset_mapping()
        # get image data from each camera on camera_array

        # for i, camera in enumerate(self.camera_array_cams):
        images = self.get_camera_array_images(self.camera_array_cams)
        # images = self.get_camera_array_images(self.camera_array_cams)
        # initialize the heightmaps
        new_mapping_hms, new_mapping_dilated_hms, new_mapping_semantics, new_mapping_depths = [], [], [], []
        old_mapping_dilated_hms = []

        for i in tqdm(range(len(images)), desc='camera_array reconstruction', miniters=50):
            # get projection matrix and view matrix for each camera
            projection_matrix = self.camera_array_cams[i]["projection_matrix"]
            view_matrix = self.camera_array_cams[i]["pybulelt_extrinsic_matrix"]
            intrinsic_matrix = self.camera_array_cams[i]["intrinsic_matrix"]
            image_data = self.hand_camera.get_image(images=images[i], remove_gripper=True,
                                                    projection_matrix=projection_matrix, view_matrix=view_matrix,
                                                    intrinsic_matrix=intrinsic_matrix, client_id=self.client_id)


            new_mapping_hms, new_mapping_dilated_hms, new_mapping_semantics, new_mapping_depths = self.collect_mapping_approach_data(image_data, new_mapping_hms, new_mapping_dilated_hms, new_mapping_semantics, new_mapping_depths)

        storage_dict, policy_Done_iter, collision_break = self.vp3_pipeline(self.rl_model, self.vae)

        if(self.debug):
            occupancy_height_map, semantic_map = self.mapping.get_2D_representation()
            plt.imshow(occupancy_height_map)
            plt.show()
            classification = semantic_map.argmax(axis = 2)
            plt.imshow(self.my_cmap[classification])
            plt.title('Estimated map from camera_array cameras')
            plt.show()
        return np.array(new_mapping_hms), \
            np.array(new_mapping_dilated_hms), \
            np.array(new_mapping_semantics), \
            np.array(new_mapping_depths), \
            storage_dict["hms"], \
            storage_dict["dilated_hms"], \
            storage_dict["depths"], \
            storage_dict["sms"], \
            storage_dict["cam_poses"], \
            storage_dict["pushes"], \
            np.array([policy_Done_iter, collision_break])




    def execute_action(self, pos, orn, link=None):
        self.reset_robot(self.initial_parameters)
        if link is None:
            link = self.tool_tip_id
        target_joint_states = self.get_ik_joints(pos, orn, link=link)
        target_joint_states.append(0)
        self.reset_robot(target_joint_states)
        # rgb, depth, pcd, hm, hm_binarized, hm_dilate, sm, transformed_depth
        img_data = self.hand_camera.get_cam_in_hand(self.robot_id, self.camera_link, client_id=self.client_id,  remove_gripper=True)
        return self.old_mapping.postprocess_cam_data(img_data)

    def step(self, action, last_action, last_map, vae, verbos=False):
        # calculate real action from [-1, 1]
        real_action = get_action_in_world_coord(action)
        # get camera image and hm from current position
        orn = self.init_ori + np.array([np.deg2rad(real_action[4]), 0, np.deg2rad(real_action[3])])
        cam_data = self.execute_action(real_action[:3], orn, link=self.tool_tip_id)
        pm = np.asarray(self.hand_camera.projection_matrix).reshape([1, 4, 4], order="F")
        vm = np.asarray(self.hand_camera.view_matrix).reshape([1, 4, 4], order="F")
        com_p, com_o, _, _, _, _ = pb.getLinkState(self.robot_id, self.camera_link,  computeForwardKinematics=True, physicsClientId=self.client_id)
        cam_pos = np.array([com_p, self._p.getEulerFromQuaternion(com_o)])
        self.old_mapping.mapping(cam_data)
        m = cp.asnumpy(self.old_mapping.occupancy_height_map)[0, :, :, 10:].max(axis=2)
        # get observation and ig, mc
        ig, ent_change, mc, obs = self.get_observation_and_data(m, last_map, real_action, last_action, vae)
        if verbos:
            print("action: ", action)
            print("obs: ", obs)
            print("real action: ", real_action)

        return_dict = {"obs": obs, "m": m, "hm": cp.asnumpy(cam_data["height_map"]), "depth": cp.asnumpy(cam_data["depth"]),
                      "sm": cp.asnumpy(cam_data["semantic_map"]), "target": real_action, "cam_pos": cam_pos, "ig": ig, "mc": mc,
                       "entropy_change": ent_change, "dilated_hm": cp.asnumpy(cam_data["hm_dilate"]), "vm": vm, "pm": pm}
        return return_dict


    def get_initial_views(self, storage_dict):
        target_points = np.array([[4.45533185990636e-05, 0.35772147664870574, 1.0336654049389589, 1.570019398429135,
                             -0.003184400360095557, 3.141559296569974],
                            [-0.27866206057278026, 0.3427375519798155, 1.3242585986002942, 2.1, -0.003438536767147895,
                             2.879277108324674],
                            [0.2788090038976225, 0.34278034211386127, 1.324262725884051, 2.1, -0.0033676080573663645,
                             -2.8803901512733208]])
        for i, tp in enumerate(target_points):
            cam_data = self.execute_action(target_points[i, :3], target_points[i, 3:], link=self.camera_link)
            self.old_mapping.mapping(cam_data)
            m = cp.asnumpy(self.old_mapping.occupancy_height_map)[0, :,:,10:].max(axis=2)
            cam_pos = np.array([target_points[i, :3], target_points[i, 3:]])

            pm = np.asarray(self.hand_camera.projection_matrix).reshape([1, 4, 4], order="F")
            vm = np.asarray(self.hand_camera.view_matrix).reshape([1, 4, 4], order="F")

            storage_dict["hms"] = np.concatenate((storage_dict["hms"], cp.asnumpy(cam_data["height_map"])[np.newaxis, :, :, :]), axis=0)
            storage_dict["dilated_hms"] = np.concatenate((storage_dict["dilated_hms"], cp.asnumpy(cam_data["hm_dilate"])[np.newaxis, :, :, :]), axis=0)
            storage_dict["depths"] = np.concatenate((storage_dict["depths"], cp.asnumpy(cam_data["depth"])[np.newaxis, :, :]), axis=0)
            storage_dict["sms"] = np.concatenate((storage_dict["sms"], cp.asnumpy(cam_data["semantic_map"])[np.newaxis, :, :]), axis=0)
            storage_dict["cam_poses"] = np.concatenate((storage_dict["cam_poses"], cam_pos[np.newaxis, :, :]), axis=0)
            storage_dict["vms"] = np.concatenate((storage_dict["vms"], vm), axis=0)
            storage_dict["pms"] = np.concatenate((storage_dict["pms"], pm), axis=0)
        return m, storage_dict

    def get_observation_and_data(self, m, old_m, target_pos, old_target_pos, vae, initial_obsv=False):
        if initial_obsv:
            ig = 0
            mc = 0
            entropy_change = 1
        else:
            ig, entropy_change = get_information_gain(m, old_m)
            mc = np.linalg.norm(target_pos[:3] - old_target_pos[:3])

        borders_map = self.old_mapping.hg.draw_borders(cp.asnumpy(m.copy()))
        obs = get_observation(borders_map, vae, target_pos, ig, mc, find_center_point(m, self.old_mapping))
        return ig, entropy_change, mc, np.array(obs).astype(np.float32)


    def vp3_pipeline(self, model, vae, sampling_num=40):
        storage_dict = {"hms": np.zeros((0, 434, 480, 2)),
                        "dilated_hms": np.zeros((0, 434, 480, 2)),
                        "depths": np.zeros((0, 480, 640)),
                        "sms": np.zeros((0, 434, 480)),
                        "cam_poses": np.zeros((0, 2, 3)),
                        "pushes": np.zeros((0, 2, 3)),
                        "push_iter": np.zeros((0, 1)),
                        "vms": np.zeros((0, 4, 4)),
                        "pms": np.zeros((0, 4, 4))}

        m, storage_dict = self.get_initial_views(storage_dict)
        target = np.array([0.00, 0.49, 0.97, 0, 0])
        first_ig, first_entropy_change, first_mc, obs = self.get_observation_and_data(m, cp.asnumpy(
            self.old_mapping.initial_prob_map), target, self.init_pos, vae, True)
        data_dict = {"m": m, "obs": obs, "target": target}
        iterations = 3
        entropy_change_history = deque(3 * [1.], 3)

        storage_dict, data_dict, iterations, entropy_change_history = self.perform_vpp(model, vae, data_dict["obs"],
                                                                                       data_dict["m"], data_dict["target"],
                                                                                       entropy_change_history, storage_dict,
                                                                                       iterations)

        collision_break = False
        policy_Done_iter = 0
        push_iter = []
        while iterations < sampling_num:
            push, object_drop = self.perform_pushing(self.old_mapping.occupancy_height_map[0])
            if push is None:
                continue
            iterations += 1
            push_iter.append([iterations])
            print(storage_dict["pushes"].shape, push.shape)
            storage_dict["pushes"] = np.concatenate((storage_dict["pushes"], push[np.newaxis, :, :]), axis=0)
            if object_drop:
                collision_break = True
                break

            data_dict["obs"][37:40] = np.array([0, 0, 0])
            data_dict["obs"][32:37] = np.array([0.00, 0.49, 0.97, 0, 0])
            entropy_change_history = deque(3 * [1.], 3)

            storage_dict, data_dict, iterations, entropy_change_history = self.perform_vpp(model, vae, data_dict["obs"],
                                                                                           data_dict["m"], data_dict["target"],
                                                                                           entropy_change_history, storage_dict,
                                                                                           iterations, random=True)
            if entropy_change_history[0] < 0.01:
                policy_Done_iter = iterations
        storage_dict["push_iter"] = np.concatenate((storage_dict["push_iter"], np.array(push_iter)), axis=0)
        return storage_dict, policy_Done_iter, collision_break


    def perform_vpp(self, model, vae, obs, m, target, entropy_change_history, storage_dict, iterations=0, random=False):
        Done = False
        data_dict = {"m": m, "obs": obs, "target": target}
        max_iter = 40
        while not Done:
            if iterations >= max_iter:
                break
            # predict next action
            if random:
                action = np.random.uniform(-1, 1, 5)
            else:
                action, _states = model.predict(data_dict["obs"], deterministic=True)
            # execute action
            data_dict = self.step(action, data_dict["target"], data_dict["m"], vae)
            iterations += 1
            entropy_change_history.appendleft(data_dict["entropy_change"])
            Done = self.stop_vpp(iterations, max_iter, entropy_change_history)
            storage_dict["hms"] = np.concatenate((storage_dict["hms"], data_dict["hm"][np.newaxis, :, :, :]), axis=0)
            storage_dict["dilated_hms"] = np.concatenate((storage_dict["dilated_hms"], data_dict["dilated_hm"][np.newaxis, :, :, :]), axis=0)
            storage_dict["depths"] = np.concatenate((storage_dict["depths"], data_dict["depth"][np.newaxis, :, :]), axis=0)
            storage_dict["sms"] = np.concatenate((storage_dict["sms"], data_dict["sm"][np.newaxis, :, :]), axis=0)
            storage_dict["cam_poses"] = np.concatenate((storage_dict["cam_poses"], data_dict["cam_pos"][np.newaxis, :]), axis=0)
            storage_dict["vms"] = np.concatenate((storage_dict["vms"], data_dict["vm"]), axis=0)
            storage_dict["pms"] = np.concatenate((storage_dict["pms"], data_dict["pm"]), axis=0)
        return storage_dict, data_dict, iterations, entropy_change_history

    def stop_vpp(self, current_iter, max_iter, entropy_change_history):
        # check if max steps reached
        max_iter_reached = current_iter >= max_iter
        # check for termination criteria success
        if current_iter >= 3:
            is_termination = max_iter_reached or \
                             np.sum(entropy_change_history) <= 0.05 or\
                             (np.array(entropy_change_history) <= 0.01).all()#or info_gain < 0.005
            return is_termination
        return False


    def generate_heightmap_3d(self, occupancy_height_map):
        '''generate 3D heightmap from 2D heightmap'''
        height_bins = np.round(0.5 / 0.005).astype(int)
        clipped_hm = np.round(np.clip( self.old_mapping.height_grid / 0.005, 0, height_bins)).astype(np.uint8)#[:, :, np.newaxis]
        unchanged = occupancy_height_map[1] < 0.024
        # unchanged = hm[:,:,0] < 0.001
        occupied = self.old_mapping.height_grid < clipped_hm
        free = np.logical_not(occupied)
        free[occupancy_height_map[1].astype(bool)] = False
        occupied[unchanged] = False
        return occupied.astype(np.uint8)

    def remove_walls(self, voxel):
        mask = np.ones(voxel.shape, dtype=bool)
        mask[142:424, 54:425, 1:102] = False
        voxel[mask] = 0
        return voxel

    def perform_pushing(self, occupancy_height_map):
        hm3d = occupancy_height_map #self.generate_heightmap_3d(occupancy_height_map)
        hm3d = self.remove_walls(hm3d)
        start, end = self.ps.perform_pushing(self, self.push_model, hm3d)
        if start is None or end is None:
            return None, False
        collision = self.obj.check_all_object_drop(self.current_obj_ids)
        if collision:
            print("Collision detected")
        return np.array([start, end]), collision



    ######################################
    '''Fundamental data Collection code'''
    ######################################
    def collect_data(self):
        self.collect_camera_array_data_for_both_mapping_approaches('pre_action')


    def collect_hard_scene_data(self):
        hard_dir = "../data/Hard_scenes/scenes"
        number = len([name for name in os.listdir(hard_dir) if name.endswith('.p')])
        print("Number of hard scenes: ", number)
        for i in range(number):
            #for file in pickle_files:
            with open(hard_dir + "/scene_data_" + str(i+1) + ".p", 'rb') as f:
                data = pickle.load(f)

            self.restore_shelf_state(data)
            self.old_mapping.reset_mapping()
            sampling_num= 40
            storage_dict, policy_Done_iter, collision_break = self.vp3_pipeline(self.rl_model, self.vae, sampling_num=sampling_num)
            dir = "../data/old_approach_data/Medium_scenes/results"
            proj_view_matrix_file = dir + "/data_" + str(i+1) + ".npz"

            left_over_iter = sampling_num - (storage_dict["vms"].shape[0] + storage_dict["push_iter"].shape[0])
            if left_over_iter > 0:
                extra_array_vm = np.repeat(storage_dict["vms"][-1:], left_over_iter, axis=0)
                extra_array_pm = np.repeat(storage_dict["pms"][-1:], left_over_iter, axis=0)
                storage_dict["vms"] = np.concatenate((storage_dict["vms"], extra_array_vm), axis=0)
                storage_dict["pms"] = np.concatenate((storage_dict["pms"], extra_array_pm), axis=0)

            if not os.path.exists(dir):
                os.makedirs(dir, exist_ok=True)
            with open(proj_view_matrix_file, 'wb') as f:
                np.savez_compressed(f, vm=storage_dict["vms"], pm=storage_dict["pms"], push=storage_dict["pushes"], push_iter=storage_dict["push_iter"])
        return


    def collect_scene_data(self, samples=100):
        iterations = self.get_iterations()
        save_dir = self.create_save_dir("vp3")
        for i in range(samples):
            self.reset()
            self.old_mapping.reset_mapping()
            storage_dict, policy_Done_iter, collision_break = self.vp3_pipeline(self.rl_model, self.vae)
            proj_view_matrix_file = save_dir + "/old_approach_data_" + str(i) + ".npz"
            scene_data_file = save_dir + "/scene_data_" + str(i) + ".p"

            with open(scene_data_file, 'wb') as f:
                pickle.dump(self.sampled_objects, f)

            with open(proj_view_matrix_file, 'wb') as f:
                np.savez_compressed(f, vm=storage_dict["vms"], pm=storage_dict["pms"], push=storage_dict["pushes"], push_iter=storage_dict["push_iter"])
        return


    def collect_camera_array_data_for_both_mapping_approaches(self, extra_annotation):
        # we first collect all the heightmaps from each camera in the camera array
        # new_hms, new_dilated_hms, new_semantic_hms, new_depths, old_hms, old_dilated_hms, old_semantic_hms, old_depths, old_cam_pos
        return_data = self.get_camera_array_heightmaps()

        # we then create the folders for saving the data

        self.get_iterations()

        save_dir = self.create_save_dir(extra_annotation)
        print('Saving data to:', save_dir)
        # Finally, we save all the data to their corresponding variables within the folder

        new_mapping_hm_file = save_dir+'/new_mapping_hms.npz'
        old_mapping_hm_file = save_dir+'/old_mapping_hms.npz'
        camera_matrices_file = save_dir + '/camera_matrices.npz'
        placed_object_file = save_dir + '/placed_objects.pkl'
        network_termination_iters_file = save_dir + '/termination_iters_matrices.npz'

        with open(placed_object_file, 'wb') as f:
            pickle.dump(self.sampled_objects, f)

        with open(network_termination_iters_file, 'wb') as f:
            np.savez_compressed(f, iter=return_data[8])

        with open(new_mapping_hm_file, 'wb') as f:
            np.savez_compressed(f, hms=return_data[0], dilated_hms=return_data[1], semantic_hms=return_data[2], depths=return_data[3])

        with open(old_mapping_hm_file, 'wb') as f:
            np.savez_compressed(f, hms=return_data[4], dilated_hms=return_data[5], semantic_hms=return_data[6], depths=return_data[7])

        with open(camera_matrices_file, 'wb') as f2:
            np.savez_compressed(f2, obj_ids=self.camera_array_cams)
        return

    def get_iterations(self):        
        # pdb.set_trace()
        if(not self.collected_iteration_number):
            parent_dir = self.save_dir.rsplit('/', maxsplit=3)[0]
            iterations = len(glob(parent_dir + '/{}/*'.format(self.job_id)))
            self.iterations = iterations
            self.collected_iteration_number = True
            #updating the tmp path to avoid clashes:
        return self.iterations

    def create_save_dir(self,extra_annotation):
        # generate save directory
        save_dir = self.save_dir.format(self.job_id,self.iterations,extra_annotation)
        os.makedirs(save_dir, exist_ok=True)
        os.environ["TMPDIR"] = os.path.abspath(save_dir)

        return save_dir

    def initialize_object_info(self):
        '''initialize object information'''
        self.current_obst_pos = np.zeros((len(self.current_obj_ids), 6))
        self.object_tilted = np.zeros((len(self.current_obj_ids)))
        self.object_contact = np.zeros((len(self.current_obj_ids)+1))




    ################################
    ''' legacy code '''
    ################################

    def _get_initial_heightmap(self, target_point):
        """Execute viewpoint planning using ground truth data."""
        _ = self.mapping.get_occupancy_heightmap(self.hand_camera.get_cam_in_hand(self.robot_id, self.camera_link, remove_gripper=True))
        for tp in target_point:
            self.man.fast_vp(tp)
            _ = self.mapping.get_occupancy_heightmap(self.hand_camera.get_cam_in_hand(self.robot_id, self.camera_link, remove_gripper=True))
        self.man.reset_robot(self.initial_parameters)
        prob_map = self.mapping.get_2D_representation()
        return prob_map


def get_single_image(args):
    camera = args[0]
    p_id = args[1]
    img = pb.getCameraImage(640, 480, camera["view_matrix"], camera["projection_matrix"],
                                 shadow=False, renderer=pb.ER_BULLET_HARDWARE_OPENGL, physicsClientId=p_id)
    return img

################################
'''main'''
################################

def load_default_task():
    env = make_vec_env('ViewPointNewEnv-v0')
    return env.envs[0].env.env

def test_env(environment):
    iterations = 0
    while (iterations <= environment.max_dataset_size):
        environment.reset()
        environment.collect_data()
        environment.step_simulation(environment.per_step_iterations)
        iterations = environment.get_iterations()

def get_evaluation_data(environment):
    environment.collect_scene_data(samples=100)

def evaluation_paralell(parallel, i):
    environment = MapComparison(render=True, shared_memory=False, hz=240, use_egl=True, debug=False, use_ycb=True,
                                show_vis=False, use_occupancy_for_placing=True, max_dataset_size=10000, job_id=i)
    environment.collect_scene_data(samples=50)



def tryout(parallel,i):
    environment = MapComparison(render=not parallel, shared_memory=False, hz=240, use_egl=True,debug=False,use_ycb=True, show_vis = False ,use_occupancy_for_placing = True, max_dataset_size=10000,job_id = i)
    test_env(environment)

def eval_hard_scenes():
    environment = MapComparison(render=False, shared_memory=False, hz=240, use_egl=True, debug=False, use_ycb=True,
                                show_vis=False, use_occupancy_for_placing=True, max_dataset_size=10000, job_id=1)
    environment.collect_hard_scene_data()
    environment.close()

if __name__ == '__main__':
    eval_hard_scenes()

    # parallel = False
    # evaluation = False
    # if evaluation:
    #     if (parallel):
    #         n_jobs = 2
    #         Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(evaluation_paralell)(parallel, i) for i in range(n_jobs))
    #     else:
    #         evaluation_paralell(parallel, 0)
    # else:
    #     if(parallel):
    #         n_jobs = 5
    #         Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(tryout)(parallel, i) for i in range(n_jobs))
    #     else:
    #         tryout(parallel, 0)

