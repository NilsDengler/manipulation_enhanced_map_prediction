import os
import numpy as np
from shelf_gym.environments.shelf_environment import ShelfEnv
from shelf_gym.utils.mapping_utils import BEVMapping
from glob import glob
from joblib import Parallel, delayed
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
import cupy as cp
multiprocessing.set_start_method('spawn', force=True)
import pickle
import yaml
import time


class MapCollection(ShelfEnv):
    def __init__(self, render=False, shared_memory=False, hz=240, all_background = True,
                 debug = False, save_dir = '../../data/map_data', max_dataset_size = 1000,
                 max_obj_num = 25, max_occupancy_threshold=.4, use_occupancy_for_placing = True,
                 use_ycb=True, show_vis = False,job_id = 1):

        super().__init__(render=render, shared_memory=shared_memory, hz=hz, max_obj_num = max_obj_num,
                         max_occupancy_threshold = max_occupancy_threshold,show_vis = show_vis, use_ycb=use_ycb)
        # parameters for speed improvement
        os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
        os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=4
        os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=6
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=4
        os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=6

        # initialize the mapping class
        initial_imgs = self.hand_camera.get_cam_in_hand(self.robot_id, self.camera_link, remove_gripper=True, no_conversion = True)
        self.mapping = BEVMapping(initial_imgs , n_classes=len(self.obj.obj_urdf_names)+1, mapping_version="MEM")

        # initialize the camera array
        self.camera_array_base = np.array([0, 0.95, 0.97 + 0.1])
        self.camera_array_cams_1 = self.generate_cameras(target_point=[-0.3, 0.95, 1.07], visualize=False)
        self.camera_array_cams_2 = self.generate_cameras(target_point=[0.3, 0.95, 1.07], visualize=False)
        self.camera_array_cams_3 = self.generate_cameras(target_point=[0, 0.95, 1.07], visualize=False)
        self.camera_array_cams = self.camera_array_cams_1 + self.camera_array_cams_2 + self.camera_array_cams_3

        #save camera matrix for safety reasons, can be disabled if not needed
        #camera_matrix_save_dir = "../model"
        #camera_matrices_file = camera_matrix_save_dir + '/camera_matrices.npz'
        #with open(camera_matrices_file,'wb') as f2:
        #    np.savez_compressed(f2, obj_ids=self.camera_array_cams)

        # define general parameters
        self.save_dir = save_dir + '/{}/{:09d}/{}'
        self.job_id = job_id
        self.max_dataset_size = max_dataset_size        
        self.all_background = all_background
        self.debug = debug
        self.collected_iteration_number = False
        self.rng = np.random.default_rng()

        # set if occupancy or object number should be used for object sampling
        self.use_occupancy_for_placing = use_occupancy_for_placing
        #define lower occupancy threshold to sample
        self.occupancy_threshold = .35
        # define min number of objects to place. only used if occupancy is not used
        self.min_object_num = 10

        self.constraint_list = []


    def reset_env(self, occupancy_threshold = None, alignment = 0, hard_only = False):
        """
                Reset the environment to its initial state.
                Parameters:
                    occupancy_threshold: should occupancy or object number be used for object sampling. If None, use default as defined in constructor
                    alignment: should the object be aligned to each other (grocery shelf style)
                    hard_only: Use only scenes that have more than 3 fully occluded object (can take some time to find)
                Returns:
                   Bool: is the sampled scene a hard one (see above)
                """

        # Disable rendering for efficient resetting
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)

        # Ensure constraints are cleared (important if grasping)
        for constraint_id in self.constraint_list:
            self._p.removeConstraint(constraint_id)
        self.constraint_list.clear()

        # Perform initial shelf reset
        self.initial_reset()

        # Determine occupancy threshold if not provided
        if occupancy_threshold is None:
            occupancy_threshold = self.rng.uniform(self.occupancy_threshold, self.max_occupancy_threshold, 1)[0]

        # Randomly sample the number of objects to place
        sampled_obj_num = self.rng.integers(self.min_object_num, self.max_obj_num, 1, endpoint=True)[0]

        # Sample objects based on occupancy constraints
        self.sampled_objects = self.obj.sample_objects_on_shelf(
            use_occupancy=self.use_occupancy_for_placing,
            occupancy_threshold=occupancy_threshold,
            sample_object_num=sampled_obj_num,
            alignment=alignment
        )
        # Set the new shelf state
        self.set_shelf_state(self.sampled_objects)

        # Re-enable rendering
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

        # Determine if the configuration is "hard only"
        return self.check_Hardness() if hard_only else True


    def set_shelf_state(self, object_arrangement):
        """
        Sets the state of the shelf by physically placing objects and updating internal mappings.
        Parameters:
            object_arrangement (list): List of objects representing their placement on the shelf.
        """
        # Physically place objects and store their IDs
        self.current_obj_ids, self.current_obj_classes = self.obj.physically_place_objects(object_arrangement)
        self.initialize_object_info()

        # Initialize object contact tracking array
        num_objects = len(self.current_obj_ids)
        self.object_contact = np.zeros((num_objects + 1,), dtype=int)

        # Reset the object mapping system
        self.mapping.reset_mapping()

        # Retrieve and update instance-to-class dictionary
        instance_to_class_dict = self.obj.get_id_to_class_dict()
        self.instance_to_class_dict = self.add_other_elements_to_instance_dict(
            instance_to_class_dict, all_background=self.all_background
        )

        # Generate color map based on unique object classes
        max_class_id = max(self.instance_to_class_dict.values(), default=0)
        self.my_cmap = np.array(sns.color_palette("husl", max_class_id + 1))
        self.my_cmap[-1, :] = 0  # Ensure the last row is black (or fully transparent)

        # Update hand camera instance mapping
        self.hand_camera.update_instance_to_class_dict(instance_to_class_dict)
        self.hand_camera.remove_cameras()

        # Reset iteration tracking
        self.collected_iteration_number = False


    def restore_shelf_state(self,object_arrangement):
        for i in self.constraint_list:
            self._p.removeConstraint(i)
        self.obj.reset_dynamics(self.current_obj_ids)
        self.initial_reset()
        self.set_shelf_state(object_arrangement)


    def collect_mapping_approach_data(self, image_data, map_data):
        """
        collect all relevant mapping data (could be reduced if needed less)
        Parameters:
            image_data (dict): Dict of images (rgb, depth, semantics, etc.) from camera
            map_data (dict): Dict to store map data in (height_map, semantic_map, etc.)
        """

        # calculate heightmap from the point cloud
        data = self.mapping.postprocess_cam_data(image_data)

        map_data["height_maps"].append(cp.asnumpy(data["height_map"]))
        map_data["semantic_maps"].append(cp.asnumpy(data["semantic_map"]))
        map_data["instance_maps"].append(cp.asnumpy(data["instance_map"]))
        map_data["dilated_maps"].append(cp.asnumpy(data["hm_dilate"]))
        map_data["depth_maps"].append(cp.asnumpy(data["transformed_depth"]))
        map_data["semantics"].append(cp.asnumpy(data["semantics_image"]))

        if (self.debug):
            # mapping the heightmap to the occupancy map for debugging using classic occupancy mapping approach
            self.mapping.mapping(data)
        return map_data


    def get_camera_array_heightmaps(self, no_tqdm = False):
        '''
        process the camera_array heightmaps and save them to disk
        Parameters:
            no_tqdm (bool): indicates if progress bar should be visualized in terminal
        Returns:
            map_data: Dict to store map data in (height_map, semantic_map, etc.)
        '''
        # reset mapping to empty map
        self.mapping.reset_mapping()

        # get image data from each camera on camera_array
        images = self.get_camera_array_images(self.camera_array_cams)

        map_data = {"height_maps": [], "semantic_maps": [], "instance_maps": [],
                    "dilated_maps": [], "depth_maps": [], "semantics": []}

        for i in tqdm(range(len(images)), desc = 'camera_array reconstruction', miniters=50, disable=no_tqdm):
            # get projection matrix and view matrix for each camera
            projection_matrix = self.camera_array_cams[i]["projection_matrix"]
            view_matrix = self.camera_array_cams[i]["pybullet_extrinsic_matrix"]
            intrinsic_matrix = self.camera_array_cams[i]["intrinsic_matrix"]
            image_data = self.hand_camera.get_image(images=images[i], remove_gripper=True,
                                                    projection_matrix=projection_matrix, view_matrix=view_matrix,
                                                    intrinsic_matrix=intrinsic_matrix, client_id=self.client_id)
            # calculate heightmap from the point cloud
            map_data = self.collect_mapping_approach_data(image_data, map_data)

        if(self.debug):
            # visualize classically mapped occupancy map
            occupancy_height_map, semantic_map = self.mapping.get_2D_representation()
            plt.imshow(occupancy_height_map)
            plt.show()
            classification = semantic_map.argmax(axis = 2)
            plt.imshow(self.my_cmap[classification])
            plt.title('Estimated map from camera_array cameras')
            plt.show()
        return map_data


    def get_single_camera_array_heightmaps(self, index):
        '''
        process only one camera from camera_array and save to disk
        Parameters:
            index (int): index of desired camera in camera_array
        Returns:
            map_data: Dict to store map data in (height_map, semantic_map, etc.)
        '''
        # get image data from each camera on camera_array
        images = self.get_camera_array_images(self.camera_array_cams[index:index+1])

        map_data = {"height_maps": [], "semantic_maps": [], "instance_maps": [], "dilated_maps": [], "depth_maps": [],
                    "semantics": []}

        # get projection matrix and view matrix for each camera
        projection_matrix = self.camera_array_cams[index]["projection_matrix"]
        view_matrix = self.camera_array_cams[index]["pybullet_extrinsic_matrix"]
        intrinsic_matrix = self.camera_array_cams[index]["intrinsic_matrix"]
        image_data = self.hand_camera.get_image(images=images[0], remove_gripper=True,
                                                projection_matrix=projection_matrix, view_matrix=view_matrix,
                                                intrinsic_matrix=intrinsic_matrix, client_id=self.client_id)

        return self.collect_mapping_approach_data(image_data, map_data)


    def get_gt_height_map(self, no_tqdm = False):
        '''
        Generate the ground truth dataof the shelf and save them to disk
        Parameters:
            no_tqdm (bool): indicates if progress bar should be visualized in terminal
        Returns:
            gt_data: Dict to store ground truth data in (height_map, semantic_map, etc.)
        '''
        # remove the racks from the shelf for better visibility
        self.remove_racks_from_shelf()

        #define the camera parameters
        pm, im = self.get_gt_projection_and_intrinsic_matrix()
        vms = self.get_gt_map_view_matrices()

        #get the camera images from the generated view and projection matrices
        all_imgs = [self.hand_camera.get_image(images=None, view_matrix=vm, projection_matrix=pm,
                                                       intrinsic_matrix=im, client_id=self.client_id) for vm in vms]

        #remove debug lines for visualization if generated
        self.hand_camera.remove_cameras()

        # reset mapping class to empty map
        self.mapping.reset_mapping()

        # put shelf racks back in
        self.put_racks_in_shelf()

        #get the point cloud and heightmap from the gt camera images
        instance_maps = []
        for images, vm in tqdm(zip(all_imgs, vms),'getting gt heightmap', miniters = 10, disable = no_tqdm):
            height_map_data = self.mapping.hg.get_heightmap(images["point_cloud"]["numpy"], images["semantics"])
            instance_maps.append(cp.asnumpy(height_map_data["instance_map"]))
            self.mapping.mapping(height_map_data)

        # get ground truth map data, generated by classic occupancy grid mapping
        final_mapped_data = self.mapping.get_2D_representation()

        # visualize groundtruth semantic map for debugging
        if(self.debug):
            classification = cp.asnumpy(final_mapped_data["semantic_map"].argmax(axis = 2))
            plt.imshow(self.my_cmap[classification])
            plt.title('GT Map')
            plt.show()


        hm3d = self.mapping.prob_map

        # postprocess ground truth semantic map
        semantic_3d_map = self.mapping.semantic_map
        semantic_2d_map = final_mapped_data["semantic_map"].argmax(axis = 2)
        n_classes = np.max(list(self.instance_to_class_dict.values()))+1
        background_value = np.zeros(n_classes).astype(np.int64)
        background_value[-1] = 1
        semantic_3d_map[semantic_2d_map == n_classes-1] = background_value
        semantic_3d_map = semantic_3d_map.argmax(axis = 3)

        #store data in dict
        gt_data = {"voxel_height_map": hm3d, "voxel_semantic_map": semantic_3d_map,
                   "occupancy_height_map": final_mapped_data["occupancy_height_map"], "semantic_gt": semantic_2d_map,
                   "instance_maps": instance_maps}
        return gt_data
            

    def get_gt_projection_and_intrinsic_matrix(self):
        '''
        create projection and intrinsic matrix from pybullet
        '''

        projection_matrix = self._p.computeProjectionMatrixFOV(self.hand_camera.fov, self.hand_camera.aspect,
                                                               self.hand_camera.near, self.hand_camera.far)
        intrinsic_matrix = self.hand_camera.projection_matrix_to_intrinsic(projection_matrix)

        return projection_matrix, intrinsic_matrix


    def get_gt_map_view_matrices(self):
        """This function creates view matrices that scan over the virtual shelf in a simple grid pattern for ground truth generation"""
        vms = []
        for i in np.arange(0.8, 1.06, 0.15):
            for j in np.arange(-0.3, 0.31, 0.25):
                vm = self._p.computeViewMatrix([j, i, 1.8], [j, i, 0.1], [0.0, 1.0, 0.0])
                vms.append(vm)
        return vms


    ################################
    '''utils'''
    ################################
    def remove_walls(self, voxel):
        # sets the height values of walls in map to zero. We assume to know the boundaries of the shelf for this
        mask = np.ones(voxel.shape, dtype=bool)
        mask[40:110, 25:175, 1:102] = False
        voxel[mask] = 0
        return voxel


    def check_Hardness(self) -> bool:
        """
        Checks how many objects are hidden from any view in the current environment state.
        """
        # Retrieve ground truth height maps and semantic instance maps
        gt_data = self.get_gt_height_map(no_tqdm=True)

        # Merge instance ground truth maps into a single array
        merged_map_gt = np.zeros_like(gt_data["instance_maps"][0])
        for i in range(len(gt_data["instance_maps"])):
            mask = gt_data["instance_maps"][i] != -1
            merged_map_gt[mask] = gt_data["instance_maps"][i][mask]

        # Retrieve camera-based heightmaps and instance maps
        mapped_data = self.get_camera_array_heightmaps(no_tqdm=True)

        # Merge observed instance maps into a single array
        merged_map_array = np.zeros_like(mapped_data["instance_maps"][0])
        for i in range(len(mapped_data["instance_maps"])):
            mask = mapped_data["instance_maps"][i] != -1
            merged_map_array[mask] = mapped_data["instance_maps"][i][mask]

        # Identify hidden objects
        unique_gt_instances = set(np.unique(gt_data["instance_maps"]))
        unique_observed_instances = set(np.unique(merged_map_array))
        #unique_observed_instances = set(np.unique(mapped_data["instance_maps"]))

        hidden_objects_set = unique_gt_instances - unique_observed_instances
        hidden_objects_count = len(hidden_objects_set)

        # Debug output
        print(f"Hidden objects ({hidden_objects_count}) of classes: {hidden_objects_set}")
        print(f"There are {hidden_objects_count} truly hidden objects.")

        # Return True if more than one object is completely hidden
        return hidden_objects_count > 3


    def show_camera_arrays(self):
        '''this function shows the camera arrays with debug lines'''
        self.hand_camera.draw_cameras(self.camera_array_cams_1, target_point=[-0.3, 0.95, 1.07])
        self.hand_camera.draw_cameras(self.camera_array_cams_2, target_point=[0.3, 0.95, 1.07])
        self.hand_camera.draw_cameras(self.camera_array_cams_3, target_point=[0, 0.95, 1.07])
        self.hand_camera.remove_cameras()


    def erase_shelf_occlusions(self):
        '''
        this erases the top shelf board of the sim so we can get the ground truth maps from a BEV perspective
        '''
        res = self._p.getVisualShapeData(self.shelf_id)
        for i in range(len(res) + 1):
            if (i not in [1, 3, 0, 5, 7]):
                self._p.changeVisualShape(self.shelf_id, linkIndex=i, rgbaColor=[0, 1, 0, 0])


    def restore_shelf_visuals(self):
        '''
        this function restores the shelf visuals after getting the ground truth maps
        '''
        res = self._p.getVisualShapeData(self.shelf_id)
        c1 = res[1][7]
        for i in range(len(res) + 1):
            if (i not in [1, 3, 0, 5, 7]):
                self._p.changeVisualShape(self.shelf_id, linkIndex=i, rgbaColor=c1)

    ###################################
    ''' camera_array camera code'''
    ###################################


    def generate_cameras(self, grid_size=(10, 10), distance_to_target=0.45,
                         curvature_factor=0.025, area_size=(0.45, 0.2),
                         target_point=[0.3, 0.95, 1.07], visualize=False):
        '''
        Generate cameras in array like structure in front of the shelf
        Parameters:
            grid_size: rectangular grid size dimensions (default: (10, 10)),
            distance_to_target: distance from array to shelf (default: 0.45)
            curvature_factor: curvature of array, 0 -Y completly flat (default: 0.025)
            area_size: How much space does the grid span (default: (0.45, 0.2))
            target_point= target point to look-at (defines orientation of camera),
            visualize: enables or disables debug visualization
        Returns:
            cameras (list of dicts): single camera dict stores camera parameters, position and look-at target
        '''

        cameras = []
        num_cameras_x, num_cameras_y = grid_size
        area_width, area_height = area_size
        target_point = np.array(target_point)

        # Curvature radius determines the bend in the camera grid.
        curvature_radius = -(distance_to_target / curvature_factor) / 2

        # Calculate step sizes for angles based on grid size and curvature factor.
        angle_step_x = curvature_factor / (num_cameras_x - 1)
        angle_step_y = curvature_factor / (num_cameras_y - 1)

        # Calculate theta and phi steps for spherical coordinates.
        theta_step = np.pi / (num_cameras_x - 1)
        phi_step = np.pi / (num_cameras_y - 1)

        # Determine the grid offsets for positioning.
        offset_x = np.linspace(-area_width / 2, area_width / 2, num_cameras_x)
        offset_y = np.linspace(-area_height / 2, area_height / 2, num_cameras_y)

        for i in range(num_cameras_x):
            for j in range(num_cameras_y):
                # Calculate spherical angles for current positions.
                theta = theta_step * i
                phi = phi_step * j

                # Apply curvature to position.
                x = curvature_radius * np.sin(angle_step_x * (i - (num_cameras_x - 1) / 2)) - offset_x[i]
                y = distance_to_target - (-0.1 * np.sin(phi) * np.sin(theta))
                z = curvature_radius * (1 - np.cos(angle_step_y * (j - (num_cameras_y - 1) / 2))) + offset_y[j]

                # Apply rotation matrix for camera orientation.
                R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                camera_pos = np.dot(np.array([x, y, z]), R) + self.camera_array_base

                # Calculate the direction the camera is facing.
                target_direction = target_point - camera_pos
                target_direction /= np.linalg.norm(target_direction)

                up_vector = np.array([0, 0, 1])

                view_matrix = self._p.computeViewMatrix(camera_pos.tolist(), target_point.tolist(), up_vector.tolist())
                projection_matrix = self._p.computeProjectionMatrixFOV(self.hand_camera.fov, self.hand_camera.aspect,
                                                                       self.hand_camera.near, self.hand_camera.far)

                # Calculate intrinsic matrix for camera.
                # Calculate focal lengths and principal points from projection matrix to convert from opengl to open3d
                transformed_view_matrix, intrinsic_matrix = self.hand_camera.opengl_to_o3d(projection_matrix, view_matrix)

                # Compile camera information including position and matrices for viewing and projection.
                camera_info = {
                    "position": camera_pos,
                    "pybullet_extrinsic_matrix": np.array(view_matrix),
                    "projection_matrix": np.array(projection_matrix),
                    "intrinsic_matrix": intrinsic_matrix,
                    "o3d_extrinsic_matrix": transformed_view_matrix.flatten()
                }
                cameras.append(camera_info)
        if visualize:
            self.hand_camera.draw_cameras(cameras, target_point=target_point)
        return cameras


    def get_camera_array_images(self, cameras):
        ''' Simple Loop to get camera data for each camera in camera_array'''

        return [self._p.getCameraImage(self.hand_camera.width, self.hand_camera.height,
                                         camera["pybullet_extrinsic_matrix"], camera["projection_matrix"],
                                         shadow=False, renderer=self._p.ER_BULLET_HARDWARE_OPENGL) for camera in cameras]


    ######################################
    '''Fundamental data Collection code'''
    ######################################

    def collect_data(self):
        self.collect_gt_and_camera_array_data('pre_action')

    def collect_gt_and_camera_array_data(self, extra_annotation):
        # we first collect all the heightmaps from each camera in the camera array
        cam_array_data = self.get_camera_array_heightmaps()

        # we then collect the ground truth maps for this current configuration
        gt_data = self.get_gt_height_map()

        self.get_iterations()

        save_dir = self.create_save_dir(extra_annotation)

        # Finally, we save all the data to their corresponding variables within the folder
        hm_file = save_dir+'/hms.npz'
        camera_matrices_file = save_dir + '/camera_matrices.npz'
        gt_hm_file = save_dir+'/gt_hms.npz'

        if extra_annotation == 'pre_action':
            placed_object_file = save_dir + '/placed_objects.pkl'
            with open(placed_object_file, 'wb') as f:
                pickle.dump(self.sampled_objects, f)

        with open(hm_file,'wb') as f:
            np.savez_compressed(f, hms=cam_array_data["height_maps"], dilated_hms=cam_array_data["dilated_maps"],
                                semantic_hms=cam_array_data["semantic_maps"], semantics=cam_array_data["semantics"],
                                depths = cam_array_data["depth_maps"])

        with open(camera_matrices_file,'wb') as f2:
            np.savez_compressed(f2, obj_ids=self.camera_array_cams)

        with open(gt_hm_file,'wb') as f3:
            np.savez_compressed(f3, gt_hms=gt_data["occupancy_height_map"],
                                hm3d=gt_data["voxel_height_map"],
                                semantic_2d = gt_data["semantic_gt"],
                                semantic_3d = gt_data["voxel_semantic_map"])

        return gt_data, cam_array_data


    def get_iterations(self):
        ''' get current iteration if using multiple threads'''
        if(not self.collected_iteration_number):
            parent_dir = self.save_dir.rsplit('/', maxsplit=3)[0]
            iterations = len(glob(parent_dir + '/{}/*'.format(self.job_id)))
            self.iterations = iterations
            self.collected_iteration_number = True
        return self.iterations


    def create_save_dir(self, extra_annotation):
        ''' creates save dir for current folder and iteration '''
        save_dir = self.save_dir.format(self.job_id, self.iterations, extra_annotation)
        os.makedirs(save_dir, exist_ok=True)
        os.environ["TMPDIR"] = os.path.abspath(save_dir)

        return save_dir

################################
'''main'''
################################
def save_only_scene_config(environment, scen_num=100):
    save_dir = "../data/Medium_scenes"
    saved_environments = 0
    for i in range(scen_num):
        hidden = environment.reset_env(hard_only = True)
        while not hidden:
            hidden = environment.reset_env(hard_only = True)

        placed_object_file = save_dir + "/placed_objects_" + str(saved_environments+4) + ".p"
        with open(placed_object_file, 'wb') as f:
            pickle.dump(environment.sampled_objects, f)
        saved_environments += 1


def run_env(environment):
    iterations = 0
    while (iterations <= environment.max_dataset_size):
        environment.reset_env()
        environment.collect_data()
        environment.step_simulation(environment.per_step_iterations)
        iterations = environment.get_iterations()


def parse_config(file_name="map_collection_config.yaml"):
    with open("config/"+file_name, "r") as file:
        config_data = yaml.safe_load(file)
    return config_data


def run(parallel, config_data,  i):
    environment = MapCollection(render=not parallel,
                                shared_memory=config_data["shared_memory"],
                                hz=config_data["hz"],
                                use_ycb=config_data["use_ycb"],
                                debug=config_data["debug"],
                                show_vis = config_data["show_vis"],
                                max_dataset_size=config_data["max_dataset_size"],
                                job_id = i,
                                save_dir = config_data["save_dir"],
                                max_obj_num = config_data["max_obj_num"],
                                max_occupancy_threshold = config_data["max_occupancy_threshold"],
                                use_occupancy_for_placing = config_data["use_occupancy_for_placing"])
    run_env(environment)
    #save_only_scene_config(environment)


if __name__ == '__main__':
    # to increase data collection speed, set parallel=True and specify number of parallel jobs n_jobs
    parallel = False
    config_data = parse_config()
    if(parallel):
        n_jobs = 22
        Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(run)(parallel, config_data, i) for i in range(n_jobs))
    else:
        run(parallel, config_data, 0)

