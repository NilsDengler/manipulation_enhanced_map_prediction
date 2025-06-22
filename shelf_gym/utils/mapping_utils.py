import cv2
import torch
import numpy as np
import cupy as cp
import open3d as o3d
import matplotlib.pyplot as plt

class BEVMapping():
    '''
    Class to handle the occupancy height mapping of the environment
    '''
    def __init__(self, init_cam_data, mapping_version="MEM", max_height=0.5, shelf_height = 0.024, height_resolution=0.005,
                 occupancy_threshold=0.75, epsilon = 0.01, n_classes = 17, raw_hm_start = False):
        '''
        Initialize the mapping class
        Args:
            init_cam_data: first camera data to generate apropriate initial maps
            mapping_version: specify the version of the mapping algorithm (MEM=Ours, old= dengler et al. 2023)
            max_height: specify the maximum height we consider the objects to be in the scene
            height_resolution: specify the height resolution of the for voxelized height map (default 0.005)
            shelf_height: specify the height of the shelf, used for updating the occupancy (default 0.024)
            occupancy_threshold: specify the threshold for the occupancy map (default 0.75)
            epsilon: specify the epsilon value for the semantic map (default 0.01)
            n_classes: specify the number of classes in the semantic map (default 15)
            raw_hm_start: define if the height map is given in the initial data (default False)
        '''

        self.hg = HeightmapGeneration(height_resolution, mapping_version, n_classes = n_classes)

        self.max_height = max_height
        self.height_resolution = height_resolution
        self.height_bins = np.round(self.max_height/self.height_resolution).astype(int)
        self.occupancy_thold = occupancy_threshold
        self.shelf_height = shelf_height
        self.epsilon = epsilon
        self.n_classes = n_classes
        self.raw_hm_start = raw_hm_start

        #initialize maps
        self.init_maps(init_cam_data)


    def init_maps(self, init_cam_data):
        '''
        Initialize the maps
        Args:
            init_cam_data: first camera data to generate apropriate initial maps
            dictionary with the following keys:
                rgb: rgb image
                depth: normalized depth image [0,255]
                transformed_depth: transformed depth image in meters
                semantics: semantic image
                point_cloud:  [open3d, numpy] point cloud
        '''

        if(not self.raw_hm_start):
            #get processed heightmap data from camera data
            processed_data = self.postprocess_cam_data(init_cam_data)
        else:
            processed_data = {"height_map": init_cam_data["height_map"]}

        #geth 3D shape dimensions for the probability height map
        hm_shape = cp.asarray(processed_data["height_map"].shape)
        shape_with_heights = cp.zeros(3).astype(int)
        shape_with_heights[:2] = hm_shape[:2]
        shape_with_heights[2] = self.height_bins + 2

        #initialize 3D probability height map
        self.initial_prob_map = cp.ones(shape_with_heights.tolist())

        # the height grid is used for quick computation
        self.height_grid = cp.zeros_like(self.initial_prob_map).astype(cp.uint8)
        self.torch_height_grid = torch.zeros_like(torch.from_numpy(cp.asnumpy(self.initial_prob_map))).char().to("cuda")
        for i in range(self.height_grid.shape[2]):
            self.torch_height_grid[:, :, i] = i
            self.height_grid[:, :, i] = i


        self.indices_map = cp.asnumpy(cp.stack(cp.indices(self.initial_prob_map.shape), axis=-1))
        self.reset_mapping()


    def reset_mapping(self):
        ''' Reset the mapping to the initial state '''

        self.prob_map = self.initial_prob_map
        self.log_map = cp.zeros_like(self.initial_prob_map)

        self.log_map -= 0.0001
        self.height_map = cp.zeros_like(self.initial_prob_map)

        self.swept_map = cp.zeros_like(self.initial_prob_map)

        prob_map_shape = list(self.initial_prob_map.shape)
        semantic_map_shape = prob_map_shape + [self.n_classes]

        # we begin with a small, but uniform prior
        self.semantic_map = cp.ones(semantic_map_shape)/self.n_classes
        self.semantic_map[:,:,:,-1] += 0.1
        self.semantic_counts = self.epsilon*np.ones_like(self.semantic_map)

        ohm = cp.expand_dims(self.initial_prob_map.copy(), axis=0)
        self.occupancy_height_map = cp.zeros_like(np.concatenate((ohm, ohm), axis=0))


    def postprocess_cam_data(self, cam_data=None):#
        '''
        Postprocess the camera data to get the necessary data for mapping
        Args:
            cam_data: direct output from the camera
        Returns:
            processed_data: dictionary with the following keys
                rgb: rgb image
                depth: normalized depth image [0,255]
                transformed_depth: transformed depth image in meters
                pointcloud: cupy point cloud
                semantics_image: semantic image
                height_map: height map
                height_map_binary: binary height map
                hm_dilate: dilated height map
                semantics_map: semantic map
                instance_map: instance map
        '''

        # get cupy data from numpy
        cam_data["rgb"] = cp.asarray(cam_data["rgb"])
        cam_data["depth"] = cp.asarray(cam_data["depth"])
        cam_data["transformed_depth"] = cp.asarray(cam_data["transformed_depth"])
        cam_data["semantics"] = cp.asarray(cam_data["semantics"])
        cam_data["point_cloud"]["numpy"] = cp.asarray(cam_data["point_cloud"]["numpy"])

        map_data = self.hg.get_heightmap(cam_data["point_cloud"]["numpy"], cam_data["semantics"])
        # preproces hm data
        hm_dilate, hm_binarized = self.hg.preprocess_heightmap(map_data["height_map"])

        processed_data = {"rgb": cam_data["rgb"], "depth": cam_data["depth"],
                          "transformed_depth": cam_data["transformed_depth"],
                          "pointcloud": cam_data["point_cloud"]["numpy"],
                          "semantics_image": cam_data["semantics"], "height_map": map_data["height_map"],
                          "height_map_binary": hm_binarized, "hm_dilate": hm_dilate,
                          "semantic_map": map_data["semantic_map"], "instance_map": map_data["instance_map"]}
        return processed_data


    def mapping(self, data):
        ''' Update the maps with the new data '''
        occupied, free = self.find_update_cells(data["height_map"])
        self.update_occupancy_map(occupied, free)
        self.occupancy_height_map[0] = self.prob_map
        self.update_height_map()
        self.update_semantic_map(data["semantic_map"], occupied, free)


    def find_update_cells(self, height_map):
        ''' Find the cells to update in the map
        Args:
            height_map: height map to update
        Returns:
            occupied: cells that are occupied
            free: cells that are free
        '''

        clipped_hm = cp.round(np.clip(height_map[:,:,0]/self.height_resolution, 0, self.height_bins)).astype(cp.uint8)[:,:,cp.newaxis]
        unchanged = height_map[:,:,0] < self.shelf_height
        occupied = self.height_grid < clipped_hm
        free = cp.logical_not(occupied)
        free[height_map[:,:,1].astype(bool)] = False
        occupied[unchanged] = False
        free[unchanged] = False
        return occupied, free


    def update_height_map(self):
        ''' Update the height map '''
        self.height_map = (self.prob_map > self.occupancy_thold).argmin(axis=2)*self.height_resolution


    def update_semantic_map(self, semantic_map, occupied, free):
        ''' Update the semantic map '''
        # if the semantic map is a class map, we need to one-hot-encode the observation,
        if(len(semantic_map.shape) == 2):
            sm_probs = cp.asarray(torch.nn.functional.one_hot(torch.from_numpy(cp.asnumpy(semantic_map.astype(np.int64))),num_classes = self.n_classes).numpy())
        else:
            sm_probs = semantic_map

        sm_probs_3d = cp.repeat(sm_probs[:,:,np.newaxis,:],axis = 2,repeats = occupied.shape[2])
        observed = cp.logical_or(free,occupied)
        # we then do the update for all observed cells using naive averaging (empirically better calibrated)
        self.semantic_map[observed] = (self.semantic_map[observed]*self.semantic_counts[observed] + sm_probs_3d[observed])/(self.semantic_counts[observed]+1)
        # and make sure to update the observation counts for semantics too
        self.semantic_counts[observed]+=1
        # and renormalize:
        self.semantic_map[observed] = self.semantic_map[observed]/self.semantic_map[observed].sum(axis =1,keepdims = True)


    def update_occupancy_map(self, occupied, free):
        ''' Update the occupancy map according to log odds'''
        self.log_map[free] += -0.85
        self.log_map[occupied] += 0.85
        self.log_map = cp.clip(self.log_map, -3.5, 3.5)
        self.prob_map = 1-(1/(1 + cp.exp(self.log_map.copy())))


    def get_2D_representation(self):
        ''' Get the 2D representation of the map
         Returns:
             - (w,h,2) array with the 2D representations (occupancy, height) of the map
             - semantic map
        '''
        occupancy_map = self.prob_map[:,:,5:].max(axis=2)
        height_map = self.height_map
        # mapped with certainty
        uncertainly_mapped = occupancy_map < self.occupancy_thold
        # hollow out the height of points that do not make it to the existence threshold
        height_map[uncertainly_mapped] = 0

        # for determining ground level 2D semantics, we take an occupancy probability weighed mean over the heights:
        semantic_map = (self.semantic_map*self.prob_map[:,:,:,cp.newaxis]).mean(axis=2)
        semantic_map[uncertainly_mapped] = 0
        semantic_map[uncertainly_mapped,-1] = 1
        data = {"occupancy_height_map": np.array([cp.asnumpy(occupancy_map), cp.asnumpy(height_map)]), "semantic_map": semantic_map}
        return data


class SweptMapGenerator:
    def __init__(self):
        """
        Initializes the SweptMapGenerator with a heightmap generator.
        """
        self.hg = HeightmapGeneration(0.005, "MEM", n_classes=15)


    def get_swept_map(self, world_positions):
        """
        Generates the swept map given world positions.
        Args:
            world_positions (list or np.ndarray): List of 3D world coordinates.
        Returns:
            np.ndarray: The swept volume as a NumPy array.
        """
        if world_positions is None:
            return None

        wp = np.array(world_positions)
        swept_volume = np.zeros((140, 200, 102), dtype=np.uint8)

        swept_volume = self.update_swept_volume_with_gripper(wp, swept_volume)

        swept_volume = self.update_swept_volume_with_rectangle(wp[:, :2, :], swept_volume)
        swept_volume = self.update_swept_volume_with_rectangle(wp[:, 2:4, :], swept_volume)

        return cp.asnumpy(swept_volume)


    def update_swept_volume_with_polygon(self, points, swept_volume, thickness=0.035):
        """
        Updates the swept volume with a polygon.
        Args:
            points (np.ndarray): Polygon vertices.
            swept_volume (np.ndarray): 3D swept volume grid.
            thickness (float): Thickness of the swept volume.
        Returns:
            np.ndarray: Updated swept volume.
        """
        swept_volume = swept_volume.astype(np.uint8)
        resolution = 0.005
        points[:, :, :] = points[:, :, [1, 0, 2]]
        #points = np.copy(points)
        #points[..., [0, 1]] = points[..., [1, 0]]  # Swap x and y for image processing

        # Compute bounding box
        map_size = np.array(swept_volume.shape[:2])
        min_index = np.minimum(np.array([0, 0]), points.reshape(-1,3).min(axis = 0)[:2])
        max_index = np.maximum(map_size, points.reshape(-1,3).max(axis = 0)[:2])
        image_shape = max_index - min_index

        image = np.zeros(image_shape.tolist(), dtype=np.uint8)
        points[..., :2] -= min_index  # Shift points to fit in the bounding box
        thickness_px = thickness / resolution

        for triangle in points:
            layer = image.copy()
            z = triangle[0, 2]
            triangle[:, [0, 1]] = triangle[:, [1, 0]]  # Swap x and y back

            cv2.drawContours(layer, [cp.asnumpy(triangle[:, :2])], 0, [1], -1)
            layer = np.asarray(layer)
            this_swept = layer[-min_index[0]:-min_index[0] + map_size[0], -min_index[1]:-min_index[1] + map_size[1]]
            this_swept = this_swept[:, ::-1]
            min_z = int(max(z - (thickness_px - 1) // 2, 0))
            max_z = int(min(z + (thickness_px - 1) // 2 + 1, swept_volume.shape[2]))

            swept_volume[:, :, min_z:max_z] += this_swept[:, :, np.newaxis]

        return swept_volume > 0


    def update_swept_volume_with_gripper(self, wp, swept_volume):
        """
        Updates the swept volume to account for the robotic gripper.
        Args:
            wp (np.ndarray): World positions.
            swept_volume (np.ndarray): 3D swept volume grid.
        Returns:
            np.ndarray: Updated swept volume.
        """
        relevant_wps = wp[:,-3:].reshape(-1,3)
        #print(relevant_wps.shape)
        #print(relevant_wps)
        points = np.asarray(self.hg.world_point_to_map_point(relevant_wps).astype(int).reshape(-1,3,3))
        return self.update_swept_volume_with_polygon(points,swept_volume,thickness = 0.035)


    def update_swept_volume_with_rectangle(self, main_axis, swept_volume):
        """
        Updates the swept volume with a rectangular swept region.
        Args:
            main_axis (np.ndarray): Main axis points.
            swept_volume (np.ndarray): 3D swept volume grid.
        Returns:
            np.ndarray: Updated swept volume.
        """
        arm_radius = 0.035

        A, B = main_axis[:, 0, :], main_axis[:, 1, :]
        direction = A - B
        direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)

        # Compute perpendicular vector
        perp = direction[:,[1,0,2]]
        perp[:, 0] = -perp[:, 0]

        # Define rectangle vertices
        BL, BR = A - arm_radius * perp, A + arm_radius * perp
        TL, TR = B - arm_radius * perp, B + arm_radius * perp
        rectangle = np.stack([BL, BR, TR, TL], axis=1)

        points = np.asarray(self.hg.world_point_to_map_point(rectangle.reshape(-1,3)).astype(int).reshape(rectangle.shape))
        return self.update_swept_volume_with_polygon(points, swept_volume, thickness=0.07)


class freeSpaceCalculator:
    def __init__(self, xlims=(-0.5, 0.5), ylims=(-0.5, -1.2), zlims=(-0.895, -1.405), resolution=0.005):
        """
        Initializes the FreeSpaceCalculator with the given space limits and resolution.

        Args:
            xlims (tuple): X-axis limits.
            ylims (tuple): Y-axis limits.
            zlims (tuple): Z-axis limits.
            resolution (float): Grid resolution.
        """
        self.resolution = resolution
        self.xlims = xlims
        self.ylims = ylims
        self.zlims = zlims

        # Create voxel grid
        xrange = np.arange(xlims[0], xlims[1], resolution)
        yrange = np.arange(ylims[0], ylims[1], -resolution)
        zrange = np.arange(zlims[0], zlims[1], -resolution)

        vox_coords_x, vox_coords_y, vox_coords_z = np.meshgrid(xrange, yrange, zrange, indexing='ij')
        all_coords = np.stack((vox_coords_x, vox_coords_y, vox_coords_z), axis=-1)

        # Add homogeneous coordinate
        ones = np.ones(all_coords.shape[:-1] + (1,))
        self.all_coords = np.concatenate((all_coords, ones), axis=-1)

        # Convert to CuPy for GPU acceleration
        self.reshaped_coords = cp.asarray(self.all_coords.reshape(-1, 4))


    def get_pcd(self, depth, intrinsic, extrinsic):
        """
        Converts depth image to a 3D point cloud.
        Args:
            depth (numpy.ndarray): Depth image.
            intrinsic (numpy.ndarray): Camera intrinsic matrix.
            extrinsic (numpy.ndarray): Camera extrinsic matrix.

        Returns:
            cupy.ndarray: Point cloud as an array.
        """

        depth_image = o3d.geometry.Image(depth.astype(np.float32))
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, intrinsic)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image, camera_intrinsic, extrinsic=extrinsic, depth_scale=1)
        return cp.asarray(np.asarray(pcd.points))


    def get_occupied_space(self, intrinsic, extrinsic, depth, free_space):
        """
        Computes occupied space from a depth image.

        Args:
            intrinsic (numpy.ndarray): Camera intrinsic matrix.
            extrinsic (numpy.ndarray): Camera extrinsic matrix.
            depth (numpy.ndarray): Depth image.
            free_space (cupy.ndarray): Precomputed free space array.

        Returns:
            cupy.ndarray: Boolean mask of occupied voxels.
        """
        pcd = self.get_pcd(depth, intrinsic, extrinsic)
        pcd[:, 1:] *= -1  # Invert y and z axes

        # Recenter point cloud around origin
        pcd[:, 0] -= self.xlims[0]
        pcd[:, 1] += self.ylims[0]
        pcd[:, 2] += self.zlims[0]

        # Quantize points w.r.t resolution
        pcd = cp.floor(pcd / self.resolution).astype(int)

        # Create empty occupied space array
        occupied = cp.zeros_like(free_space, dtype=bool)

        # Filter valid coordinates
        valid_mask = cp.all(pcd >= 0, axis=1) & \
                     (pcd[:, 0] < occupied.shape[0]) & \
                     (pcd[:, 1] < occupied.shape[1]) & \
                     (pcd[:, 2] < occupied.shape[2])
        pcd = pcd[valid_mask]

        # Mark occupied positions
        occupied[pcd[:, 0], pcd[:, 1], pcd[:, 2]] = True

        return occupied


    def get_free_space(self, intrinsic, extrinsic, depth):
        """
        Computes free space from a depth image.

        Args:
            intrinsic (numpy.ndarray): Camera intrinsic matrix.
            extrinsic (numpy.ndarray): Camera extrinsic matrix.
            depth (numpy.ndarray): Depth image.

        Returns:
            tuple: Free space (numpy.ndarray) and occupied space (numpy.ndarray).
        """
        P = cp.asarray(np.matmul(intrinsic, extrinsic[:3, :]))
        projected_coords = cp.matmul(P, self.reshaped_coords.T).T
        depth_values = projected_coords[:, 2:]

        # Compute pixel coordinates
        uv = cp.around((projected_coords[:, :2] / depth_values)).astype(int)

        # Handle invalid depth values
        depth_values[depth_values < 0] = 1e6  # Set large value for negative depths
        less_than_min = uv < 0

        more_than_max = uv >= cp.asarray([640, 480])[::-1]
        invalid_mask = cp.logical_or(less_than_min, more_than_max)
        uv[invalid_mask] = 0

        # Read depth map values
        depth_map_values = cp.asarray(depth)[uv[:, 1], uv[:, 0]]
        depth_map_values[invalid_mask.sum(axis=1) > 0] = -100000  # Assign low value for invalid points

        # Compute free space mask
        free_mask = depth_values.flatten() < depth_map_values.flatten() - self.resolution / 2
        free_mask[invalid_mask.sum(axis=1)] = False
        free = free_mask.reshape(self.all_coords[:, :, :, 0].shape)

        # Compute occupied space
        occupied = self.get_occupied_space(intrinsic, extrinsic, depth, free)
        occupied[:, :, 50:] = False  # Remove higher occupied areas
        free[occupied] = False

        # Ensure continuity in space occupancy
        free = cp.cumsum(free, axis=2) > 0
        occupied = cp.cumsum(occupied[:, :, ::-1], axis=2)[:, :, ::-1] > 0

        # then we need to move axis 0 to 1 and invert axis 1 afterwards
        free_voxels = free[::-1, :, :]
        occupied = occupied[::-1, :, :]
        return free_voxels.get(), occupied.get()


    def get_free_space_coordinates(self, intrinsic, extrinsic, depth):
        """
        Retrieves the 3D coordinates of free space.

        Args:
            intrinsic (numpy.ndarray): Camera intrinsic matrix.
            extrinsic (numpy.ndarray): Camera extrinsic matrix.
            depth (numpy.ndarray): Depth image.

        Returns:
            numpy.ndarray: Free space coordinates.
        """
        free_space, _ = self.get_free_space(intrinsic, extrinsic, depth)
        free_voxel_coords = self.all_coords[free_space]
        return free_voxel_coords[:, :3]


class HeightmapGeneration:
    """
    Generates a heightmap from 3D point cloud data and provides utility functions
    for mapping between world coordinates and pixel/grid coordinates.
    """
    def __init__(self, height_resolution: float, mapping_version: str = "MEM", n_classes: int = 15):
        """
        Initializes the HeightmapGeneration class with predefined mapping parameters.
        Args:
            height_resolution (float): The resolution of the height dimension.
            mapping_version (str): Mapping version ('old' or 'new').
            n_classes (int): Number of semantic classes.
        """
        self.height_resolution = height_resolution
        self.n_classes = n_classes

        self.bounds, self.pixel_size, self.default_z = self._get_mapping_params(mapping_version)


    def _get_mapping_params(self, version: str):
        """
        Retrieves mapping parameters based on the specified version.
        Args:
            version (str): Mapping version ('old' or 'MEM').
        Returns:
            tuple: Bounds, pixel size, and default z-value.
        """
        if version == "MEM":
            return (
                np.array([[-0.5, 0.5], [0.5, 1.2], [0.89, 1.1690]]),
                np.array([0.005, 0.005]),
                0.91
            )
        elif version == "old":
            return (
                np.array([[-0.5, 0.5], [0.5, 1.1], [0.89, 1.170]]),
                np.array([0.002082, 0.001382]),
                0.97
            )
        else:
            raise ValueError("Invalid mapping version specified. Choose 'MEM' or 'old'.")


    def get_heightmap(self, points, semantic):
        """
        Generates a heightmap and semantic/instance maps from a 3D point cloud.
        Args:
            points: 3D point cloud data.
            semantic: Semantic labels corresponding to the points.
        Returns:
            dict: Heightmap, semantic map, and instance map.
        """

        points = cp.asarray(points)
        semantic = cp.asarray(semantic)
        # Calculate width and height based on the new point cloud dimensions and pixel size
        width = int(cp.round((self.bounds[0, 1] - self.bounds[0, 0]) / self.pixel_size[0]))
        height = int(cp.round((self.bounds[1, 1] - self.bounds[1, 0]) / self.pixel_size[1]))

        #if the semantic map is contains both semantics and instance information, we create a top-down semantic-only vector
        if semantic.ndim > 2:
            instance, semantic = semantic[:, :, 0], semantic[:, :, 1]
        else:
            instance = semantic.copy()

        # Initialize heightmap and semanticmap with dimensions matching the new point cloud layout
        heightmap = cp.zeros((height, width,2), dtype=cp.float32)
        semanticmap = cp.zeros((height, width), dtype=cp.float32)
        instancemap = cp.zeros((height, width), dtype=cp.float32)
        semanticmap[:,:] = self.n_classes-1
        instancemap[:,:] = -1

        if points.size != 0:
            # Filter out 3D points that are outside the predefined bounds
            ix = (points[..., 0] >= self.bounds[0, 0]) & (points[..., 0] < self.bounds[0, 1])
            iy = (points[..., 1] >= self.bounds[1, 0]) & (points[..., 1] < self.bounds[1, 1])
            iz = (points[..., 2] >= self.bounds[2, 0]) & (points[..., 2] < self.bounds[2, 1])
            valid = ix & iy & iz

            # Apply the valid mask to filter points
            points = points[valid]
            # Ensure semantic is reshaped to the new point cloud layout (480, 640)
            semantic = semantic.reshape(480, 640)[valid]
            instance = instance.reshape(480, 640)[valid]


            # Sort 3D points by their z-values to simulate z-buffering
            iz = np.argsort(points[:, -1])
            points = points[iz]
            semantic = semantic[iz]
            instance = instance[iz]

            # Compute pixel indices for heightmap and semanticmap
            px = (cp.floor((points[:, 0] - self.bounds[0, 0]) / self.pixel_size[0])).astype(cp.int32)
            py = (cp.floor((points[:, 1] - self.bounds[1, 0]) / self.pixel_size[1])).astype(cp.int32)

            # Clip indices to avoid out-of-bound errors
            px = cp.clip(px, 0, width - 1)
            py = cp.clip(py, 0, height - 1)

            # Assign height and semantic values to the maps - as well as the border pixel binary mask - True -> pixel comes from a border
            heightmap[py, px, 0] = points[:, 2] - self.bounds[2, 0]
            heightmap[py, px, 1] = points[:, 3]
            semanticmap[py, px] = semantic[:]
            instancemap[py, px] = instance[:]

        data = {"height_map": heightmap, "semantic_map": semanticmap, "instance_map": instancemap}
        return data


    def preprocess_heightmap(self, hm):
        """
        Applies morphological processing to refine heightmap data.
        Args:
            hm (np.ndarray): Raw heightmap data.
        Returns:
            tuple: Processed heightmap and binarized version.
        """
        kernel = np.ones((3, 3), np.uint8)
        preprocessed_hm = cv2.morphologyEx(cp.asnumpy(hm)[:, :, 0].astype(np.float32), cv2.MORPH_CLOSE, kernel, iterations=2)
        preprocessed_mask = cv2.morphologyEx(cp.asnumpy(hm)[:, :, 1].astype(np.float32), cv2.MORPH_CLOSE, kernel, iterations=2)

        preprocessed_all = np.stack((preprocessed_hm, preprocessed_mask), axis=-1)
        hm_binarized = np.where(preprocessed_hm >= 0.075, 255, np.where(preprocessed_hm == 0, 100, 0))

        return preprocessed_all, hm_binarized


    def map_pixel_to_world_point(self, map_pix: np.ndarray):
        return np.array([
            (map_pix[0] * self.pixel_size[0]) + self.bounds[0, 0],
            (map_pix[1] * self.pixel_size[1]) + self.bounds[1, 0],
            self.default_z
        ])


    def map_point_to_world_point(self, map_point):
        return cp.asarray([
            (map_point[0] * self.pixel_size[0]) + self.bounds[0, 0],
            (map_point[1] * self.pixel_size[1]) + self.bounds[1, 0],
            (map_point[2] * self.height_resolution) + self.default_z
        ])


    def world_point_to_map_point(self, world_point):
        world_point = np.asarray(world_point)
        orig_shape_len = len(world_point.shape)
        if len(world_point.shape) != 2:
            world_point = np.expand_dims(world_point, axis=0)
        map_point = np.zeros((world_point.shape[0], 3))
        map_point[:, 0] = (np.floor((world_point[:, 0] - self.bounds[0, 0]) / self.pixel_size[0])).astype(np.int32)
        map_point[:, 1] = (np.floor((world_point[:, 1] - self.bounds[1, 0]) / self.pixel_size[1])).astype(np.int32)
        map_point[:, 2] = (np.floor((world_point[:, 2] - self.default_z) / self.height_resolution)).astype(np.int32)
        if orig_shape_len != 2:
            map_point = map_point.reshape(-1)
        return np.asnumpy(map_point)


    def world_point_to_map_point(self, world_point: np.ndarray):
        world_point = np.atleast_2d(world_point)
        map_point = np.zeros((world_point.shape[0], 3), dtype=np.int32)

        map_point[:, 0] = np.floor((world_point[:, 0] - self.bounds[0, 0]) / self.pixel_size[0]).astype(np.int32)
        map_point[:, 1] = np.floor((world_point[:, 1] - self.bounds[1, 0]) / self.pixel_size[1]).astype(np.int32)
        map_point[:, 2] = np.floor((world_point[:, 2] - self.default_z) / self.height_resolution).astype(np.int32)

        return map_point.reshape(-1) if world_point.shape[0] == 1 else map_point


    def world_point_to_map_pixel(self, world_point: np.ndarray):
        px = int(np.floor((world_point[0] - self.bounds[0, 0]) / self.pixel_size[0]))
        py = int(np.floor((world_point[1] - self.bounds[1, 0]) / self.pixel_size[1]))
        return px, py


    def draw_borders(self, heightmap, intensity=1.):
        heightmap = self.draw_shelf_borders_hm([0.3925, 0.7, 0.99], [0.3925, 1.089, 0.99], heightmap, intensity)
        heightmap = self.draw_shelf_borders_hm([-0.3975, 0.7, 0.99], [-0.3975, 1.089, 0.99], heightmap, intensity)
        heightmap = self.draw_shelf_borders_hm([0.3925, 1.089, 0.99], [-0.3975, 1.089, 0.99], heightmap, intensity)
        return heightmap


    def draw_shelf_borders_cm(self, point_1, point_2, img):
        px_1 = (np.floor((point_1[0] - self.bounds[0, 0]) / self.pixel_size[0])).astype(np.int32)
        py_1 = (np.floor((point_1[1] - self.bounds[1, 0]) / self.pixel_size[1])).astype(np.int32)
        px_2 = (np.floor((point_2[0] - self.bounds[0, 0]) / self.pixel_size[0])).astype(np.int32)
        py_2 = (np.floor((point_2[1] - self.bounds[1, 0]) / self.pixel_size[1])).astype(np.int32)
        return cv2.line(img, (px_1, py_1), (px_2, py_2), [118,79,45], 3)


    def draw_shelf_borders_hm(self, point_1, point_2, img, intensity):
        px_1 = (np.floor((point_1[0] - self.bounds[0, 0]) / self.pixel_size[0])).astype(np.int32)
        py_1 = (np.floor((point_1[1] - self.bounds[1, 0]) / self.pixel_size[1])).astype(np.int32)
        px_2 = (np.floor((point_2[0] - self.bounds[0, 0]) / self.pixel_size[0])).astype(np.int32)
        py_2 = (np.floor((point_2[1] - self.bounds[1, 0]) / self.pixel_size[1])).astype(np.int32)
        return cv2.line(img, (int(px_1), int(py_1)), (int(px_2), int(py_2)), intensity, 5)


if __name__ == '__main__':
    pass
