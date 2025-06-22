import time

import open3d as o3d
import cupy as cp
import numpy as np
import pdb
from tqdm import tqdm
from copy import deepcopy
import cupyx
import gc
#import torch
#import torch.special as special
import scipy

def update_mapper_preprocessed(hms,semantic_hms,run,point,this_mapper):
    this_mapper.mapping(hm=hms[run,point],hm_b=hms[run,point],sm=semantic_hms[run,point])
    return this_mapper


def update_voxelmapper(depth,semantic_hms,run,point,this_mapper,subsample):
    this_mapper.mapping(depth[point][::subsample,::subsample],semantic_hms[point],point)
    return this_mapper


def extract_region_of_interest_and_add_walls(this_mapper, raw_prob_map=False):
    """
    Extracts the region of interest from a 3D probability map and adds walls
    for proper bounding.

    Parameters:
    - this_mapper: The probability map class
    - raw_prob_map (bool): If True, assumes `this_mapper` is a direct probability map.
                            Otherwise, extracts `prob_map` from `this_mapper`.

    Returns:
    - pm (cp.ndarray): Processed probability map with walls added.
    """

    # Convert to CuPy array
    vg = cp.asarray(this_mapper) if raw_prob_map else cp.asarray(this_mapper.prob_map)

    # Ensure correct shape (140, 200, 102)
    expected_shape = (140, 200, 102)
    if vg.shape != expected_shape:
        padded_vg = cp.zeros(expected_shape, dtype=vg.dtype)
        padded_vg[10:-10, :, :] = vg  # Preserve center data
    else:
        padded_vg = vg

    # Add walls (Mark areas as occupied)
    padded_vg[:, 22, :] = 1   # Left wall
    padded_vg[:, -22, :] = 1  # Right wall
    padded_vg[119, :, :] = 1  # Back wall
    padded_vg[:, :, 57:] = 1  # Top covered (reducing representation)

    # Remove irrelevant areas (outside shelf)
    padded_vg[:40, :, :] = 0   # Front region
    padded_vg[-20:, :, :] = 0  # Back region
    padded_vg[:, :20, :] = 0   # Left side region
    padded_vg[:, -20:, :] = 0  # Right side region

    # Clip values to prevent extreme probabilities
    pm = cp.clip(padded_vg, 0.0001, 0.9999)

    return pm


def get_map_entropy(this_mapper):
    """
    Computes the entropy of a probability map.
    Parameters: The probability map class
    Returns: The mean entropy of the probability map (float)
    """

    pm = extract_region_of_interest_and_add_walls(this_mapper).astype(cp.float32)
    entropy_map = -pm * cp.log(pm) - (1 - pm) * cp.log(1 - pm)
    return cp.asnumpy(entropy_map.mean())


class InfoGainEval:
    def __init__(self,camera_matrices_dir,subsample = 8,width = 640,height=480,start_k = 60,max_k = 420,step =0.002,
                 object_origin=[-0.4,-0.65,-0.895], parallel = False,parallel_passes = 4,occupancy_thold = 0.95,
                 cached = True):
        self.resolution = 0.005
        xrange = np.arange(-0.4,0.40,self.resolution)
        zrange = np.arange(-0.895,-1.18,-self.resolution)
        yrange = np.arange(-0.65,-1.20,-self.resolution)
        self.eps = 0.0000001
        tmp = np.meshgrid(xrange,yrange,zrange,indexing = 'ij')
        vox_coords_x,vox_coords_y,vox_coords_z = tmp
        vox_coords_x = vox_coords_x
        vox_coords_y = vox_coords_y
        vox_coords_z = vox_coords_z
        self.max_x_bounds = vox_coords_x.max()
        self.min_x_bounds = vox_coords_x.min()
        self.min_y_bounds = vox_coords_y.min()
        self.max_y_bounds = vox_coords_y.max()
        self.max_z_bounds = vox_coords_z.max()
        self.min_z_bounds = vox_coords_z.min()
        self.camera_matrices = np.load(camera_matrices_dir,allow_pickle = True)['obj_ids']
        self.subsample = subsample
        self.width = width
        self.height = height
        self.start_k = start_k
        self.max_k = max_k
        self.step = step
        self.object_origin = object_origin
        self.ks = cp.asarray(np.arange(self.start_k,self.start_k+max_k+1,1)*step)
        self.parallel = parallel
        self.parallel_passes = parallel_passes
        self.occupancy_thold = occupancy_thold
        self.cached = cached
        self.init_camera_rays()
        if(self.cached):
            self.get_all_raycasts()


    def init_camera_rays(self):
        self.all_rays = []
        
        for camera in range(len(self.camera_matrices)):
            extrinsic = o3d.core.Tensor(np.linalg.inv(self.camera_matrices[camera]['o3d_extrinsic_matrix'].reshape(4,4,order = 'C')))
            intrinsic = o3d.core.Tensor(self.camera_matrices[camera]['intrinsic_matrix'].reshape(3,3,order = 'C'))
            rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(intrinsic_matrix=intrinsic,extrinsic_matrix = extrinsic,
                                                                    width_px=self.width,height_px=self.height)
            self.all_rays.append(rays[::self.subsample,::self.subsample,:].numpy().reshape((-1,6)))


    def get_raycast(self,camera_idx,as_cupy = False):
        if(self.cached):
            return self.all_raycasts[camera_idx]
        else:
            if(as_cupy):
                return self.get_raycast_indices(camera_idx)
            else:
                return cp.asnumpy(self.get_raycast_indices(camera_idx))


    def get_raycast_indices(self, camera):
        """
        Computes the voxel indices where rays from a given camera intersect within a 3D grid.

        The function:
        1. Extracts ray origins and directions from the given camera.
        2. Computes the full ray path using a scaling factor `ks`.
        3. Identifies which ray points lie within predefined 3D bounds.
        4. Converts valid hit coordinates into discrete voxel indices.
        5. Assigns a unique base offset to each valid voxel index.
        6. Removes duplicate ray indices along the path to avoid redundant storage.
        7. Stores the total shape of the resulting raycast data for later use.

        Parameters:
        - camera: Index of the camera from which rays are cast.

        Returns:
        - raycast (CupPy array): A 3D array of voxel indices where the rays hit valid points.
        """

        rays = self.all_rays[camera]

        # Separate ray origins and directions
        origins_cp = cp.asarray(rays[:, :3])
        directions_cp = cp.asarray(rays[:, 3:])

        # Compute full ray path
        full_ray = origins_cp[:, None, :] + self.ks[None, :, None] * directions_cp[:, None, :]

        # Check if rays are within bounds
        within_bounds = (
                (self.min_x_bounds <= full_ray[:, :, 0]) & (full_ray[:, :, 0] <= self.max_x_bounds) &
                (self.min_y_bounds <= full_ray[:, :, 1]) & (full_ray[:, :, 1] <= self.max_y_bounds) &
                (self.min_z_bounds <= full_ray[:, :, 2]) & (full_ray[:, :, 2] <= self.max_z_bounds)
        )

        # Get hit coordinates
        hit_coords = full_ray[within_bounds]

        # Convert to voxel indices
        shifted_indices = cp.abs(((hit_coords - cp.array(self.object_origin)) / self.resolution).astype(int))

        # Prepare raycast output
        raycast = cp.zeros((within_bounds.shape[0], within_bounds.shape[1], 3), dtype=cp.uint8)
        raycast[within_bounds] = shifted_indices + cp.array([20, 31, 1], dtype=cp.uint8)

        # Remove duplicate indices along the ray path
        diff = cp.diff(raycast, axis=1, prepend=0)
        repeats = cp.linalg.norm(diff, axis=-1) == 0
        raycast[repeats] = 0

        # Store shape metadata
        self.all_cast_shape = raycast.shape[0] * raycast.shape[1]

        return raycast


    def get_occlusion_aware_information_gain(self,raycast,voxels_of_interest, use_beta_entropy=False, beta_entropy=None, n_cameras = None,information_voxels_of_interest = None):
            negative_indices = cp.all(raycast == 0,axis = 2)

            all_hits_shape = self.all_cast_shape-negative_indices.sum()

            # pdb.set_trace()
            readings = cp.ascontiguousarray(voxels_of_interest[raycast[:,:,1],voxels_of_interest.shape[1]-raycast[:,:,0],raycast[:,:,2]])
            # we make high confidence regions fully opaque
            readings[readings> self.occupancy_thold] = 1
            # we make very low probability regions also fully transparent
            readings[readings<1-self.occupancy_thold] = 0
            readings = cp.clip(readings*(1-negative_indices.astype(cp.uint8)),self.eps,1-self.eps)
            probability_of_transmission = cp.roll(cp.cumprod((1-readings),axis = 1),1,axis = 1)
            inv_raycast_y = voxels_of_interest.shape[1] - raycast[:, :, 0]
            # pdb.set_trace()
            if use_beta_entropy and beta_entropy is not None:
                H = cp.ascontiguousarray(beta_entropy[raycast[:, :, 1], inv_raycast_y, raycast[:, :, 2]])
                H = H * (1 - negative_indices)
            else:
                if(information_voxels_of_interest is None):
                    H = -readings*cp.log(readings)-(1-readings)*cp.log(1-readings)
                    H = H*(1-negative_indices) # setting zero entropy for empty voxels
                    # we adjust the entropy to value more points closer to the ground (of the parabollic kind, with 3 times the value at the middle)
                else:
                    nreadings = information_voxels_of_interest[raycast[:,:,1],inv_raycast_y,raycast[:,:,2]]
                    nreadings[nreadings> self.occupancy_thold] = 1
                    # ew make very low probability regions also fully transparent
                    nreadings[nreadings<1-self.occupancy_thold] = 0
                    nreadings = cp.clip(nreadings*(1-negative_indices.astype(cp.uint8)),self.eps,1-self.eps)
                    H = -nreadings*cp.log(nreadings)-(1-nreadings)*cp.log(1-nreadings)


            del readings
            if(n_cameras is None):
                occlusion_aware_vi = H*probability_of_transmission
                occlusion_aware_vi = self.reduce_vi(occlusion_aware_vi)

            else:
                occlusion_aware_vi = ((H[:,1:]*probability_of_transmission[:,:-1]).sum(axis = 1)+H[:,0]).reshape((n_cameras,-1)).mean(axis = 1)
            del H,probability_of_transmission
            return occlusion_aware_vi


    def reduce_vi(self,vi):
        prop = 7/102
        return vi.sum()*prop


    def alt_get_occlusion_aware_information_gain(self,raycast,prob,prob_bar,log_prob,log_prob_bar, use_beta_entropy=False, beta_entropy=None, n_cameras = None,information_voxels_of_interest = None):
        negative_indices = cp.all(raycast == 0,axis = 2)

        all_hits_shape = self.all_cast_shape-negative_indices.sum()
        inv_raycast_y = prob.shape[1]-raycast[:,:,0]
        readings = cp.ascontiguousarray(prob[raycast[:,:,1],inv_raycast_y,raycast[:,:,2]])
        readings_bar =  cp.ascontiguousarray(prob_bar[raycast[:,:,1],inv_raycast_y,raycast[:,:,2]])
        log_readings =  cp.ascontiguousarray(log_prob[raycast[:,:,1],inv_raycast_y,raycast[:,:,2]])
        log_readings_bar =  cp.ascontiguousarray(log_prob_bar[raycast[:,:,1],inv_raycast_y,raycast[:,:,2]])
        probability_of_transmission = cp.roll(cp.cumprod((readings_bar),axis = 1),1,axis = 1)
        probability_of_transmission[:,0] = 1
        if use_beta_entropy and beta_entropy is not None:
            H = cp.ascontiguousarray(beta_entropy[raycast[:, :, 1], inv_raycast_y, raycast[:, :, 2]])
            H = H * (1 - negative_indices)
        else:
            if(information_voxels_of_interest is None):
                H = -readings*log_readings-readings_bar*log_readings_bar
                H = H*(1-negative_indices)            # we adjust the entropy to value more points closer to the ground (of the parabollic kind, with 3 times the value at the middle)

            else:
                nreadings = information_voxels_of_interest[raycast[:,:,1],voxels_of_interest.shape[1]-raycast[:,:,0],raycast[:,:,2]]
                nreadings[nreadings> self.occupancy_thold] = 1
                # ew make very low probability regions also fully transparent
                nreadings[nreadings<1-self.occupancy_thold] = 0
                nreadings = cp.clip(nreadings*(1-negative_indices.astype(cp.uint8)),self.eps,1-self.eps)
                H = -nreadings*cp.log(nreadings)-(1-nreadings)*cp.log(1-nreadings)

        del readings
        if(n_cameras is None):
            occlusion_aware_vi = H*probability_of_transmission#.sum(axis = 1).sum()#/all_hits_shape
            #print("vi: ",probability_of_transmission.shape)

            occlusion_aware_vi = self.reduce_vi(occlusion_aware_vi)
            # occlusion_aware_vi = H.mean()
        else:
            occlusion_aware_vi = ((H[:,1:]*probability_of_transmission[:,:-1]).sum(axis = 1)+H[:,0]).reshape((n_cameras,-1)).mean(axis = 1)

        del H,probability_of_transmission
        return occlusion_aware_vi


    def alt_get_all_igs(self,voxels_of_interest, use_beta_entropy=False, beta_entropy=None, skip = 1):
        igs = []
        prob = cp.copy(voxels_of_interest)
        prob[0,0,0] = 0
        prob[prob>self.occupancy_thold] = 1
        # we make very low probability regions also fully transparent
        prob[prob<1-self.occupancy_thold] = 0
        prob = cp.clip(prob,self.eps,1-self.eps)
        prob_bar = 1-prob
        log_prob = cp.log(prob)
        log_prob_bar = cp.log(prob_bar)
        for camera in range(0,len(self.camera_matrices),skip):
            raycast = self.get_raycast(camera,as_cupy=True)
            if(self.cached):
                raycast = cp.asarray(raycast)
            information_gain = self.alt_get_occlusion_aware_information_gain(raycast,prob,prob_bar,log_prob,
                                                                             log_prob_bar,
                                                                             use_beta_entropy=use_beta_entropy,
                                                                             beta_entropy=beta_entropy)
            igs.append(cp.asnumpy(information_gain).tolist())
            del raycast
        return igs


    def get_beta_entropy(self, beta_dist):
        """ Compute the entropy of a Beta distribution Beta(alpha, beta).
        Args:
            alpha (torch.Tensor): Shape parameters alpha
            beta (torch.Tensor): Shape parameters beta
        Returns:
            torch.Tensor: Entropy of the Beta distribution
        """
        alpha = beta_dist[0, 1::2, :, :].cpu().numpy()
        beta = beta_dist[0, ::2, :, :].cpu().numpy()

        entropy = (scipy.special.betaln(alpha, beta) # log B(alpha, beta)
                - (alpha - 1) * scipy.special.psi(alpha)
                - (beta - 1) * scipy.special.psi(beta)
                + (alpha + beta - 2) * scipy.special.psi(alpha + beta)
        )
        return cp.asarray(entropy)



    def get_dirichlet_entropy(self, alpha):
        """Computes the entropy of a Dirichlet distribution"""
        alpha = alpha.cpu().numpy()[0]
        alpha_sum = np.sum(alpha, axis=0)
        K = len(alpha)

        entropy = ((scipy.special.gammaln(alpha_sum) - np.sum(scipy.special.gammaln(alpha), axis=0))
                    + ((alpha_sum - K) * scipy.special.psi(alpha_sum))
                    - (np.sum((alpha - 1) * scipy.special.psi(alpha))))

        return cp.asarray(entropy)


    def visualize_beta_entropy(self, beta_entropy):
        import matplotlib
        # Get coordinate grids
        z, y, x = np.meshgrid(
            np.arange(beta_entropy.shape[0]),  # Depth dimension (slices)
            np.arange(beta_entropy.shape[1]),  # Height dimension
            np.arange(beta_entropy.shape[2]),  # Width dimension
            indexing="ij"
        )

        # Flatten for scatter plot
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        entropy_flat = beta_entropy.flatten()

        # Normalize entropy values for color mapping
        entropy_norm = (entropy_flat - entropy_flat.min()) / (entropy_flat.max() - entropy_flat.min())
        mask_1 = entropy_norm > 0.3
        mask_2 = entropy_norm > 0.5
        mask_3 = entropy_norm > 0.8
        masks = [mask_1, mask_2, mask_3]
        thresholds = [0.3, 0.5, 0.8]

        for i, mask in enumerate(masks):
            #entropy_norm = entropy_norm[mask]
            # Convert to Open3D PointCloud format
            points = np.vstack((x_flat.copy()[mask], y_flat.copy()[mask], z_flat.copy()[mask])).T  # Shape (N, 3)
            viridis = matplotlib.cm.get_cmap("viridis")
            colors = viridis(entropy_norm.copy()[mask])[:, :3]  # Extract RGB from colormap
            # Set transparency (lower entropy → more transparent)
            alpha_values = np.clip(entropy_norm, 0.2, 1.0)  # Min transparency at 0.2

            # Convert to Open3D format
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)  # RGB colors

            # Visualize in Open3D
            o3d.io.write_point_cloud("./entropy_pointcloud_"+ str(thresholds[i]) + ".ply", pcd)

        #o3d.visualization.draw_geometries([pcd], window_name="Entropy Point Cloud")

    def get_all_raycasts(self):
        self.all_raycasts = None
        for camera_index,camera in enumerate(range(len(self.camera_matrices))):
            raycast = self.get_raycast_indices(camera)
            if(self.all_raycasts is None):
                raycast_shape = raycast.shape
                final_shape = [len(self.camera_matrices),*raycast_shape]
                self.all_raycasts = cupyx.zeros_pinned(final_shape,dtype = cp.uint8)
            self.all_raycasts[camera_index,:,:,:] = raycast[:,:,:].get()
        gc.collect()
        if(self.parallel):
            self.original_shape = [300] + list(self.all_raycasts[0].shape)
            goal_shape = list(self.all_raycasts[0].shape)
            goal_shape[0] *=300
            self.raycast = cp.asarray(np.array(self.all_raycasts)).reshape(goal_shape)


    def get_all_igs(self,voxels_of_interest,skip = 1,start = 9):
        igs = []
        for camera in range(0,len(self.camera_matrices),skip):
            raycast = self.get_raycast(camera,as_cupy=True)
            if(self.cached):
                raycast = cp.asarray(raycast)
            # pdb.set_trace()
            information_gain = self.get_occlusion_aware_information_gain(raycast,voxels_of_interest)
            igs.append(cp.asnumpy(information_gain).tolist())
            del raycast
        return igs


    def get_all_subsequent_igs(self,voxels_of_interest, already_viewed_views, use_beta_entropy=False, beta_entropy=None):
        information_voxels_of_interest = cp.copy(voxels_of_interest)
        for view in already_viewed_views:
            raycast = self.get_raycast(view,as_cupy=True)
            information_voxels_of_interest[raycast[:,:,1],voxels_of_interest.shape[1]-raycast[:,:,0],raycast[:,:,2]] = 0
        igs = []

        for camera in range(len(self.camera_matrices)):
            raycast = self.get_raycast(camera,as_cupy=True)
            if(self.cached):
                raycast = cp.asarray(raycast)
            # pdb.set_trace()
            information_gain = self.get_occlusion_aware_information_gain(raycast,voxels_of_interest,
                                                                         use_beta_entropy=use_beta_entropy,
                                                                         beta_entropy=beta_entropy,
                                                                         information_voxels_of_interest=information_voxels_of_interest)
            igs.append(cp.asnumpy(information_gain).tolist())
            del raycast
        return igs


    def get_all_igs_parallel(self,voxels_of_interest):
        total_ig = []
        for i in range(self.parallel_passes):
            information_gain = self.get_occlusion_aware_information_gain(self.raycast[i*self.parallel_passes*self.original_shape[1]:(i+1)*(self.parallel_passes)*self.original_shape[1]],voxels_of_interest,n_cameras = 300//self.parallel_passes)            
            total_ig.append(cp.asnumpy(information_gain).tolist())
        return total_ig
        


if __name__=="__main__":
    from shelf_gym.utils.information_gain_utils import InfoGainEval, update_mapper_preprocessed
    from shelf_gym.utils.information_gain_utils import extract_region_of_interest_and_add_walls
    from matplotlib import pyplot as plt
    import cupy as cp
    import numpy as np
    import open3d as o3d
    import pickle
    from matplotlib import pyplot as plt
    from tqdm import tqdm
    from shelf_gym.utils.mapping_utils import BEVMapping
    import h5py 
    import seaborn as sns
    from copy import deepcopy

    ig_calc = InfoGainEval('../tasks/hallucination/data/map_completion_test/0/000000000/pre_action/camera_matrices.npz',subsample = 16)

    f = h5py.File('../tasks/hallucination/model_training/map_completion_enhanced.hdf5',mode = 'r')
    hms = f['hms']
    semantic_hms = f['semantic_hms']
    run = 10
    starting_obs = np.random.choice(np.arange(300))
    imgs = []
    heights = []
    observed_mapper = BEVMapping(hms[run,0],raw_hm_start = True,occupancy_threshold = 0.6)
    observed_mapper = update_mapper_preprocessed(hms,semantic_hms,run,starting_obs,observed_mapper)
    for i in tqdm(range(5)):
        voxels_of_interest = extract_region_of_interest_and_add_walls(observed_mapper)
        igs = ig_calc.get_all_igs(voxels_of_interest)
        print(igs)
        selected_view = np.argmax(igs)
        print(selected_view,igs[selected_view])
        observed_mapper = update_mapper_preprocessed(hms,semantic_hms,run,selected_view,observed_mapper)
        oc2,sem2 = observed_mapper.get_2D_representation()
        oc2 = cp.asnumpy(oc2)
        plt.imshow(hms[run,selected_view,:,:,0])
        plt.show(block = True)
        imgs.append(oc2[0][40:-20,20:-20][::-1,::-1])
        heights.append(oc2[1][40:-20,20:-20][::-1,::-1])
    #     imgs,Hns,H0 = get_information_gain_counterfactuals(hms,semantic_hms,run,observations)
    #     gain = H0-Hns
    #     best = np.argmax(gain)
    #     selected = np.argmax(igs)
    #     a = np.empty(len(gain), np.intp)
    #     a[np.argsort(-gain)] = np.arange(len(gain))
    #     print('run {} selected point {} is the {} best viewpoint - IG= {:.4f} vs best IG = {:.4f} '.format(run,selected,a[selected],gain[selected],gain[best]))


