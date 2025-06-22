import time

import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import torch
from shelf_gym.utils.scaling_utils import scale_semantic_map
from shelf_gym.utils.mapping_utils import freeSpaceCalculator
from shelf_gym.utils.information_gain_utils import extract_region_of_interest_and_add_walls
import torch.nn as nn
from copy import deepcopy
class EvaluationHelper:
    def __init__(self,camera_matrices_file, is_real_world_data=False):
        self.is_real_world_data = is_real_world_data
        self.FSC = freeSpaceCalculator()
        self.extrinsics = []
        self.intrinsics = []
        if is_real_world_data:
            camera_matrices = np.load(camera_matrices_file,allow_pickle = True)['matrices']
            for camera in range(len(camera_matrices)):
                self.extrinsics.append(
                    np.linalg.inv(camera_matrices[camera]['corrected_matrix'].reshape(4, 4, order='C')))
                self.intrinsics.append(np.array(
                    [910.6857299804688, 0.0, 642.8099975585938, 0.0, 910.9524536132812, 382.1358337402344, 0.0, 0.0,
                     1.0]).reshape(3, 3, order='C'))
        else:
            camera_matrices = np.load(camera_matrices_file,allow_pickle = True)['obj_ids']
            for camera in range(len(camera_matrices)):
                self.extrinsics.append(camera_matrices[camera]['o3d_extrinsic_matrix'].reshape(4, 4, order='C'))
                self.intrinsics.append(camera_matrices[camera]['intrinsic_matrix'].reshape(3, 3, order='C'))


    def get_map_completion(self,viewpoint,dataset,hms,depths,semantic_hms,map_completion_model,previous_map,previous_semantic_map):
        this_hm = hms[viewpoint][10:-10,:,0]
        this_semantic_hm = semantic_hms[viewpoint][10:-10]
        this_semantic_hm[this_hm == 0] = 15
        old_occupied,old_free = dataset.local_find_update_cells(this_hm[np.newaxis,:,:])

        free,occupied = self.FSC.get_free_space(self.intrinsics[viewpoint],self.extrinsics[viewpoint],depths[viewpoint].astype(float)/1000)
        free = np.moveaxis(free,[0,2],[2,0])[:,10:-10]
        occupied = torch.from_numpy(np.moveaxis(occupied,[0,2],[2,0])[:,10:-10])[np.newaxis,np.newaxis,:].to('cuda')
        
        old_free[0] = free
        free = torch.from_numpy(old_free[np.newaxis,:]).to('cuda')
        semantics = torch.from_numpy(this_semantic_hm[np.newaxis,np.newaxis,:,:]).to('cuda')
        batch = {'free':free,'semantic_hms':semantics,'occupied':occupied,'gt_3d':torch.eye(3),'gt_semantics':torch.eye(3)}

        with torch.no_grad():
            outputs = map_completion_model.get_outputs(batch,previous_map = previous_map,previous_semantic_map = previous_semantic_map)
        return outputs,free,this_semantic_hm,this_hm


    def get_direct_map_completion(self,viewpoint,dataset,this_hm,depth,this_semantic_hm,map_completion_model,
                                  previous_map,previous_semantic_map, intrinsic_matrix=None, extrinsic_matrix=None):

        this_hm = this_hm[10:-10,:,0]
        this_semantic_hm = this_semantic_hm[10:-10]
        this_semantic_hm[this_hm == 0] = 15
        old_occupied, old_free = dataset.local_find_update_cells(this_hm[np.newaxis,:,:])

        if intrinsic_matrix is None:
            intrinsic_matrix = self.intrinsics[viewpoint]
        if extrinsic_matrix is None:
            extrinsic_matrix = self.extrinsics[viewpoint]

        free, occupied = self.FSC.get_free_space(intrinsic_matrix,extrinsic_matrix, depth.astype(float)/1000)

        free = np.moveaxis(free,[0,2],[2,0])[:,10:-10]
        occupied = torch.from_numpy(np.moveaxis(occupied,[0,2],[2,0])[:,10:-10])[np.newaxis,np.newaxis,:].to('cuda')

        # # 1. pull out the [D,H,W] array
        voxel = occupied.cpu().numpy()[0, 0]
        heightmap = voxel.argmin(axis=0) * 0.005 # now shape (H, W)
        # # 5. visualize
        # f, ax = plt.subplots(1, 2)
        # ax[0].imshow(heightmap)
        # ax[1].imshow(this_hm)
        # plt.show()

        old_free[0] = free
        free = torch.from_numpy(old_free[np.newaxis,:]).to('cuda')
        semantics = torch.from_numpy(this_semantic_hm[np.newaxis,np.newaxis,:,:]).to('cuda')
        batch = {'free':free,'semantic_hms':semantics,'occupied':occupied,'gt_3d':torch.eye(3),'gt_semantics':torch.eye(3)}
        with torch.no_grad():
            outputs = map_completion_model.get_outputs(batch,previous_map = previous_map,previous_semantic_map = previous_semantic_map)
        return outputs, free, this_semantic_hm, heightmap,



def get_igs_for_map(previous_map, ig_calc, skip = 1, use_beta_entropy = False, camera_matrix=None, use_alternative=False, raycast=None):
    prob_map = get_prob_map(previous_map)
    prob_map = remove_edge_effects(prob_map)
    voxels_of_interest = extract_region_of_interest_and_add_walls(prob_map.squeeze().cpu().numpy(),raw_prob_map = True)
    beta_entropy = ig_calc.get_relative_beta_entropy(previous_map) if use_beta_entropy else None

    if camera_matrix is None:
        igs = ig_calc.get_all_igs(voxels_of_interest,
                                           use_beta_entropy=use_beta_entropy,
                                           beta_entropy=beta_entropy,
                                           skip=skip,
                                           use_alternative=use_alternative)
        return np.array(igs), None
    else:
        igs, raycast = ig_calc.get_ig_single_view(voxels_of_interest,
                                                  camera_matrix,
                                                  use_beta_entropy=use_beta_entropy,
                                                  beta_entropy=beta_entropy,
                                                  raycast=raycast)
        return np.array(igs), raycast

def get_uncertainty_for_map(ig_calc, camera_matrix, beta_map=None, dirichlet=None, calc_occ=False, n_classes=15, raycast=None):
    prob_map = get_prob_map(beta_map)
    prob_map = remove_edge_effects(prob_map)
    voxels_of_interest = extract_region_of_interest_and_add_walls(prob_map.squeeze().cpu().numpy(),raw_prob_map = True)
    prob = cp.copy(voxels_of_interest)
    prob[0, 0, 0] = 0
    prob[prob > ig_calc.occupancy_thold] = 1
    # we make very low probability regions also fully transparent
    prob[prob < 1 - ig_calc.occupancy_thold] = 0
    prob = cp.clip(prob, ig_calc.eps, 1 - ig_calc.eps)
    if raycast is None:
        raycast = ig_calc.get_raycast(camera_matrix=camera_matrix, as_cupy=True)

    if beta_map is not None and calc_occ:
        alpha = beta_map[:, ::2, :, :]
        beta = beta_map[:, 1::2, :, :]
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        epistemic_uncertainty = cp.asarray(variance.cpu().numpy())[0]
        max_entropy = (1 / 12)
    elif dirichlet is not None and not calc_occ:
        # epistemic uncertainty according to sensoy et al
        epistemic_uncertainty = cp.asarray(n_classes/dirichlet.sum(axis=1, keepdims=True)[0,0,:,:].cpu().numpy())
        # print(epistemic_uncertainty.size())
        # probs = (dirichlet / (dirichlet.sum(axis=1, keepdims=True) + ig_calc.eps))  # .cpu().numpy()
        # probs = torch.from_numpy(scale_semantic_map(probs.cpu().numpy(), axis=1)).to('cuda')
        # entropy = -(probs * torch.log(probs + ig_calc.eps)).sum(dim=1, keepdim=True)[0,0,:,:]
        # print(entropy.size())
        # entropy = cp.asarray(entropy.cpu().numpy())
        max_entropy =  np.log(n_classes) + 1e-8
    else:
        raise Exception("Decide for either occupancy or Semantic uncertainty")

    uncertainty_value = get_occlusion_aware_uncertainty_bonus(raycast, prob, epistemic_uncertainty, use_occ_unc=calc_occ, max_uncertainty_value=max_entropy)
    return uncertainty_value, raycast


def get_occlusion_aware_uncertainty_bonus(raycast, occupancy_map, uncertainty_map, max_uncertainty_value=1/12, use_occ_unc=False):
    """
    Compute an occlusion-aware bonus based on the uncertainty visible from this viewpoint.

    Parameters:
    - raycast: 3D int array [rays, steps, 3] containing (x, y, z) voxel indices along each ray
    - uncertainty_map: 3D array [X, Y, Z] with uncertainty values (e.g. beta entropy or semantic entropy)
    """
    negative_indices = cp.all(raycast == 0, axis=2)
    inv_raycast_y = uncertainty_map.shape[1] - raycast[:, :, 0]

    # Get the uncertainty values along each ray
    if use_occ_unc:
        H = cp.ascontiguousarray(uncertainty_map[raycast[:, :, 1], inv_raycast_y, raycast[:, :, 2]])
    else:
        H = cp.ascontiguousarray(uncertainty_map[raycast[:, :, 1], inv_raycast_y])
    H = H * (1 - negative_indices)

    if use_occ_unc:
        inv_occupancy_map = 1 - occupancy_map
        readings = cp.ascontiguousarray(inv_occupancy_map[raycast[:, :, 1], inv_raycast_y, raycast[:, :, 2]])
    else:
        inv_occupancy_map = 1 - occupancy_map.max(axis=2)
        readings = cp.ascontiguousarray(inv_occupancy_map[raycast[:, :, 1], inv_raycast_y])

    probability_of_transmission = cp.roll(cp.cumprod(readings, axis=1), 1, axis=1)
    probability_of_transmission[:, 0] = 1.0

    # Weighted actual uncertainty
    occlusion_aware_uncertainty = H * probability_of_transmission
    uncertainty_score = occlusion_aware_uncertainty.sum()

    # Weighted max uncertainty for this view (what it *could* be)
    max_uncertainty = cp.full_like(H, max_uncertainty_value)
    max_occlusion_aware_uncertainty = max_uncertainty * probability_of_transmission
    max_uncertainty_score = max_occlusion_aware_uncertainty.sum()
    del H, probability_of_transmission

    if max_uncertainty_score == 0:
        return 0.0  # nothing visible

    # Normalize
    print(occlusion_aware_uncertainty.size)
    print(occlusion_aware_uncertainty.shape)
    normalized_uncertainty_score = uncertainty_score / occlusion_aware_uncertainty.size #/ max_uncertainty_score
    return float(normalized_uncertainty_score)


def get_subsequent_igs_for_map(previous_map, observed_viewpoints, ig_calc, use_beta_entropy = False):
    prob_map = get_prob_map(previous_map)
    prob_map = remove_edge_effects(prob_map)
    voxels_of_interest = extract_region_of_interest_and_add_walls(prob_map.squeeze().cpu().numpy(),raw_prob_map = True)
    beta_entropy = ig_calc.get_relative_beta_entropy(previous_map) if use_beta_entropy else None
    igs = np.array(ig_calc.get_all_subsequent_igs(voxels_of_interest,
                                                  observed_viewpoints,
                                                  use_beta_entropy=use_beta_entropy,
                                                  beta_entropy=beta_entropy))
    return igs


def get_prob_map(out_map):
    alpha = out_map[:,1::2,:,:]
    beta = out_map[:,::2,:,:]
    probs = alpha/(alpha+beta)
    probs = torch.permute(probs,[0,2,3,1])
    return probs


def get_valid_choices():
    return  ['hard','random','harder','new_hard','intermediate']


def remove_edge_effects(oc,max_occ_thold = 0.8):
    pool = nn.MaxPool3d(3,padding = 1,stride =1,ceil_mode = True)
    tmp2 = torch.logical_and(oc<max_occ_thold,oc>1-max_occ_thold)
    tmp3 = pool((oc>max_occ_thold).to(torch.float32).unsqueeze(0)).squeeze(0)
    tmp4 = torch.logical_and(tmp2>0.5,tmp3>0.5)
    oc2= deepcopy(oc)
    oc2[tmp4] = tmp3[tmp4]
    return oc2
