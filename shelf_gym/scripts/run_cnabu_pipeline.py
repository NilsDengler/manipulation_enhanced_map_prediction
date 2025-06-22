from shelf_gym.utils.model_evaluation_utils import EvaluationHelper, get_igs_for_map, get_subsequent_igs_for_map
from shelf_gym.scripts.model_training.train_ycb_map_completion import SemanticMapPredictorYCB
from shelf_gym.scripts.model_training.train_ycb_push_prediction import PushPredictor
from shelf_gym.scripts.data_generation.pushing_collection import PushingCollection
from shelf_gym.utils.uncertainty_informed_push_utils import generate_push_samples
from shelf_gym.utils.learning_utils.datasets import MapDatasetH5py
from shelf_gym.utils.result_visualization_utils import get_my_cmap
from shelf_gym.utils.information_gain_utils import InfoGainEval
from shelf_gym.utils.scaling_utils import scale_semantic_map
from shelf_gym.utils.mapping_utils import SweptMapGenerator
from shelf_gym.utils.pushing_utils import execute_push
from scipy.ndimage import median_filter
from matplotlib import pyplot as plt
import numpy as np
import cupy as cp
import logging
import pickle
import torch
import time
import os
logger = logging.getLogger("trimesh")
logger.setLevel(40)


class ManipulationEnhancedMapping(PushingCollection):
    def __init__(self,render=False, shared_memory=False, hz=240, all_background = True, debug = False,
                 save_dir = '../../data/pipeline_data', max_dataset_size = 1000, use_occupancy_for_placing=True,
                 max_obj_num = 25, max_occupancy_threshold=.4, use_ycb=True, show_vis = False, job_id = 0,
                 use_uncertainty_informed_sampling=False):
        super().__init__(render=render, shared_memory=shared_memory, hz=hz,
                         all_background = all_background, debug=debug, save_dir = save_dir,
                         max_dataset_size = max_dataset_size, max_obj_num=max_obj_num,
                         max_occupancy_threshold=max_occupancy_threshold,
                         use_occupancy_for_placing=use_occupancy_for_placing,
                         use_ycb=use_ycb,show_vis = show_vis,job_id = job_id)

        # set path to models, camera matices and dummy dataset
        map_predictor_dir = os.path.join(os.path.dirname(__file__), "./model/")
        push_predictor_dir = os.path.join(os.path.dirname(__file__), "./model/push_predictor_new.ckpt")
        camera_matrix_dir = os.path.join(os.path.dirname(__file__), "./model/camera_matrices.npz")
        dummy_dataset_dir = os.path.join(os.path.dirname(__file__), './model/dataset.hdf5')

        # init prediction networks
        self.init_cnabu_models(map_predictor_dir, push_predictor_dir, load_from_push_model=True)

        # init evaluation helpers
        self.ig_calc = InfoGainEval(camera_matrix_dir, subsample=8, occupancy_thold=0.95, cached=False)
        self.push_ig_calc = InfoGainEval(camera_matrix_dir, subsample=16, occupancy_thold=0.95, cached=False)
        self.EH = EvaluationHelper(camera_matrix_dir)
        self.dataset = MapDatasetH5py(dummy_dataset_dir, max_samples=10, skip=1, noise=False, move_and_rotate=False,
                                      camera_params_dir=camera_matrix_dir, use_continous_cameras=False)
        self.smg = SweptMapGenerator()

        # hyperparameter of MEM-pipeline
        self.prob_cutoff = 0.85
        self.stopping_criterion = 0.99
        self.action_budget = 40
        self.max_sampled_pushes = 80
        self.n_classes = 15
        self.my_cmap = get_my_cmap(n_classes=self.n_classes)

        # set if random or uncertainty informed push sampling is used
        self.use_uncertainty_informed_sampling = use_uncertainty_informed_sampling
        if self.use_uncertainty_informed_sampling:
            print("Use uncertainty informed sampling strategy")


    def init_cnabu_models(self, map_predictor_dir, push_predictor_dir, load_from_push_model=False):
        """
            Initialize and configure the push prediction and map completion models.

            This method loads the PushPredictor from a checkpoint, then sets up the
            corresponding map completion model either by extracting it from the loaded
            push predictor or by loading a standalone SemanticMapPredictorYCB checkpoint.

            Parameters
            ----------
            map_predictor_dir (str): Path to the directory (or file prefix) containing the map predictor checkpoint
            push_predictor_dir (str): Path to the directory containing the push predictor checkpoint.
            load_from_push_model (bool, optional): If True, extract the map completion model from the loaded PushPredictor
        """

        self.push_prediction_model = PushPredictor.load_from_checkpoint(push_predictor_dir,
                                                                        map_predictor_dir=map_predictor_dir)

        self.map_completion_model = self.push_prediction_model.dp.map_completion_model if load_from_push_model \
            else SemanticMapPredictorYCB.load_from_checkpoint(map_predictor_dir+"model-5dburcae:v4.ckpt").eval()

        self.map_completion_model.predictor.normalize = True
        self.map_completion_model.eval()
        self.push_prediction_model.eval()


    def run(self, predefined_scene_dir=None, use_push=True, debug=True):
        """
        Execute the full MEM perception-push cycle for the shelf rearrangement task.

        Steps:
          1. (Optional) Load and restore a predefined scene arrangement.
          2. Initialize heightmaps and semantic observations.
          3. Build initial occupancy and semantic maps via the map completion model.
          4. Iteratively select viewpoints or push actions based on information gain (IG)
             and predicted push benefits, up to `action_budget` steps or until mapping
             is sufficiently complete.
          5. After each action, store timing metrics, predicted maps, and GT data.

        Parameters
        ----------
        predefined_scene_dir (str, optional) : Path to a pickle file defining an initial shelf arrangement
        use_push (bool): If False, only viewpoint observations will be executed (no pushes).

        Returns
        -------
        output_data (dict): Accumulated results containing occupancy/semantic maps, ground truth,  push commands, timing, and positional differences.
        """
        # Reset GPU cache
        torch.cuda.empty_cache()

        # 1. Load predefined scene if provided
        if predefined_scene_dir:
            with open(predefined_scene_dir, 'rb') as f:
                arrangement = pickle.load(f)
            self.restore_shelf_state(arrangement)

        # Initial sensor observation
        cam_data, gt_data = self.get_processed_array_and_gt_data()
        start_positions, _ = self.obj.update_obj_states(self.current_obj_ids)

        # Convert and mask invalid semantic observations
        height_hms = np.array(cam_data['height_maps'])
        semantic_hms = np.array(cam_data['semantic_maps'])
        invalid_mask = height_hms[..., 0] == 0
        semantic_hms[invalid_mask] = self.n_classes

        # Initialize belief maps
        previous_map, previous_semantic_map = (self.map_completion_model.dp.get_initial_map(torch.ones((1, 1, 204, 120, 200), device='cuda')))
        torch.cuda.empty_cache()

        # Prepare logging and state
        previous_views = []
        pushes = []
        collision = False
        fresh_push = False
        pos_diff_collision = 0
        done_mapping = False

        output_data = {
            'occupancy_map': [],
            'semantic_map': [],
            'occupancy_gt': [],
            'semantic_gt': [],
            'pos_diffs': [],
            'pos_diffs_without_collision': [],
            'pushes': [],
            'push_time': [],
            'vpp_time': [],
            'step_time': []
        }

        # Main loop: plan and execute actions
        for step in range(self.action_budget):
            t_start = time.time()
            push_time = vpp_time = 0.0

            if not collision:
                # Compute next viewpoint IG if no recent push
                if not fresh_push:
                    first_igs, _ = get_igs_for_map(previous_map, self.ig_calc,
                                             skip=1, use_alternative=True)
                    first_igs[previous_views] = 0
                    max_obs_ig = float(first_igs.max())
                    viewpoint = int(first_igs.argmax())
                    vpp_time = time.time() - t_start
                    if debug:
                        print(f"Selected viewpoint={viewpoint} IG={max_obs_ig:.3f}")

                # Decide between push and observe
                can_push = (use_push and (3 <= step < self.action_budget - 1) and not done_mapping)

                if can_push:
                    if not fresh_push:
                        #compute IG for (view, view) horizon
                        second_igs = get_subsequent_igs_for_map(previous_map,[viewpoint], self.ig_calc)
                        second_igs[previous_views] = 0
                        best_observation_ig = second_igs.max() + max_obs_ig

                        # get push candidates
                        t_push_plan = time.time()
                        push_candidates = self.get_possible_maps_push(previous_map, previous_semantic_map,
                                                                      num_points=self.max_sampled_pushes)


                        best_push = 0
                        best_push_ig = 0.0
                        if push_candidates['paths'] is not None:
                            _, best_push, best_push_ig = self.eval_push_igs(push_candidates, previous_semantic_map,
                                                                            use_delta_H=True, skip=5)
                            push_time = time.time() - t_push_plan
                        if debug: print(f"IG obs+1={max_obs_ig:.3f} vs IG push={best_push_ig:.3f}")

                    # Execute push if beneficial
                    if best_push_ig > best_observation_ig and not fresh_push:
                        if debug: print("Performing push action")
                        execute_push(self, push_candidates['paths'][best_push],
                                     path_annotations=push_candidates['path_annotations'][best_push])

                        collision = self.obj.check_all_object_drop(self.current_obj_ids)
                        pushes.append(2)
                        fresh_push = not collision
                        if fresh_push:
                            previous_views.clear()
                            previous_map = (push_candidates['possible_previous_maps'][best_push][None])
                            previous_semantic_map = (push_candidates['possible_semantic_maps'][best_push][None])

                    else:
                        # Otherwise perform an observation
                        if fresh_push:
                            if debug: print("Observing after push")
                            fresh_push = False
                            previous_map, previous_semantic_map = self.execute_observation(previous_views, viewpoint,
                                                                                           previous_map,
                                                                                           previous_semantic_map)
                            pushes.append(0)

                else:
                    # Only observation allowed
                    if debug: print("Observation-only step")
                    previous_map, previous_semantic_map = self.execute_observation( previous_views, viewpoint,
                                                                                    previous_map, previous_semantic_map)
                    pushes.append(0)

                # Check for mapping completion
                sem_conf = self.get_semantic_certainty(previous_semantic_map)
                certainly_mapped_fraction = (self.get_certainly_mapped_fraction(sem_conf, self.prob_cutoff))
                done_mapping = certainly_mapped_fraction >= self.stopping_criterion

                if debug: print(f"Step {step}: mapped {certainly_mapped_fraction:.3f}")

            # Compute positional differences
            end_positions, _ = self.obj.update_obj_states(self.current_obj_ids)
            pos_diff = sum(np.linalg.norm(np.array(s[:2]) - np.array(e[:2]))for s, e in zip(start_positions, end_positions))

            if not collision:
                pos_diff_collision = pos_diff

            # Record timing
            step_time = time.time() - t_start
            if vpp_time == 0 and push_time == 0:
                step_time = 0.0

            # 4.8. Store results
            output_data, gt_data = self.store_results(
                output_data,
                previous_map,
                previous_semantic_map,
                gt_data,
                fresh_push,
                pos_diff,
                pos_diff_collision,
                pushes,
                collision,
                step_time,
                push_time,
                vpp_time,
                store_gt_after_push=True
            )

            # Early termination
            if done_mapping and debug:
                print("Mapping complete; ending run.")
                break

        return output_data

    # def run(self, predefined_scene_dir=None, visualize=True, use_push=True):
    #     torch.cuda.empty_cache()
    #     if predefined_scene_dir is not None:
    #         # we load the new arrangement
    #         new_arrangement = pickle.load(open(predefined_scene_dir, 'rb'))
    #         # set it in the scene
    #         self.restore_shelf_state(new_arrangement)
    #
    #     camera_array_data, gt_data =  self.get_processed_array_and_gt_data()
    #     start_positions, _ = self.obj.update_obj_states(self.current_obj_ids)
    #
    #     hms = np.array(camera_array_data["height_maps"].copy())
    #     semantic_hms =  np.array(camera_array_data["semantic_maps"].copy())
    #
    #     null_obs = hms[:, :, :, 0] == 0
    #     semantic_hms[null_obs] = self.n_classes
    #
    #
    #     total_free = torch.from_numpy(np.zeros((102, 120, 200))).to('cuda')
    #     previous_map, previous_semantic_map = self.map_completion_model.dp.get_initial_map(torch.ones((1, 1, 204, 120, 200)).to('cuda'))
    #     torch.cuda.empty_cache()
    #
    #     previous_views = []
    #     pushes = []
    #     done_mapping = False
    #     fresh_push = False
    #     certainly_mapped_fraction = 0
    #     output_data = {'occupancy_map': [], 'semantic_map': [], 'occupancy_gt': [], 'semantic_gt': [], 'pos_diffs' : [],'pos_diffs_without_collision' : [], 'pushes': [], 'push_time': [], 'vpp_time': [], 'step_time': []}
    #     collision = False
    #     pos_dif_collision = 0
    #     for i in range(self.action_budget):
    #         push_time = 0
    #         vpp_time = 0
    #         step_time = 0
    #         t_0 = time.time()
    #         if not collision:
    #             t_1 = time.time()
    #             if (not fresh_push):
    #                 t0 = time.time()
    #                 obs_igs, _ = get_igs_for_map(previous_map,
    #                                           self.ig_calc,
    #                                           skip=1,
    #                                           use_alternative=True)
    #                 obs_igs[previous_views] = 0
    #                 max_obs_ig = obs_igs.max()
    #                 viewpoint = obs_igs.argmax()
    #                 vpp_timing = time.time() - t0
    #                 print('calculating argmax - viewpoint = {}'.format(viewpoint))
    #
    #             if ((i >= 3) and (i < self.action_budget - 1) and (not done_mapping)) and use_push:
    #                 if (not fresh_push):
    #                     second_obs_igs = get_subsequent_igs_for_map(previous_map,
    #                                                                 [viewpoint],
    #                                                                 self.ig_calc)
    #
    #                     second_obs_igs[previous_views] = 0
    #                     final_obs_ig = second_obs_igs.max() + max_obs_ig
    #                     vpp_time = time.time() - t_1
    #                     t_2 = time.time()
    #                     push_tmps = self.get_possible_maps_push(previous_map, previous_semantic_map,
    #                                                             num_points=self.max_sampled_pushes,
    #                                                             vis_debug=False)
    #
    #                     if (push_tmps['paths'] is None):
    #                         max_push_ig = 0
    #                     else:
    #                         push_view, max_push, max_push_ig = self.eval_push_igs(push_tmps, previous_semantic_map, use_delta_H=True, skip=5)
    #
    #                     push_time = (time.time() - t_2)
    #                     print('IGS: OBS_t + OBS_t+1 = {} | PUSH+OBS = {} | Mapped = {} | Step = {}'.format(final_obs_ig, max_push_ig,
    #                                                                                          certainly_mapped_fraction, i))
    #                 # perform push if it is better than the current observation
    #                 if ((max_push_ig > 1.0 * final_obs_ig) and (not fresh_push)):
    #                     print('\n\n Perform Push!!\n\n')
    #                     #actually perform the push
    #                     execute_push(self,
    #                                  push_tmps['paths'][max_push],
    #                                  path_annotations=push_tmps['path_annotations'][max_push])
    #
    #                     collision = self.obj.check_all_object_drop(self.current_obj_ids)
    #                     fake_collision = False #self.obj.check_all_object_drop(self.current_obj_ids)
    #                     if collision:
    #                         pushes.append(2)
    #
    #                         print("COLLISION OCCURED; WILL END EPISODE")
    #                     elif fake_collision:
    #                         pushes.append(2)
    #
    #                         fresh_push = True
    #                         # we clear previous views counter
    #                         previous_views = []
    #                         previous_map = push_tmps['possible_previous_maps'][max_push][torch.newaxis]
    #                         previous_semantic_map = push_tmps['possible_semantic_maps'][max_push][torch.newaxis]
    #                     else:
    #                         pushes.append(1)
    #
    #                         fresh_push = True
    #                         # we clear previous views counter
    #                         previous_views = []
    #                         previous_map = push_tmps['possible_previous_maps'][max_push][torch.newaxis]
    #                         previous_semantic_map = push_tmps['possible_semantic_maps'][max_push][torch.newaxis]
    #
    #
    #                 else:
    #                     # if we pushed in last step, perform observation now
    #                     if (fresh_push):
    #                         fresh_push = False
    #                         viewpoint = push_view
    #                         print('executing viewpoint after best push')
    #                     previous_map, previous_semantic_map = self.execute_observation(previous_views,
    #                                                                                    viewpoint,
    #                                                                                    previous_map,
    #                                                                                    previous_semantic_map)
    #                     pushes.append(0)
    #
    #             else:
    #                 print('executing viewpoint after best push')
    #
    #                 # if were at the first or last step, only consider observations
    #                 previous_map, previous_semantic_map = self.execute_observation(previous_views,
    #                                                                                viewpoint,
    #                                                                                previous_map,
    #                                                                                previous_semantic_map)
    #                 pushes.append(0)
    #
    #             # check termination criteria
    #             sem_conf = self.get_semantic_certainty(previous_semantic_map)
    #             certainly_mapped_fraction = self.get_certainly_mapped_fraction(sem_conf, self.prob_cutoff)
    #             done_mapping = True if (certainly_mapped_fraction >= self.stopping_criterion) else False
    #
    #         print('Mapped = {} | Step = {}'.format(certainly_mapped_fraction, i))
    #
    #         if done_mapping:
    #             print("DONE MAPPING; NO PUSHING ANYMORE")
    #
    #         end_positions , _  = self.obj.update_obj_states(self.current_obj_ids)
    #         pos_dif = 0
    #
    #         for start, end in zip(start_positions, end_positions):
    #             pos_dif += np.linalg.norm(np.array(start[:2]) - np.array(end[:2]))
    #         # get data to return
    #         if not collision:
    #             pos_dif_collision = pos_dif
    #         step_time = time.time() - t_0
    #         if vpp_time == 0 and push_time == 0:
    #             step_time = 0
    #
    #         output_data, gt_data = self.store_results(output_data, previous_map, previous_semantic_map, gt_data, fresh_push, pos_dif, pos_dif_collision,pushes, collision, step_time,push_time,vpp_time, store_gt_after_push=True)
    #
    #     return output_data


    def get_processed_array_and_gt_data(self, only_array=False, only_gt=False):
        """
        Retrieve and preprocess camera array heightmaps and ground-truth heightmap data.

        Parameters
        ----------
        only_array (bool, optional): If True, return only the processed camera array data
        only_gt (bool, optional): If True, return only the processed ground-truth data

        Returns
        -------
        A tuple `(camera_array_data, processed_gt)`
        """

        camera_array_data = {}
        if not only_gt:
            camera_array_data = self.get_camera_array_heightmaps(no_tqdm=True)
            if only_array:
                return camera_array_data, {}

        # Fetch ground-truth heightmap and semantic labels
        gt_data = self.get_gt_height_map(no_tqdm=True)

        # Process voxel heightmap: threshold, cast, and crop
        gt_voxel_heightmap = gt_data["voxel_height_map"].copy()
        gt_voxel_heightmap = (gt_voxel_heightmap > 0.5).astype(int)
        gt_voxel_heightmap = gt_voxel_heightmap[35:119, 21:-21, :59]

        # Crop semantic ground-truth map to the same XY region
        semantic_gt = gt_data["semantic_gt"].copy()[35:119, 21:-21]

        processed_gt = {"semantic_gt": semantic_gt, "voxel_height_map": gt_voxel_heightmap}
        return camera_array_data, processed_gt


    def execute_observation(self, prior_viewpoint_list, viewpoint, previous_map, previous_semantic_map):
        """
        Execute a single observation step: update viewpoint history, obtain sensor data,
        perform map completion, and integrate semantic labels.

        Parameters
        ----------
        prior_viewpoint_list (list): A list of previously visited viewpoints
        viewpoint (dict): The current viewpoint matrix dict
        previous_map (np.ndarray): The occupancy map from the previous timestep (3D array).
        previous_semantic_map (np.ndarray): The semantic label map from the previous timestep (3D array).

        Returns
        -------
        next_map (np.ndarray): The updated occupancy map after map completion.
        next_semantic_map (np.ndarray): The updated semantic label map after map completion.
        """

        prior_viewpoint_list.append(viewpoint)
        viewpoint_data = self.get_single_camera_array_heightmaps(viewpoint)
        outputs, new_free, this_semantic_hm, this_hm = self.EH.get_direct_map_completion(viewpoint, self.dataset,
                                                                                         viewpoint_data["height_maps"][0],
                                                                                         viewpoint_data["depth_maps"][0],
                                                                                         viewpoint_data["semantic_maps"][0],
                                                                                         self.map_completion_model,
                                                                                         previous_map,
                                                                                         previous_semantic_map)

        this_semantic_hm[this_hm == 0] = self.n_classes - 1
        next_map = outputs['occupancy_map']
        next_semantic_map = outputs['semantic_map']
        return next_map, next_semantic_map


    def create_batched_push_input(self, beta_map, dirichlet_map, occupancy_beta, psm, svs,
                                  motion_parametrization, batch_size=15):
        """
        Build batched inputs for the push prediction model by injecting push features
        into occupancy and semantic maps in chunks

        Parameters
        ----------
        beta_map (torch.Tensor):  beta map of beta distribution used as a static reference for feature injection
        dirichlet_map (torch.Tensor): Base semantic (Dirichlet) map used as a static reference for feature injection
        occupancy_beta (torch.Tensor): beta distribution
        psm (torch.Tensor): previous semantic map
        svs (torch.Tensor): swept volumes
        motion_parametrization (torch.Tensor): Per-sample motion parameters for each push action
        batch_size (int, optional): Number of samples per sub-batch for processing (default: 15).

        Returns
        -------
        batches_map_and_push, batches_semantic_map
        """
        batched_occupancy_beta = torch.split(occupancy_beta, batch_size)
        batched_psm = torch.split(psm, batch_size)
        batched_svs = torch.split(svs, batch_size)
        batched_motion_parametrization = torch.split(motion_parametrization, batch_size)
        map_and_push = torch.empty(0, 308, 120, 200).to('cpu')
        semantic_map = torch.empty(0, 15, 120, 200).to('cpu')

        for i, b in enumerate(batched_occupancy_beta):
            with torch.no_grad():
                tmp = self.push_prediction_model.dp.add_push_features(batched_occupancy_beta[i].to('cuda'),
                                                                      batched_psm[i].to('cuda'),
                                                                      batched_svs[i].to('cuda'),
                                                                      batched_motion_parametrization[i].to('cuda'),
                                                                      all_steps=False,
                                                                      outputs={'occupancy_map': beta_map,
                                                                               'semantic_map': dirichlet_map})
                map_and_push = torch.cat([map_and_push, tmp['map_and_push'].cpu()], dim=0)
                semantic_map = torch.cat([semantic_map, tmp['semantic_map'].cpu()], dim=0)

        batches_map_and_push = torch.split(map_and_push, batch_size)
        batches_semantic_map = torch.split(semantic_map, batch_size)
        return batches_map_and_push, batches_semantic_map



    def occupancy_mean_variance(self, beta_dist):
        """
       computes mean, variance (epistemic uncertainty), and aleatoric of the beta distribution

        Parameters
        ----------
        beta_dist (torch.Tensor):  beta distribution

        Returns
        -------
        mean, epistemic, aleatoric
        """
        alpha = beta_dist[0, 1::2, :, :]
        beta = beta_dist[0, ::2, :, :]
        mean = alpha / (alpha + beta)
        epistemic = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        aleatoric = (alpha * beta) / ((alpha + beta) * (alpha + beta + 1))
        return mean, epistemic, aleatoric


    def prepare_uncertainty_informed_push_sampling(self, beta_map, dirichlet_map, use_no_uncertainty=True, use_bin_uncertainty=False, use_semantic_distance=False, use_occ_distance=True, num_points=30):
        # get mean, as well as epistemic and aleatoric uncertainty for occupancy map
        mean, epi, alea = self.occupancy_mean_variance(beta_map)
        mean = mean.permute(1, 2, 0).cpu().numpy()[:, :, 10:].max( axis=2)
        epi = epi.permute(1, 2, 0).cpu().numpy()[:, :, 10:].mean(axis=2)
        alea = alea.permute(1, 2, 0).cpu().numpy()[:, :, 10:].mean(axis=2)
        uncertainty = mean + np.sqrt(epi)

        if use_bin_uncertainty:
            uncertainty = (uncertainty >= 0.2).astype(np.uint8) * 10
        if use_no_uncertainty:
            uncertainty = self.occupancy_mean_variance(beta_map)[0].permute(1, 2, 0).cpu().numpy()[:, :, 10:].max( axis=2)  # [C, H, W] → [H, W, C]

        if use_semantic_distance:
            # get semantic confidence and uncertainty
            label = self.get_semantic_rgb_image(dirichlet_map)[2][::-1, ::-1]
            sem_conf = self.get_semantic_rgb_image(dirichlet_map)[0][::-1, ::-1]
            semantic_total_uncertainty = self.n_classes / dirichlet_map.sum(axis=1, keepdims=True)[0, 0, :, :].cpu().numpy()

            distance_foundation = semantic_total_uncertainty[::-1, ::-1][15:90, 21:-21]
            distance_foundation = np.where((distance_foundation >= 0.1), distance_foundation, 0)
        elif use_occ_distance:
            total_uncertainty = epi + alea
            distance_foundation = np.where((total_uncertainty >= 0.01), total_uncertainty, 0)[::-1, ::-1][15:90, 21:-21]
        else:
            occupacy_mask = self.occupancy_mean_variance(beta_map)[0].permute(1, 2, 0).cpu().numpy()[::-1, ::-1, 10:].max(axis=2)[15:90, 21:-21]
            distance_foundation = np.where((occupacy_mask >= 0.2) & (occupacy_mask <= 0.85), occupacy_mask, 0)



        samples = generate_push_samples(distance_foundation, uncertainty, label, sem_conf,
                                        num_samples=5, uncertain_distance=use_occ_distance or use_semantic_distance,
                                        debug=False)

        if len(samples) == 0:
            return {
                'paths': None,
                'path_annotations': None,
                'motion_parametrization': None,
                'possible_semantic_maps': None,
                'possible_previous_maps': None,
                'swept_volumes': None
            }

        j = 0
        push_data = {"paths": None}
        prob_padded = np.pad(self.get_prob_map(beta_map).cpu().numpy(), ((10, 10), (0, 0), (0, 0)), mode='constant')
        while push_data['paths'] == None and j < len(samples):
            push_data = self.ps.get_samples(env=self, samples=samples[j], occ_map=prob_padded, num_points=num_points, just_endpoints=True)
            j = j + 1

        return push_data


    def get_possible_maps_push(self, beta_map, dirichlet_map, num_points=30):
        """ Computes possible maps after a push action.
        AParameters
        ----------
            beta_map (torch.Tensor): Previous beta map.
            dirichlet_map (torch.Tensor): Previous semantic dirichlet map.
            vis_debug (bool, optional): If True, visualizes the results. Defaults to False.
            num_points (int, optional): Number of push points to sample. Defaults to 30.
        Returns:
            dict: Dictionary containing motion parameters, semantic maps, occupancy maps, etc.
        """
        # Compute probability map
        prob3d = self.get_prob_map(beta_map).cpu().numpy()  # (H, W, D)
        occ2d = prob3d[:, :, 10:].max(axis=-1)

        # Get push samples
        if self.use_uncertainty_informed_sampling:
            push_data = self.prepare_uncertainty_informed_push_sampling(beta_map, dirichlet_map, use_semantic_distance=True, num_points=num_points)
        else:
            prob_padded = np.pad(self.get_prob_map(beta_map).cpu().numpy(), ((10, 10), (0, 0), (0, 0)), mode='constant')
            push_data = self.ps.get_samples(env=self, occ_map=prob_padded, num_points=num_points, just_endpoints=True, verbose=False)

        #Convert motion parameters to numpy
        motion_parametrization = np.array([int(x) if isinstance(x, cp.ndarray) else x for x in push_data['motion_parametrization']]).reshape(-1, 6)

        if motion_parametrization.shape[0] == 0:
            return {
                'paths': None,
                'path_annotations': None,
                'motion_parametrization': None,
                'possible_semantic_maps': None,
                'possible_previous_maps': None,
                'swept_volumes': None
            }
        else:
            motion_parametrization = torch.tensor(motion_parametrization, dtype=torch.float32, device='cuda')

        # Compute swept volumes for each push
        swepts = []
        for path, annot in zip(push_data['paths'], push_data['path_annotations']):
            traj = self.linear_interpolate_motion_klampt_joint_traj( path, traj_annotation=annot, imagined=True, verbose=False)
            sv = self.smg.get_swept_map(traj[::2])[10:-10]
            swepts.append(np.moveaxis(sv, (2, 0, 1), (0, 1, 2)))
        swepts = torch.from_numpy(np.stack(swepts)).to('cuda')

        # Compute alpha and beta occupancy maps
        alpha = beta_map[0, 1::2, :, :]
        beta = beta_map[0, ::2, :, :]
        occupancy_beta = torch.concatenate((beta[:, :, :, torch.newaxis], alpha[:, :, :, torch.newaxis]), axis=-1)
        occupancy_beta = occupancy_beta.expand(swepts.shape[0], -1, -1, -1, -1)

        # Expand previous semantic map for processing
        psm = dirichlet_map.expand(swepts.shape[0], -1, -1, -1)

        batches_map_and_push, batches_semantic_map = self.create_batched_push_input(beta_map,
                                                                                    dirichlet_map,
                                                                                    occupancy_beta,
                                                                                    psm,
                                                                                    swepts,
                                                                                    motion_parametrization,
                                                                                    batch_size=15)

        preds_occ = torch.empty(0, 2, 102, 120, 200).to('cpu')
        preds_sem = torch.empty(0, 15, 120, 200).to('cpu')
        diffs = torch.empty(0, 2, 120, 200).to('cpu')
        prev_maps = torch.empty(0, 204, 120, 200).to('cpu')

        for i, b in enumerate(batches_map_and_push):
            with torch.no_grad():
                batches_possible_occupancy_maps, batches_possible_semantic_maps, batches_pred_difference, batches_possible_previous_maps = self.push_prediction_model.push_predictor(
                    batches_map_and_push[i].to('cuda'), batches_semantic_map[i].to('cuda'))
                preds_occ = torch.cat([preds_occ, batches_possible_occupancy_maps.cpu()], dim=0)
                preds_sem = torch.cat([preds_sem, batches_possible_semantic_maps.cpu()], dim=0)
                diffs = torch.cat([diffs, batches_pred_difference.cpu()], dim=0)
                prev_maps = torch.cat([prev_maps, batches_possible_previous_maps.cpu()],dim=0)

        # Compute difference-based probabilities
        diff_probs = (diffs[:, 1] / (diffs[:, 0] + diffs[:, 1])).unsqueeze(1)
        prev_maps = beta_map.cpu() * (1 - diff_probs) + prev_maps * diff_probs
        preds_sem = dirichlet_map.cpu() * (1 - diff_probs) + preds_sem * diff_probs

        return {
            'paths': push_data['paths'],
            'path_annotations': push_data['path_annotations'],
            'motion_parametrization': motion_parametrization,
            'possible_semantic_maps': preds_sem,
            'possible_previous_maps': prev_maps,
            'swept_volumes': swepts.cpu().numpy()
        }


    def eval_push_igs_rl(self, push_tmps, previous_semantic_map, use_delta_H=False, skip=1):
        """
        Evaluates push IGs and selects the best viewvor continous reinforcemnt learnin agent.
        Args:
            push_tmps (dict): Contains 'possible_previous_maps' and 'possible_semantic_maps'.
            previous_semantic_map (torch.Tensor): The previous semantic map.
            total_free (torch.Tensor): Free space map.
            use_delta_H (bool, optional): Whether to use delta entropy. Defaults to False.
            skip (int, optional): Skipping parameter for IG calculation. Defaults to 1.
        Returns:
            tuple: (best view index, best push index, max information gain)
        """

        # Extract possible occupancy and semantic maps
        possible_occupancies = push_tmps['possible_previous_maps']
        possible_semantic_maps = push_tmps['possible_semantic_maps']

        # Compute semantic entropies
        possible_semantic_Hs = self.get_semantic_map_entropy(possible_semantic_maps)
        original_entropy = self.get_semantic_map_entropy(previous_semantic_map)

        all_igs = []
        cameras = []
        actions = []
        delta_Hs = [] if use_delta_H else None
        for i in range(possible_occupancies.shape[0]):
            action, _states = self.rl_model.predict(self.current_observation, deterministic=False)
            camera = self.minimal_step(action)
            cameras.append(camera)
            actions.append(action)
            tmp_igs, _ = get_igs_for_map(possible_occupancies[i].unsqueeze(0),
                                         self.ig_calc,
                                         skip=1,
                                         camera_matrix=camera,
                                         use_alternative=True)

            if use_delta_H:
                H1 = possible_semantic_Hs[i].sum()
                H0 = original_entropy[0].sum()
                deltaH = (H0 - H1) / 5  # Normalizing deltaH to the same scale

                # Ensure deltaH is non-positive
                deltaH = min(deltaH.item(), 0)

                all_igs.append(tmp_igs)
                delta_Hs.append(deltaH)
            else:
                all_igs.append(tmp_igs)

        # Find max IG and corresponding push index
        all_igs = np.array(all_igs)
        max_ig = all_igs.max()
        max_push = all_igs.argmax()
        action = actions[max_push]
        camera = cameras[max_push]

        return action, camera, max_push, max_ig + delta_Hs[max_push]


    def eval_push_igs(self, push_tmps, previous_semantic_map, use_delta_H=False, skip=1):
        """
        Evaluates push IGs and selects the best view.
        Args:
            push_tmps (dict): Contains 'possible_previous_maps' and 'possible_semantic_maps'.
            previous_semantic_map (torch.Tensor): The previous semantic map.
            total_free (torch.Tensor): Free space map.
            use_delta_H (bool, optional): Whether to use delta entropy. Defaults to False.
            skip (int, optional): Skipping parameter for IG calculation. Defaults to 1.
        Returns:
            tuple: (best view index, best push index, max information gain)
        """

        # Extract possible occupancy and semantic maps
        possible_occupancies = push_tmps['possible_previous_maps']
        possible_semantic_maps = push_tmps['possible_semantic_maps']

        # Compute semantic entropies
        possible_semantic_Hs = self.get_semantic_map_entropy(possible_semantic_maps)
        original_entropy = self.get_semantic_map_entropy(previous_semantic_map)

        all_igs = []
        delta_Hs = [] if use_delta_H else None
        for i in range(possible_occupancies.shape[0]):
            tmp_igs, _ = get_igs_for_map(possible_occupancies[i].unsqueeze(0),self.push_ig_calc,
                                         skip=skip, use_alternative=True)
            if use_delta_H:
                H1 = possible_semantic_Hs[i].sum()
                H0 = original_entropy[0].sum()
                deltaH = (H0 - H1) / 5  # Normalizing deltaH to the same scale

                # Ensure deltaH is non-positive
                deltaH = min(deltaH.item(), 0)

                all_igs.append(tmp_igs)
                delta_Hs.append(deltaH)
            else:
                all_igs.append(tmp_igs)

        # Find max IG and corresponding push index
        all_igs = np.array(all_igs)
        max_igs = all_igs.max(axis=1)
        max_push = max_igs.argmax()

        # Compute IG for the best push on the entire camera array
        best_igs, _ = get_igs_for_map(possible_occupancies[max_push].unsqueeze(0),
                                      self.ig_calc, skip=1, use_alternative=True)

        if use_delta_H:
            print(delta_Hs[max_push])
            print(best_igs.max())

            best_igs += delta_Hs[max_push]

        # Select the best view based on max IG
        view = best_igs.argmax()
        max_ig = best_igs.max()
        return view, max_push, max_ig


    def store_results(self, data, predicted_map, semantic_map, gt_data, fresh_push,pos_diff, pos_diff_collision,pushes, collision,step_time, push_time,vpp_time, store_gt_after_push=False):
        """
        Store experiment metrics, predicted maps, and ground-truth data after a push.

        Parameters
        ----------
        data (dict): Accumulator dictionary to return with keys like "occupancy_map", "semantic_map", etc.
        predicted_map (torch.Tensor): beta-parameterized occupancy output from the model
        semantic_logits  (torch.Tensor): dirichlet semantic output
        gt_data (dict): Ground-truth maps with keys "voxel_height_map" and "semantic_gt"
        fresh_push (bool): True if the last push was executed this step.
        pos_diff (float): Position difference metric including collisions.
        pos_diff_collision (float): Position difference metric excluding collisions.
        pushes (list): push commands executed.
        collision (bool): True if a collision occurred during the push.
        step_time (float): Total time for the step (seconds).
        push_time (float): Time spent planning/executing the push (seconds).
        vpp_time (float):  Time spent in viewpoint planning (seconds).
        store_gt_after_push (bool, optional):  If True and `fresh_push` is True without collision, re-fetch GT after the push.

        Returns
        -------
        data (dict): The updated `data` dictionary with appended results.
        gt_data (dict): The final ground-truth data used for this step.
        """

        occupancy_map = self.get_prob_map(predicted_map).cpu().numpy()
        semantic_map = self.to_channels_last((semantic_map / semantic_map.sum(axis=1, keepdims=True))[0]).cpu().numpy()
        semantic_map = semantic_map[25:109, 21:-21]
        semantic_map = scale_semantic_map(semantic_map)
        occupancy_map = occupancy_map[25:109, 21:-21, :59]
        data['pushes'] = pushes
        data["occupancy_map"].append(occupancy_map)
        data["semantic_map"].append(semantic_map)
        data["pos_diffs"].append(pos_diff)
        data["pos_diffs_without_collision"].append(pos_diff_collision)
        data["step_time"].append(step_time)
        data["push_time"].append(push_time)
        data["vpp_time"].append(vpp_time)
        if fresh_push and store_gt_after_push and not collision:
            _, gt_data = self.get_processed_array_and_gt_data(only_gt=True)
            gt_voxel_heightmap = gt_data["voxel_height_map"].copy()
            semantic_gt = gt_data["semantic_gt"].copy()
        else:
            gt_voxel_heightmap = gt_data["voxel_height_map"]
            semantic_gt = gt_data["semantic_gt"]

        data["occupancy_gt"].append(gt_voxel_heightmap)
        data["semantic_gt"].append(semantic_gt)
        torch.cuda.empty_cache()
        return data, gt_data


    def get_certainly_mapped_fraction(self, sem_conf, prob_cutoff):
        """ Computes the fraction of pixels that are confidently mapped.
        Args:
            sem_conf (numpy.ndarray or torch.Tensor): Semantic confidence map.
            prob_cutoff (float): Probability threshold.
        Returns:  float: Fraction of certainly mapped pixels.
        """
        # Apply median filtering (more efficient than OpenCV for grayscale images)
        sem_conf_filtered = median_filter(sem_conf, size=5)
        # return fraction of pixels above the cutoff
        return (sem_conf_filtered > prob_cutoff).sum() / sem_conf.flatten().shape[0]


    def get_semantic_certainty(self, semantic_map):
        sem_probs = (semantic_map / semantic_map.sum(axis=1, keepdims=True))  # .cpu().numpy()
        sem_probs = scale_semantic_map(sem_probs.cpu().numpy(), axis=1)
        return sem_probs[0].max(axis=0)


    def get_prob_map(self, out_map):
        """
        Computes probability maps from a tensor.
        Args:
            out_map (torch.Tensor): Input tensor of shape (B, C, H, W).
            eps (float): Small constant for numerical stability.
        Returns:
            torch.Tensor: Probability map of shape (B, H, W, C/2).
        """
        alpha = out_map[:, ::2, :, :]
        beta = out_map[:, 1::2, :, :]
        probs = self.to_channels_last((beta / (alpha + beta))[0], 0)

        return probs


    def get_semantic_rgb_image(self, semantics):

        sem_probs = (semantics / semantics.sum(axis=1, keepdims=True))  # .cpu().numpy()
        sem_probs = scale_semantic_map(sem_probs.cpu().numpy(), axis=1)
        sem_conf = sem_probs[0].max(axis=0)
        sem_color = (self.my_cmap[semantics.squeeze().cpu().numpy().argmax(axis=0)] * sem_conf[:, :, np.newaxis] + (
                    1 - sem_conf[:, :, np.newaxis])) * 255
        #np.save("unknown", ( 1 - sem_conf[:, :, np.newaxis]) * 255)
        #pdb.set_trace()

        sem_labels = semantics.squeeze().cpu().numpy().argmax(axis=0)
        mask = sem_conf >= 0.2
        sem_labels_filtered = np.where(mask, sem_labels, 15)

        return sem_color[::-1, ::-1].astype(np.uint8), ( 1 - sem_conf[:, :, np.newaxis]), sem_labels_filtered, sem_conf


    def get_semantic_map_entropy(self, sem_map, eps=1e-8):
        """
        Computes the entropy of a semantic probability map.
        Args:
            sem_map (torch.Tensor): Semantic probability tensor of shape (B, C, H, W).
            eps (float): Small constant for numerical stability.
        Returns: torch.Tensor: Entropy map of shape (B, H, W).
        """
        # Normalize along the channel axis while avoiding division by zero and Crop the relevant region
        semantic_probs = (sem_map / (sem_map.sum(axis=1, keepdim=True) + eps))[:, :, 35:119, 21:-21]
        log_probs = torch.log(semantic_probs + eps)
        # return entropy
        return -(semantic_probs * log_probs).sum(axis=1)


    def to_channels_last(self, tmp, channel=0):
        '''
        helper function, move the channel axis to the last axis
        '''
        tmp = torch.movedim(tmp, channel, -1)
        return tmp




if __name__ == '__main__':
    mem = ManipulationEnhancedMapping(render=True, show_vis=False, use_uncertainty_informed_sampling=False)
    mem.reset_env()
    for i in range(5,25):
        scene_dir = './data/Hard_scenes/scenes/scene_data_'+ str(i) +'.p'
        mem.run(predefined_scene_dir=scene_dir)
    mem.close()
