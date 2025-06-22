import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import numpy as np
from joblib import Parallel, delayed
from shelf_gym.scripts.data_generation.map_collection import MapCollection
from shelf_gym.utils.mapping_utils import SweptMapGenerator
from shelf_gym.utils.pushing_utils import PushSampler
import yaml


class PushingCollection(MapCollection):
    def __init__(self,render=False, shared_memory=False, hz=240, all_background = True,
                 debug = False,  save_dir = '../../data/push_data', max_dataset_size = 1000,
                 use_occupancy_for_placing=True, max_obj_num = 25, max_occupancy_threshold=.4, use_ycb=False,
                 show_vis = False, job_id = 0):
        super().__init__(render=render, shared_memory=shared_memory, hz=hz,
                         all_background = all_background, debug=debug, save_dir = save_dir,
                         max_dataset_size = max_dataset_size, max_obj_num=max_obj_num,
                         max_occupancy_threshold=max_occupancy_threshold,
                         use_occupancy_for_placing=use_occupancy_for_placing,
                         use_ycb=use_ycb,show_vis = show_vis,job_id = job_id)

        self.ps = PushSampler()
        self.smg = SweptMapGenerator()


    def collect_data(self):
        self.get_iterations()
        # we first collect the data before the action happens
        gt_data, array_data = self.collect_gt_and_camera_array_data('pre_action')
        height_map_3d = self.remove_walls(gt_data["voxel_height_map"])

        # perform the pushing
        swept_volume, mps = self.ps.perform_single_pushing(self, height_map_3d, execute=True)

        swept_save_dir = self.create_save_dir('swept_volume')

        np.savez_compressed(swept_save_dir+'/swept_map.npz',swept_map = swept_volume, motion_parametrization = mps)

        # and collect the gt positions and camera array data after the push has been made
        self.collect_gt_and_camera_array_data('post_action')


################################
'''main'''
################################

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


def run(parallel,config_data, i):
    environment = PushingCollection(render=not parallel,
                                    shared_memory=config_data["shared_memory"],
                                    hz=config_data["hz"],
                                    use_ycb=config_data["use_ycb"],
                                    debug=config_data["debug"],
                                    show_vis=config_data["show_vis"],
                                    max_dataset_size=config_data["max_dataset_size"],
                                    job_id=i,
                                    save_dir=config_data["save_dir"],
                                    max_obj_num=config_data["max_obj_num"],
                                    max_occupancy_threshold=config_data["max_occupancy_threshold"],
                                    use_occupancy_for_placing=config_data["use_occupancy_for_placing"])
    run_env(environment)


if __name__ == '__main__':
    config_data = parse_config()
    parallel = False
    if(parallel):
        n_jobs = 12
        Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(run)(parallel, config_data, i) for i in range(n_jobs))
    else:
        run(parallel, config_data, 0)