import numpy as np
import scipy
import torch
from glob import glob 
import cv2 
from shelf_gym.utils.mapping_utils import BEVMapping as Mapping
from matplotlib import pyplot as plt
import pdb
from torch.utils.data import Dataset
import h5py
import cupy as cp
import hdf5plugin
import os 
from torchvision.transforms import v2
from shelf_gym.utils.mapping_utils import freeSpaceCalculator
import numpy 
os.environ["OPENMP_NUM_THREADS"] = '16'
os.environ["BLOSC_NTHREADS"] = '16'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MapDataset(Dataset):
    def __init__(self,dataset_location,max_samples = 10):
        self.dataset_location = dataset_location
        self.hm_files = sorted(glob(self.dataset_location + '/**/hms.npz',recursive = True))
        self.gt_hm_files = sorted(glob(self.dataset_location + '/**/gt_hms.npz',recursive = True))
        self.max_samples = max_samples
        self.rng = np.random.default_rng()
        self.mapper = Mapping(height = 120)
        self.height_bins = cp.asnumpy(self.mapper.height_bins.copy())
        self.height_grid = cp.asnumpy(self.mapper.height_grid.numpy().copy())
        self.height_resolution = cp.asnumpy(self.mapper.height_resolution)
    def __len__(self):
        return len(self.hm_files)-1

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        tmp = np.load(self.hm_files[idx])
        tmp2 = np.load(self.gt_hm_files[idx])
        hms = tmp['hms']
        semantic_hms = tmp['semantic_hms']

        del tmp
        gt_3d = np.moveaxis(tmp2['hm3d'],[2,0,1],[0,1,2])
        gt_3d[gt_3d > 0.5] = 1.0
        gt_3d[gt_3d <= 0.5] = 0
        gt_2d = tmp2['gt_hms']
        gt_semantics_3d = tmp2['semantic_3d']
        gt_semantics = tmp2['semantic_2d']
        del tmp2
        # pdb.set_trace()
        selected_indices = self.rng.choice(np.arange(0,hms.shape[0]),self.max_samples,replace = False)
        sampled_hms = hms[selected_indices]
        sampled_semantics = semantic_hms[selected_indices]


        occupied,free = self.local_find_update_cells(sampled_hms)
        # occupied = []
        # free = []
        # for hm in sampled_hms:
        #     # _,prob_map = mapper.mapping(hm,log_map)
        #     # mapper.reset_mapping()
        #     occupied_map,free_map = self.mapper.find_update_cells(torch.from_numpy(hm))
        #     occupied.append(occupied_map.permute(2,0,1).numpy())
        #     free.append(free_map.permute(2,0,1).numpy())

        # free = np.array(free)
        # occupied = np.array(occupied)


        return {'gt_3d':gt_3d,'free':free,'occupied':occupied,'gt_2d':gt_2d,'hms':sampled_hms,'semantic_hms':sampled_semantics,'gt_semantics':gt_semantics,'gt_semantics_3d':gt_semantics_3d}

    def local_find_update_cells(self,hms):
        # pdb.set_trace()
        cp_hms = np.array(hms)
        clipped_hm = np.round(np.clip(cp_hms[:,:,:,0]/self.height_resolution, 0, self.height_bins)).astype(np.uint8)
        unchanged = cp_hms[:,:,:,0] ==0
        occupied = self.height_grid[np.newaxis,:,:,:] < clipped_hm[:,:,:,np.newaxis]
        free = np.logical_not(occupied)
        occupied[unchanged,:] = False
        free[unchanged,:] = False
        free[cp_hms[:,:,:,1].astype(bool)] = False 
        
        occupied = np.moveaxis(occupied,[1,2,3],[2,3,1])
        free = np.moveaxis(free,[1,2,3],[2,3,1])
        return occupied,free

class MapDatasetH5py(Dataset):
    def __init__(self,dataset_location,max_samples = 10,skip = 1,n_classes = 15,noise = True,move_and_rotate = True,camera_params_dir = '../../camera_matrices.npz', is_real_world_data=False, use_continous_cameras=False ):
        self.use_continous_cameras = use_continous_cameras

        self.dataset_location = dataset_location
        self.skip = skip
        self.f = h5py.File(self.dataset_location,'r',locking = False,swmr = True)

        if self.use_continous_cameras:
            dataset_keys = list(self.f["cameras"].keys())  # e.g., ["position", "intrinsic_matrix", ...]
            camera_dsets = {key: self.f["cameras"][key] for key in dataset_keys}
            num_samples, num_objects = camera_dsets[dataset_keys[0]].shape[:2]
            self.all_cameras = [
                [
                    {
                        key: camera_dsets[key][i, j]
                        for key in dataset_keys
                    }
                    for j in range(num_objects)
                ]
                for i in range(num_samples)
            ]

        self.skip = skip
        self.all_hms = self.f['hms']
        self.size = self.all_hms.shape[0]
        self.n_classes = n_classes
        self.noise = noise
        self.affine = v2.functional.affine
        # self.all_semantic_hms = self.f['semantic_hms']
        # self.all_gt_3d = self.f['gt_3d']
        # self.all_gt_2d = self.f['gt_2d']
        # self.all_gt_semantics_3d = self.f['gt_semantics_3d']
        # self.all_gt_semantics = self.f['gt_semantics']
        self.f.close()
        del self.f
        del self.all_hms
        if self.use_continous_cameras:
            del self.all_cameras
        self.max_samples = max_samples
        self.rng = np.random.default_rng()

        dummy_data = {"height_map": np.zeros((120, 200))}
        self.mapper = Mapping(dummy_data, raw_hm_start=True)
        self.height_bins = cp.asnumpy(self.mapper.height_bins.copy())   
        self.height_grid = cp.asnumpy(self.mapper.torch_height_grid.cpu().numpy().copy())
        self.height_resolution = cp.asnumpy(self.mapper.height_resolution)
        self.move_and_rotate = move_and_rotate
        self.camera_params_dir = camera_params_dir
        self.extrinsics = []
        self.intrinsics = []
        if is_real_world_data:
            camera_matrices = np.load(self.camera_params_dir,allow_pickle = True)['matrices']
            for camera in range(len(camera_matrices)):
                self.extrinsics.append(
                    np.linalg.inv(camera_matrices[camera]['corrected_matrix'].reshape(4, 4, order='C')))
                self.intrinsics.append(np.array(
                    [910.6857299804688, 0.0, 642.8099975585938, 0.0, 910.9524536132812, 382.1358337402344, 0.0, 0.0,
                     1.0]).reshape(3, 3, order='C'))
        else:
            camera_matrices = np.load(self.camera_params_dir,allow_pickle = True)['obj_ids']
            for camera in range(len(camera_matrices)):
                self.extrinsics.append(np.linalg.inv(camera_matrices[camera]['o3d_extrinsic_matrix'].reshape(4, 4, order='C')))
                self.intrinsics.append(camera_matrices[camera]['intrinsic_matrix'].reshape(3, 3, order='C'))

    def __len__(self):
        return self.size//self.skip-1

    def reset_h5py(self):
        try:
            self.f.close()
        except Exception as e:
            pass
        self.f = h5py.File(self.dataset_location,'r',locking = False,swmr = True)
        self.all_hms = self.f['hms']
        self.all_semantic_hms = self.f['semantic_hms']
        self.all_gt_3d = self.f['gt_3d']
        self.all_gt_2d = self.f['gt_2d']
        self.all_gt_semantics_3d = self.f['gt_semantics_3d']
        self.all_gt_semantics = self.f['gt_semantics']
        self.all_depths = self.f['depths']

        dataset_keys = list(self.f["cameras"].keys())  # e.g., ["position", "intrinsic_matrix", ...]
        camera_dsets = {key: self.f["cameras"][key] for key in dataset_keys}
        num_samples, num_objects = camera_dsets[dataset_keys[0]].shape[:2]

        self.all_cameras = [
            [
                {
                    key: camera_dsets[key][i, j]
                    for key in dataset_keys
                }
                for j in range(num_objects)
            ]
            for i in range(num_samples)
        ]
        del dataset_keys, camera_dsets, num_samples, num_objects

        self.fsc = freeSpaceCalculator()

    def get_sampled_hms_and_depths(self,selected_indices,idx):
        sampled_hms = []
        sampled_semantics = []
        sampled_depths = []
        sampled_cameras = []
        for i in selected_indices:
            sampled_hms.append(self.all_hms[idx,i])
            sampled_semantics.append(self.all_semantic_hms[idx,i])
            sampled_depths.append(self.all_depths[idx,i].astype(float)/1000)
            if self.use_continous_cameras:
                sampled_cameras.append(self.all_cameras[idx][i])
            
        sampled_hms = numpy.array(sampled_hms)[:,10:-10]
        invalid = sampled_hms[:,:,:,0] > 0.28
        sampled_hms[invalid] = 0
        sampled_semantics = numpy.array(sampled_semantics)[:,10:-10]
        sampled_semantics[invalid] = 15
        sampled_depths = numpy.array(sampled_depths)
        # reshuffling the indices for added randomization
        sampled_viewpoints = selected_indices
    
        return sampled_hms,sampled_semantics,sampled_depths,sampled_viewpoints, sampled_cameras
        
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = np.array(idx.tolist())
            idx = idx*self.skip
        selected_indices = sorted(self.rng.choice(numpy.arange(0,self.all_hms.shape[1]),self.max_samples,replace = False))
        # sampled indices in order for easy memory retrieval
        

        gt_3d = self.all_gt_3d[idx][:,10:-10,:]
        gt_3d[:,:20,:] = False
        gt_2d = self.all_gt_2d[idx][:,10:-10,:]
        gt_semantics_3d = self.all_gt_semantics_3d[idx][:,10:-10,:]
        gt_semantics = self.all_gt_semantics[idx][10:-10,:]

        sampled_hms,sampled_semantics,sampled_depths,sampled_viewpoints, sampled_cameras = self.get_sampled_hms_and_depths(selected_indices,idx)

        if(self.noise):
            noise = self.rng.normal(0,scale = 0.1,size = sampled_depths.shape)
            salt_noise = self.rng.choice([False,True],p = [0.995,0.005],replace = True,size = sampled_depths.shape) 
            sampled_depths[salt_noise] += noise[salt_noise]
            if self.use_continous_cameras:
                free,occupied = self.get_all_free_and_occupied(sampled_depths, sampled_viewpoints, sampled_cameras)
            else:
                free,occupied = self.get_all_free_and_occupied(sampled_depths, sampled_viewpoints)
            stuff = occupied[:,:,:,:].sum(axis =3)>0
            null = stuff == False
            sampled_semantics[null] = 15
            semantic_stuff = sampled_semantics <=14
            new_points = ((semantic_stuff).astype(int)-(stuff>0).astype(int))!=0
            salt_noise = self.rng.choice([False,True],p = [0.95,0.05],replace = True,size = sampled_semantics.shape) 
            noised_points = (new_points.astype(int) + salt_noise.astype(int)) > 0
            semantic_noise = self.rng.choice(numpy.arange(16),replace = True,size = sampled_semantics.shape)
            sampled_semantics[noised_points] = semantic_noise[noised_points]
            
            
        else:
            if self.use_continous_cameras:
                free, occupied = self.get_all_free_and_occupied(sampled_depths, sampled_viewpoints, sampled_cameras)
            else:
                free, occupied = self.get_all_free_and_occupied(sampled_depths, sampled_viewpoints)
            stuff = occupied[:,:,:,:].sum(axis =3)>0
            null = stuff == False
            sampled_semantics[null] = 15

        # pdb.set_trace()

        if(self.move_and_rotate):
            sampled_semantics,gt_semantics,gt_3d,gt_semantics_3d,gt_2d,free,occupied = self.augment_rotation_translation(sampled_semantics,gt_semantics,gt_3d,gt_semantics_3d,gt_2d,free,occupied)

        stuff = occupied[:,:,:,:].sum(axis =3)>0
        null = stuff == False
        sampled_semantics[null] = 15

        occupied = numpy.moveaxis(occupied,[1,2,3],[2,3,1])
        free = numpy.moveaxis(free,[1,2,3],[2,3,1])



        return {'gt_3d':gt_3d,
            'free':free,'occupied':occupied,'gt_2d':gt_2d,'hms':sampled_hms,
            'semantic_hms':sampled_semantics,'gt_semantics':gt_semantics,
            'gt_semantics_3d':gt_semantics_3d}


    def get_all_free_and_occupied(self,sampled_depths, sampled_viewpoints, sampled_cameras=None):
        frees = []
        occupieds = []
        for depth, viewpoint, camera in zip(sampled_depths,sampled_viewpoints,sampled_cameras):
            if sampled_cameras is None:
                intrinsic = self.intrinsics[viewpoint]
                extrinsic = self.extrinsics[viewpoint]
            else:
                intrinsic = camera["intrinsic_matrix"].reshape(3, 3, order='C')
                extrinsic = camera['o3d_extrinsic_matrix'].reshape(4, 4, order='C')

            free,occupied = self.fsc.get_free_space(intrinsic,extrinsic,depth)
            free = numpy.moveaxis(free,[0,1],[1,0])
            occupied = numpy.moveaxis(occupied,[0,1],[1,0])
            frees.append(free)
            occupieds.append(occupied)
        free = numpy.array(frees)
        occupied = numpy.array(occupieds)
        return free[:,10:-10,:],occupied[:,10:-10,:]


    def augment_rotation_translation(self,sampled_semantics,gt_semantics,gt_3d,gt_semantics_3d,gt_2d,free,occupied):

        angle = np.random.uniform(-15,15)
        translation = np.random.uniform(-10,10,2).tolist()

        sampled_semantics = torch.from_numpy(sampled_semantics)
        gt_semantics = torch.from_numpy(gt_semantics)
        free = torch.from_numpy(free)
        occupied = torch.from_numpy(occupied)
        gt_3d = torch.from_numpy(gt_3d)
        gt_semantics_3d = torch.from_numpy(gt_semantics_3d)
        gt_2d = torch.from_numpy(gt_2d)

        augmented_semantics = self.affine(sampled_semantics,angle,translation,1,0).numpy()
        augmented_semantic_gt = self.affine(gt_semantics.unsqueeze(0),angle,translation,1,0).squeeze().numpy()
        augmented_free = self.affine(free.permute(0,3,1,2),angle,translation,1,0).permute(0,2,3,1).numpy()
        augmented_occupied = self.affine(occupied.permute(0,3,1,2),angle,translation,1,0).permute(0,2,3,1).numpy()

        augmented_gt_3d = self.affine(gt_3d,angle,translation,1,0).numpy()
        augmented_gt_semantics_3d = self.affine(gt_semantics_3d,angle,translation,1,0).numpy()
        augmented_gt_2d = self.affine(gt_2d.float(),angle,translation,1,0).half().numpy()
        augmented_semantic_gt[augmented_gt_2d[1] == 0] = self.n_classes-1
        return augmented_semantics,augmented_semantic_gt,augmented_gt_3d,augmented_gt_semantics_3d,augmented_gt_2d,augmented_free,augmented_occupied


    def local_find_update_cells(self,hms):
        cp_hms = np.array(hms)
        clipped_hm = np.round(np.clip(cp_hms[:,:,:]/self.height_resolution, 0, self.height_bins)).astype(np.uint8)
        if(len(cp_hms.shape)==4):
            unchanged = cp_hms[:,:,:,0] ==0
        else:
            unchanged = cp_hms[:,:,:] == 0
        if(len(cp_hms.shape)==4):
            occupied = self.height_grid[np.newaxis,:,:,:] < clipped_hm[:,:,:,0][:,:,:,np.newaxis]
        else:
            occupied = self.height_grid[np.newaxis,:,:,:] < clipped_hm[:,:,:,np.newaxis]

        free = np.logical_not(occupied)
        occupied[unchanged,:] = False
        free[unchanged,:] = False
        if(len(cp_hms.shape)==4):
            free[cp_hms[:,:,:,1].astype(bool)] = False 
        occupied = np.moveaxis(occupied,[1,2,3],[2,3,1])
        free = np.moveaxis(free,[1,2,3],[2,3,1])
        return occupied,free

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset.dataset
        dataset.reset_h5py()


class PushDatasetH5pyNPY(MapDatasetH5py):
    def __init__(self,dataset_location,max_samples = 10,n_classes = 15,noise = True,move_and_rotate = True,camera_params_dir = '../../camera_matrices.npz'):
        self.dataset_location = dataset_location
        self.f = h5py.File(self.dataset_location,'r',locking = False,swmr = True)
        self.all_hms = self.f['hms']
        self.size = self.all_hms.shape[0]
        # self.all_semantic_hms = self.f['semantic_hms']
        # self.all_gt_3d = self.f['gt_3d']
        # self.all_gt_2d = self.f['gt_2d']
        # self.all_gt_semantics_3d = self.f['gt_semantics_3d']
        # self.all_gt_semantics = self.f['gt_semantics']
        self.f.close()
        del self.f
        del self.all_hms
        self.max_samples = max_samples
        self.rng = np.random.default_rng()
        self.mapper = Mapping(height = 120)
        self.height_bins = cp.asnumpy(self.mapper.height_bins.copy())
        self.height_grid = cp.asnumpy(self.mapper.height_grid.numpy().copy())
        self.height_resolution = cp.asnumpy(self.mapper.height_resolution)
        self.n_classes = n_classes
        self.noise = noise
        self.move_and_rotate = move_and_rotate
        self.affine = v2.functional.affine
        self.camera_params_dir = camera_params_dir
        camera_matrices = np.load(self.camera_params_dir,allow_pickle = True)['obj_ids']
        self.extrinsics = []
        self.intrinsics = []
        for camera in range(len(camera_matrices)):
            self.extrinsics.append(np.linalg.inv(camera_matrices[camera]['o3d_extrinsic_matrix'].reshape(4,4,order = 'C')))
            self.intrinsics.append(camera_matrices[camera]['intrinsic_matrix'].reshape(3,3,order = 'C'))
        del self.mapper

    def __len__(self):
        return self.size
    

    def get_height_grid(self):
        self.height_resolution = 0.005
        self.max_height = 0.5
        self.height_bins = np.round(self.max_height/self.height_resolution).astype(int)
        self.height_grid = np.zeros((120,200,102)).astype(np.uint8)
        for i in range(self.height_grid.shape[2]):
            self.height_grid[:, :, i] = i
    def reset_h5py(self):
        try:
            self.f.close()
        except Exception as e:
            pass
        self.f = h5py.File(self.dataset_location,'r',locking = False,swmr = True)
        self.all_hms = self.f['hms']
        self.all_semantic_hms = self.f['semantic_hms']
        self.all_gt_3d = self.f['gt_3d']
        self.all_gt_2d = self.f['gt_2d']
        self.all_gt_semantics = self.f['gt_semantics']
        self.all_post_push_gt_3d = self.f['post_push_gt_3d']
        self.all_post_push_gt_semantics = self.f['post_push_gt_semantics']
        self.all_differences = self.f['differences']
        self.swept_volumes = self.f['swept_volume']
        self.push_parametrizations = self.f['push_parametrization']
        self.all_depths = self.f['depths']

        self.fsc = freeSpaceCalculator()


    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = np.array(idx.tolist())
        selected_indices = sorted(self.rng.choice(np.arange(0,self.all_hms.shape[1]),self.max_samples,replace = False))
        # sampled indices in order for easy memory retrieval
        # sampled_hms = self.all_hms[idx,selected_indices]


        # sampled_semantics = self.all_semantic_hms[idx,selected_indices]
        # # reshuffling the indices for added randomization
        # shuffle_indices = self.rng.choice(np.arange(0,self.max_samples),self.max_samples,replace = False)
        # sampled_hms = sampled_hms[shuffle_indices][:,10:-10,:]
        # sampled_semantics = sampled_semantics[shuffle_indices][:,10:-10,:]
        # sampled_semantics[sampled_hms[:,:,:,0] == 0] = self.n_classes
        try:
            sampled_hms,sampled_semantics,sampled_depths,sampled_viewpoints, sampled_cameras = self.get_sampled_hms_and_depths(selected_indices,idx)
        except Exception as e:
            print(e,'index {} failed, defaulting to zero'.format(idx))            
            idx = 0
            sampled_hms,sampled_semantics,sampled_depths,sampled_viewpoints, sampled_cameras = self.get_sampled_hms_and_depths(selected_indices,idx)


        # if(self.noise):
        #     height_maps_size = sampled_hms[:,:,:,0].shape
        #     salt_noise = self.rng.choice([False,True],p = [0.95,0.05],replace = True,size = height_maps_size) 
        #     sampled_hms[salt_noise,0] = self.rng.uniform(0,0.3,size = height_maps_size)[salt_noise]
        #     non_zero_hms = sampled_hms[:,:,:,0] > 0
        #     noise = self.rng.normal(0,0.02,size = height_maps_size)
        #     sampled_hms[non_zero_hms,0] += noise[non_zero_hms]
        #     sampled_hms[sampled_hms <0] = 0
        #     semantic_noise = self.rng.choice(np.arange(16),replace = True,size = height_maps_size)
        #     sampled_semantics[salt_noise] = semantic_noise[salt_noise]

        if(self.noise):
            noise = self.rng.normal(0,scale = 0.1,size = sampled_depths.shape)
            salt_noise = self.rng.choice([False,True],p = [0.995,0.005],replace = True,size = sampled_depths.shape) 
            sampled_depths[salt_noise] += noise[salt_noise]
            free,occupied = self.get_all_free_and_occupied(sampled_depths,sampled_viewpoints)
            stuff = occupied[:,:,:,:].sum(axis =3)>0
            null = stuff == False
            sampled_semantics[null] = 15
            semantic_stuff = sampled_semantics <=14
            new_points = ((semantic_stuff).astype(int)-(stuff>0).astype(int))!=0
            salt_noise = self.rng.choice([False,True],p = [0.95,0.05],replace = True,size = sampled_semantics.shape) 
            noised_points = (new_points.astype(int) + salt_noise.astype(int)) > 0
            semantic_noise = self.rng.choice(numpy.arange(16),replace = True,size = sampled_semantics.shape)
            sampled_semantics[noised_points] = semantic_noise[noised_points]
        else:
            free,occupied = self.get_all_free_and_occupied(sampled_depths,sampled_viewpoints)
            stuff = occupied[:,:,:,:].sum(axis =3)>0
            null = stuff == False
            sampled_semantics[null] = 15


        gt_3d = self.all_gt_3d[idx][:,10:-10,:]
        gt_3d[:,:20,:] = False

        gt_2d = self.all_gt_2d[idx][:,10:-10,:]
#         gt_semantics_3d = self.all_gt_semantics_3d[idx][:,10:-10,:]
        gt_semantics = self.all_gt_semantics[idx][10:-10,:]
        post_push_gt_semantics = self.all_post_push_gt_semantics[idx][10:-10,:]
        post_push_gt_3d = self.all_post_push_gt_3d[idx][:,10:-10,:]
        swept_volume = self.swept_volumes[idx][:,10:-10,:]
        push_parametrization = self.push_parametrizations[idx]
        difference = self.all_differences[idx][10:-10,:]
        # pdb.set_trace()

        # occupied,free = self.local_find_update_cells(sampled_hms)

        if(self.move_and_rotate):
            sampled_hms,sampled_semantics,gt_semantics,gt_3d,gt_2d,post_push_gt_semantics,post_push_gt_3d,swept_volume,push_parametrization,difference,occupied,free = self.augment_rotation_translation(sampled_hms,sampled_semantics,gt_semantics,gt_3d,gt_2d,post_push_gt_semantics,post_push_gt_3d,swept_volume,push_parametrization,difference,occupied,free)

        occupied = numpy.moveaxis(occupied,[1,2,3],[2,3,1])
        free = numpy.moveaxis(free,[1,2,3],[2,3,1])


        return {'gt_3d':gt_3d,'free':free,'occupied':occupied,'gt_2d':gt_2d,
                'hms':sampled_hms,'semantic_hms':sampled_semantics,
                'gt_semantics':gt_semantics,
                'post_push_gt_semantics':post_push_gt_semantics,
                'post_push_gt_3d':post_push_gt_3d,'swept_volume':swept_volume,
                'push_parametrization':push_parametrization,
                'difference':difference, #gt_semantics_3d':gt_semantics_3d,
                'sampled_depths':sampled_depths
               }



    def augment_rotation_translation(self,sampled_hms,sampled_semantics,gt_semantics,gt_3d,gt_2d,post_push_gt_semantics,post_push_gt_3d,swept_volume,push_parametrization,difference,occupied,free):

        angle = np.random.uniform(-15,15)
        translation = np.random.uniform(-10,10,2).tolist()


        sampled_semantics = torch.from_numpy(sampled_semantics)
        gt_semantics = torch.from_numpy(gt_semantics)
        sampled_hms = torch.from_numpy(sampled_hms)
        gt_3d = torch.from_numpy(gt_3d)
        gt_2d = torch.from_numpy(gt_2d)

        post_push_gt_semantics = torch.from_numpy(post_push_gt_semantics)
        post_push_gt_3d = torch.from_numpy(post_push_gt_3d)
        swept_volume = torch.from_numpy(swept_volume)
        push_parametrization = torch.from_numpy(push_parametrization)
        difference = torch.from_numpy(difference)

        free = torch.from_numpy(free)
        occupied = torch.from_numpy(occupied)

        p1 = push_parametrization[:2]
        p2 = push_parametrization[3:5]
        augmented_push_parametrization = torch.clone(push_parametrization)

        
        augmented_semantics = self.affine(sampled_semantics,angle,translation,1,0).numpy()
        augmented_semantic_gt = self.affine(gt_semantics.unsqueeze(0),angle,translation,1,0).squeeze().numpy()
        augmented_hms = self.affine(sampled_hms.permute(0,3,1,2).float(),angle,translation,1,0).permute(0,2,3,1).half().numpy()
        augmented_gt_3d = self.affine(gt_3d,angle,translation,1,0).numpy()
        augmented_gt_2d = self.affine(gt_2d.float(),angle,translation,1,0).half().numpy()
        augmented_post_push_gt_semantics = self.affine(post_push_gt_semantics.unsqueeze(0),angle,translation,1,0).squeeze().numpy()
        augmented_post_push_gt_3d = self.affine(post_push_gt_3d,angle,translation,1,0).numpy()
        augmented_swept_volume = self.affine(swept_volume,angle,translation,1,0).numpy()
        augmented_difference = self.affine(difference.unsqueeze(0),angle,translation,1,0).squeeze().numpy()
        augmented_free = self.affine(free.permute(0,3,1,2),angle,translation,1,0).permute(0,2,3,1).numpy()
        augmented_occupied = self.affine(occupied.permute(0,3,1,2),angle,translation,1,0).permute(0,2,3,1).numpy()


        # rotating the push parametrization:
        augmented_p1,r1,r11 = self.augment_push_parametrization(p1,gt_semantics,angle,translation)
        augmented_push_parametrization[0] = augmented_p1[0][0]
        augmented_push_parametrization[1] = augmented_p1[1][0]
        augmented_p2,r2,r22 = self.augment_push_parametrization(p2,gt_semantics,angle,translation)
        augmented_push_parametrization[3] = augmented_p2[0][0]
        augmented_push_parametrization[4] = augmented_p2[1][0]
        augmented_push_parametrization = augmented_push_parametrization.numpy()

        augmented_semantics[augmented_hms[:,:,:,0] == 0] = 15
        augmented_semantic_gt[augmented_gt_2d[1] == 0] = 14
        augmented_post_push_gt_semantics[augmented_post_push_gt_3d.sum(axis =0)==0] = 14



        return augmented_hms,augmented_semantics,augmented_semantic_gt,augmented_gt_3d,augmented_gt_2d,augmented_post_push_gt_semantics,augmented_post_push_gt_3d,augmented_swept_volume,augmented_push_parametrization,augmented_difference,augmented_occupied,augmented_free

    def augment_push_parametrization(self,p,gt_semantics,angle,translation):
        zeros = torch.zeros_like(gt_semantics)
        x = torch.clamp(p[0],0,119)
        y = torch.clamp(p[1],0,199)
        zeros[x,y] = 1
        augmented_zeros = self.affine(zeros.unsqueeze(0),angle,translation,1,0).squeeze().numpy()
        if(np.all(augmented_zeros==0)):
            augmented_p = ([x],[y])
        else:
            augmented_p = np.where(augmented_zeros>0)
        return augmented_p,augmented_zeros,zeros

    def local_find_update_cells(self,hms):
        # pdb.set_trace()
        cp_hms = np.array(hms)
        # if(self.noise_level > 0):
        #     noised = self.rng.choice([False,True],size = cp_hms.shape,replace = True, p = [1-self.noise_level,self.noise_level])
        #     cp_hms[noised] = self.rng.uniform(0.05,0.35,cp_hms.shape)[noised]
        
        clipped_hm = np.round(np.clip(cp_hms[:,:,:,0]/self.height_resolution, 0, self.height_bins)).astype(np.uint8)
        unchanged = cp_hms[:,:,:,0] ==0
#         pdb.set_trace()
        occupied = self.height_grid[np.newaxis,:,:,:] < clipped_hm[:,:,:,np.newaxis]
        # if(self.noise_level>0):
        #     pepper_noise = self.rng.choice([0,1],occupied[:,:,:70].shape,replace = True,p = [1-self.noise_level,self.noise_level])
        #     occupied[pepper_noise] = True
        #     occupied[:,:,:70] = occupied[:,:,:70][:,:,::-1].cumsum(axis = 2)[:,:,::-1]>0
        free = np.logical_not(occupied)
#         pdb.set_trace()
        # if(self.noise_level > 0):
        #     occupied[unchanged,:] = False
        free[unchanged,:] = False
        free[cp_hms[:,:,:,1].astype(bool)] = False 
        occupied = np.moveaxis(occupied,[1,2,3],[2,3,1])
        free = np.moveaxis(free,[1,2,3],[2,3,1])
        return occupied,free

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset.dataset
        dataset.reset_h5py()