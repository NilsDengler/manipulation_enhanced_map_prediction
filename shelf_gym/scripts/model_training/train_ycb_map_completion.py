import multiprocessing
import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

import lightning as L
import numpy as np
import scipy
import torch
from torch import nn
import torch.nn.functional as F
from glob import glob
import cv2
from matplotlib import pyplot as plt
import pdb
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from shelf_gym.utils.learning_utils.losses import evidential_occupancy_crossentropy,evidential_semantic_crossentropy
from shelf_gym.utils.learning_utils.datasets import MapDatasetH5py
from shelf_gym.utils.learning_utils.metrics import map_accuracy,semantic_accuracy
from shelf_gym.utils.learning_utils.data_preprocessing import DataPrepper
# from my_calibration import Calibration_calc_3D
from shelf_gym.utils.models.UNet import UNet,SemanticUNet
from tqdm import tqdm
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
import pdb
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler
import gc
import argparse
import wandb




class SemanticMapPredictorYCB(L.LightningModule):
    def __init__(self, predictor, mode, lr=0.001, n_classes=7, k_type='fixed', k=0, update_frequency='every',
                 total_steps=30, recursive=False, normalize=False,max_alpha = 50):
        super().__init__()
        self.predictor = predictor
        self.mode = mode
        self.automatic_optimization = False
        self.lr = lr
        self.n_classes = n_classes
        self.total_steps = total_steps
        self.norm = 'inf'
        self.n_classes
        self.valid_models = []
        self.non_recursive_models = ['3D_denoising','3D_semantic_augmented_denoising','3D_augmented_denoising']


        assert k_type in ['fixed',
                          'variable'], 'selected k_type {} is NOT a valid k_type. Please select a mode that is in [fixed,variable]'
        self.k_type = k_type
        self.k = k
        assert update_frequency in ['every',
                                    'end'], 'Selected update_frequency choice {} is not a valid option. Please choose one in [end,every]'
        self.update_frequency = update_frequency
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.normalize = normalize
        self.max_alpha = max_alpha
        self.dp = DataPrepper(device, self.n_classes, self.mode, self.normalize,max_alpha = self.max_alpha)
        self.recursive = recursive
        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.semantic_weight = 1
        self.save_hyperparameters()

        # self.cal = Calibration_calc_3D(no_void = False,one_hot = False)

    def get_mixing_coefficient(self):
        if (self.k_type == 'fixed'):
            return self.k
        elif (self.k_type == 'variable'):
            new_k = min(self.k*self.global_step / self.total_steps, self.k)
            return new_k
        else:
            raise NotImplementedError('This k_type mode ({}) is not yet implemented '.format(self.k_type))

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        self.predictor.train()
        # self.predictor.eval()
        optim = self.optimizers()
        gt_3d, permuted_free, permuted_occupied, permuted_semantics, gt_semantics = self.dp.data_prep(batch)
        # pdb.set_trace()
        with torch.no_grad():
            previous_map, previous_semantic_map = self.dp.get_initial_map(permuted_free)
        optim.zero_grad()
        step_losses = []
        step_accuracies = []
        semantic_accuracies = []
        sem_losses = []
        occ_losses = []
        torch.cuda.empty_cache()
        for occupied, free, semantics in zip(permuted_occupied, permuted_free, permuted_semantics):

            if (self.update_frequency == 'every'):
                optim.zero_grad()
            with torch.no_grad():
                model_input, previous_semantic_map = self.dp.get_model_input(occupied, free, previous_map,
                                                                             previous_semantic_map, semantics)
            occupancy_map, semantic_map = self.predictor(model_input, previous_semantic_map)
            # pdb.set_trace()

            a = occupancy_map[:, ::2, :, :]
            b = occupancy_map[:, 1::2, :, :]
            occupancy_beta = torch.stack([a, b], dim=1)
            this_k = self.get_mixing_coefficient()
            occupancy_loss = evidential_occupancy_crossentropy(occupancy_beta, gt_3d, k=this_k)
            semantic_loss = evidential_semantic_crossentropy(semantic_map, gt_semantics, k=this_k,n_classes = self.n_classes)
            sem_losses.append(semantic_loss.detach().cpu().numpy().tolist())
            occ_losses.append(occupancy_loss.detach().cpu().numpy().tolist())

            this_loss = occupancy_loss + semantic_loss/self.semantic_weight
            # print(this_loss.detach().cpu().numpy())
            self.manual_backward(this_loss)
            acc = map_accuracy(occupancy_beta.detach(), gt_3d).cpu().numpy().tolist()
            step_accuracies.append(acc)
            sem_acc = semantic_accuracy(semantic_map.detach(), gt_semantics)
            semantic_accuracies.append(sem_acc.cpu().numpy().tolist())
            step_losses.append(this_loss.detach().cpu().numpy().tolist())
            if (self.recursive):
                previous_map = occupancy_map.detach().clone()
                previous_semantic_map = semantic_map.detach().clone()
            else:
                if (self.mode not in self.non_recursive_models):
                    raise NotImplementedError(
                        'The only non-recursive mode implemented is 3D denoising. You selected {}'.format(self.mode))
            del sem_acc, acc
            # torch.nn.utils.clip_grad_norm_(self.predictor.parameters(),0.5,error_if_nonfinite = True,norm_type = self.norm)
            del free, occupied, semantics, occupancy_map, semantic_map, model_input
            if (self.update_frequency == 'every'):
                self.clip_gradients(optim, gradient_clip_val=0.5, gradient_clip_algorithm='norm')
                del occupancy_loss, semantic_loss
                optim.step()

        if (self.update_frequency == 'end'):
            self.clip_gradients(optim, gradient_clip_val=0.5, gradient_clip_algorithm='norm')
            optim.step()
        loss = np.mean(step_losses)
        accuracy = np.mean(step_accuracies)
        sem_accuracy = np.mean(semantic_accuracies)
        sem_loss = np.mean(sem_losses)
        occ_loss = np.mean(occ_losses)

        # print(step_accuracies[0],semantic_accuracies[0],step_accuracies[0].shape,semantic_accuracies[0].shape,type(step_accuracies[0]),type(semantic_accuracies[0]),)
        split = 'train'
        self.log_dict({'{}_loss'.format(split): loss, '{}_occupancy_accuracy'.format(split): accuracy,
                       '{}_semantic_accuracy'.format(split): sem_accuracy,
                       '{}_early_accuracy'.format(split): step_accuracies[0],
                       '{}_late_accuracy'.format(split): step_accuracies[-1],
                       '{}_early_semantic_accuracy'.format(split): semantic_accuracies[0],
                       '{}_late_semantic_accuracy'.format(split): semantic_accuracies[-1],
                       '{}_semantic_loss'.format(split): sem_loss, '{}_occupancy_loss'.format(split): occ_loss})
        del sem_losses
        del occ_losses
        del step_losses, step_accuracies, semantic_accuracies
        torch.cuda.empty_cache()
        gc.collect()

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        self.predictor.eval()
        with torch.no_grad():
            gt_3d, permuted_free, permuted_occupied, permuted_semantics, gt_semantics = self.dp.data_prep(batch)
            # pdb.set_trace()
            previous_map, previous_semantic_map = self.dp.get_initial_map(permuted_free)
            step_losses = []
            step_accuracies = []
            semantic_accuracies = []
            sem_losses = []
            occ_losses = []
            torch.cuda.empty_cache()
            for occupied, free, semantics in zip(permuted_occupied, permuted_free, permuted_semantics):

                model_input, previous_semantic_map = self.dp.get_model_input(occupied, free, previous_map,
                                                                             previous_semantic_map, semantics)
                occupancy_map, semantic_map = self.predictor(model_input, previous_semantic_map)
                a = occupancy_map[:, ::2, :, :]
                b = occupancy_map[:, 1::2, :, :]
                occupancy_beta = torch.stack([a, b], dim=1)
                if (self.k_type == 'fixed'):
                    k = self.get_mixing_coefficient()
                else:
                    k =max(1,self.k)
                occupancy_loss = evidential_occupancy_crossentropy(occupancy_beta, gt_3d, k=k)
                semantic_loss = evidential_semantic_crossentropy(semantic_map, gt_semantics, k=k,n_classes = self.n_classes)
                sem_losses.append(semantic_loss.detach().cpu().numpy().tolist())
                occ_losses.append(occupancy_loss.detach().cpu().numpy().tolist())
                this_loss = occupancy_loss + semantic_loss/self.semantic_weight

                acc = map_accuracy(occupancy_beta.detach(), gt_3d).cpu().numpy().tolist()
                step_accuracies.append(acc)
                sem_acc = semantic_accuracy(semantic_map.detach(), gt_semantics)
                semantic_accuracies.append(sem_acc.cpu().numpy().tolist())
                step_losses.append(this_loss.detach().cpu().numpy().tolist())
                if (self.recursive):
                    previous_map = occupancy_map.detach().clone()
                    previous_semantic_map = semantic_map.detach().clone()
                else:
                    if (self.mode not in self.non_recursive_models):
                        raise NotImplementedError(
                            'The only non-recursive mode implemented is 3D denoising. You selected {}'.format(
                                self.mode))
                del free, occupied, semantics, occupancy_map, semantic_map, model_input
                del sem_acc, acc

            loss = np.mean(step_losses)
            accuracy = np.mean(step_accuracies)
            sem_accuracy = np.mean(semantic_accuracies)
            sem_loss = np.mean(sem_losses)
            occ_loss = np.mean(occ_losses)

        split = 'val'
        self.log_dict({'{}_loss'.format(split): loss, '{}_occupancy_accuracy'.format(split): accuracy,
                       '{}_semantic_accuracy'.format(split): sem_accuracy,
                       '{}_early_accuracy'.format(split): step_accuracies[0],
                       '{}_late_accuracy'.format(split): step_accuracies[-1],
                       '{}_early_semantic_accuracy'.format(split): semantic_accuracies[0],
                       '{}_late_semantic_accuracy'.format(split): semantic_accuracies[-1],
                       '{}_semantic_loss'.format(split): sem_loss, '{}_occupancy_loss'.format(split): occ_loss},
                      on_epoch=True)
        self.predictor.train()
        torch.cuda.empty_cache()
        gc.collect()

    def test_step(self, batch, bathc_idx):
        torch.cuda.empty_cache()

        self.predictor.eval()
        with torch.no_grad():
            gt_3d, permuted_free, permuted_occupied, permuted_semantics, gt_semantics = self.dp.data_prep(batch)
            # pdb.set_trace()
            previous_map, previous_semantic_map = self.dp.get_initial_map(permuted_free)
            step_losses = []
            step_accuracies = []
            semantic_accuracies = []
            sem_losses = []
            occ_losses = []
            torch.cuda.empty_cache()
            for occupied, free, semantics in zip(permuted_occupied, permuted_free, permuted_semantics):

                model_input, previous_semantic_map = self.dp.get_model_input(occupied, free, previous_map,
                                                                             previous_semantic_map, semantics)
                occupancy_map, semantic_map = self.predictor(model_input, previous_semantic_map)
                a = occupancy_map[:, ::2, :, :]
                b = occupancy_map[:, 1::2, :, :]
                occupancy_beta = torch.stack([a, b], dim=1)
                if (self.k_type == 'fixed'):
                    k = self.get_mixing_coefficient()
                else:
                    k =max(1,self.k)
                occupancy_loss = evidential_occupancy_crossentropy(occupancy_beta, gt_3d, k=k)
                semantic_loss = evidential_semantic_crossentropy(semantic_map, gt_semantics, k=k,n_classes = self.n_classes)
                sem_losses.append(semantic_loss.detach().cpu().numpy().tolist())
                occ_losses.append(occupancy_loss.detach().cpu().numpy().tolist())
                this_loss = occupancy_loss + semantic_loss/self.semantic_weight
                acc = map_accuracy(occupancy_beta, gt_3d).cpu().numpy().tolist()
                step_accuracies.append(acc)
                sem_acc = semantic_accuracy(semantic_map, gt_semantics)
                semantic_accuracies.append(sem_acc.cpu().numpy().tolist())
                step_losses.append(this_loss.cpu().numpy().tolist())
                if (self.recursive):
                    previous_map = occupancy_map.detach().clone()
                    previous_semantic_map = semantic_map.detach().clone()
                else:
                    if (self.mode not in self.non_recursive_models):
                        raise NotImplementedError(
                            'The only non-recursive mode implemented is 3D denoising. You selected {}'.format(
                                self.mode))

            loss = np.mean(step_losses)
            accuracy = np.mean(step_accuracies)
            sem_accuracy = np.mean(semantic_accuracies)
            sem_loss = np.mean(sem_losses)
            occ_loss = np.mean(occ_losses)
        split = 'test'
        self.log_dict({'{}_loss'.format(split): loss, '{}_occupancy_accuracy'.format(split): accuracy,
                       '{}_semantic_accuracy'.format(split): sem_accuracy,
                       '{}_early_accuracy'.format(split): step_accuracies[0],
                       '{}_late_accuracy'.format(split): step_accuracies[-1],
                       '{}_early_semantic_accuracy'.format(split): semantic_accuracies[0],
                       '{}_late_semantic_accuracy'.format(split): semantic_accuracies[-1],
                       '{}_semantic_loss'.format(split): sem_loss, '{}_occupancy_loss'.format(split): occ_loss})
        self.predictor.train()
        torch.cuda.empty_cache()
        gc.collect(0)
        print(
            f"[GPU Memory] Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB | Reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")

    def configure_optimizers(self):
        return torch.optim.Adam(self.predictor.parameters(), lr=self.lr)

    def get_outputs(self, batch,intermediates = True,max_obs = None,normalize = False,previous_map = None,previous_semantic_map = None):
        self.predictor.eval()
        with torch.no_grad():
            gt_3d, permuted_free, permuted_occupied, permuted_semantics, gt_semantics = self.dp.data_prep(batch)
            # pdb.set_trace()
            if((previous_map is None) or (previous_semantic_map is None)):
                previous_map, previous_semantic_map = self.dp.get_initial_map(permuted_free)
            occupancy_predictions = []
            semantic_predictions = []
            torch.cuda.empty_cache()
            if(max_obs is None):
                max_obs = permuted_occupied.shape[0]
            for occupied, free, semantics in zip(permuted_occupied, permuted_free, permuted_semantics):

                model_input, previous_semantic_map = self.dp.get_model_input(occupied, free, previous_map,
                                                                             previous_semantic_map, semantics)
                occupancy_map, semantic_map = self.predictor(model_input, previous_semantic_map)
                a = occupancy_map[:, ::2, :, :]
                b = occupancy_map[:, 1::2, :, :]
                occupancy_beta = torch.stack([a, b], dim=1)
                if(normalize):
                    occupancy_probs = (occupancy_beta / occupancy_beta.sum(axis=1, keepdims=True))
                    semantic_probs = semantic_map / semantic_map.sum(axis=1, keepdims=True)
                else:
                    occupancy_probs = occupancy_beta.detach().clone()
                    semantic_probs = semantic_map.detach().clone()
                if(intermediates):
                    occupancy_predictions.append(occupancy_probs.permute(0, 2, 3, 4, 1).cpu().numpy())
                    semantic_predictions.append(semantic_probs.permute(0, 2, 3, 1).cpu().numpy())
                else:
                    del occupancy_beta,model_input
                if (self.recursive):
                    previous_map = occupancy_map.detach().clone()
                    previous_semantic_map = semantic_map.detach().clone()
                else:
                    if (self.mode not in self.non_recursive_models):
                        raise NotImplementedError(
                            'The only non-recursive mode implemented is 3D denoising. You selected {}'.format(
                                self.mode))
        self.predictor.train()
        del permuted_free,permuted_occupied,permuted_semantics
        torch.cuda.empty_cache()

        if(not intermediates):
            occupancy_predictions.append(occupancy_probs.permute(0, 2, 3, 4, 1))
            semantic_predictions.append(semantic_probs.permute(0, 2, 3, 1))
        print(
            f"[GPU Memory] Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB | Reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
        return {'occupancy_gt': gt_3d, 'semantic_gt': gt_semantics, 'occupancy_probs': occupancy_predictions,
                'semantic_probs': semantic_predictions,'occupancy_map':occupancy_map,'semantic_map':semantic_map}


class DataSanitationCallback(Callback):
    def on_train_start(self, *args, **kwargs):
        trainer.train_dataloader.dataset.reset_h5py()

    def on_validation_start(self, *args, **kwargs):
        trainer.val_dataloaders.dataset.reset_h5py()
        trainer.train_dataloader.dataset.reset_h5py()
        # trainer.

    def on_train_end(self, *args, **kwargs):
        trainer.train_dataloader.dataset.reset_h5py()

    def on_validation_end(self, *args, **kwargs):
        trainer.val_dataloaders.dataset.reset_h5py()
        trainer.train_dataloader.dataset.reset_h5py()


if (__name__ == '__main__'):
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--k_type',default = 'variable')
    parser.add_argument('--update_frequency', default = 'every')
    parser.add_argument('--mode', default = '3D_semantic_augmented_all_inputs')
    parser.add_argument('--k', type=float, default=1.0,
                        help="""max (or actual) mixing coefficient between loss and regularization""")
    parser.add_argument('--max_alpha', type=float, default=50.0,
                        help="""Max value for the normalization of the dirichlet (beta) distributions as inputs""")
    parser.add_argument('--do_dropout',action='store_true')
    parser.add_argument('--no_recursion',action='store_false')
    parser.add_argument('--normalize',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--checkpoint',default = "jmc12_team/YCB Semantic Map Completion - Fixed/model-5dburcae:v4")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="""Learning Rate""")
    args = parser.parse_args()



    k_type = args.k_type
    update_frequency = args.update_frequency
    mode = args.mode
    k = args.k
    recursive = args.no_recursion
    do_dropout = args.do_dropout
    normalize = args.normalize
    resume = args.resume
    checkpoint_to_resume = args.checkpoint
    max_alpha = args.max_alpha
    lr = args.lr
    n_classes = 15
    max_epochs = 1000
    total_steps = 1000
    if(not resume):
        val_check_interval = 0.33
    else:
        val_check_interval = 0.2


    torch.set_float32_matmul_precision('medium')
    # torch._dynamo.config.guard_nn_modules = True
    # np.random.seed(42)
    generator = torch.Generator().manual_seed(42)
    skip = 1

    dataset_dir = './map_completion_fine_tune.hdf5'
    dataset = MapDatasetH5py(dataset_dir, max_samples=10, skip=skip, camera_params_dir='../model/camera_matrices.npz')
    # val_set = MapDatasetH5py(val_dir,max_samples = 10,skip = skip)
    # pdb.set_trace()
    # train_set,val_set,test_set = torch.utils.data.random_split(dataset,[0.7,0.2,0.1],generator=generator)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1], generator=generator)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True,
                              num_workers=4, prefetch_factor=1, pin_memory=True, pin_memory_device='cuda',
                              persistent_workers=True, worker_init_fn=MapDatasetH5py.worker_init_fn)
    val_loader = DataLoader(val_set, batch_size=10, shuffle=False,
                            num_workers=4, prefetch_factor=1, pin_memory=True, pin_memory_device='cuda',
                            persistent_workers=True, worker_init_fn=MapDatasetH5py.worker_init_fn)
    # test_loader = DataLoader(test_set,batch_size = 30,shuffle = False,num_workers= 5,prefetch_factor = 1,pin_memory= True, pin_memory_device = 'cuda',persistent_workers = True)

    print(recursive,normalize)
    # mode = '3D_all_inputs'
    # mode = '2D_all_inputs'
    wandb_logger = None
    # update_frequency = 'every'
    # k_type = 'variable'
    if(not resume):
        name = 'Affine_Augmentation_UA_CROSSENTROPY_{}_{}_{}_recursive_{}_Normalized_{}_max_alpha_{}'.format(
        mode,k_type,update_frequency,recursive,normalize,max_alpha).replace('.','_')
    if(resume):
        name = 'Noisy_Resumed_{}_lr_{}_k_{}'.format(checkpoint_to_resume,lr,k).replace('.','_')
    wandb_logger = WandbLogger(name = name,
        project = 'YCB Semantic Map Completion - Fixed',
        log_model ='all',save_dir = './wandb_logs')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if (mode == '3D_all_inputs'):
        underlying_model = SemanticUNet(408, 204, n_semantic_channel_in=2 * n_classes+1, 
                                        do_dropout=do_dropout,max_alpha = max_alpha,normalize = normalize,
                                        n_classes = n_classes).to(device)
    elif (mode == '3D_semantic_augmented_all_inputs'):
        underlying_model = SemanticUNet(408+2*n_classes+1, 204, 
                                        n_semantic_channel_in=2 * n_classes+1, do_dropout=do_dropout,
                                        max_alpha = max_alpha,normalize = normalize,n_classes = n_classes).to(device)
    elif (mode == '3D_denoising'):
        underlying_model = SemanticUNet(204, 204, n_semantic_channel_in=n_classes, 
                                        do_dropout=do_dropout,max_alpha = max_alpha,
                                        normalize = normalize, n_classes = n_classes).to(device)
    elif(mode == '3D_augmented_denoising'):
        underlying_model = SemanticUNet(204+1, 204, n_semantic_channel_in=n_classes, 
                                        do_dropout=do_dropout,max_alpha = max_alpha,
                                        normalize = normalize,n_classes = n_classes).to(device)
    elif(mode == '3D_semantic_augmented_denoising'):
        underlying_model = SemanticUNet(204+1+n_classes, 204, n_semantic_channel_in=n_classes,
                                         do_dropout=do_dropout,max_alpha = max_alpha,
                                         normalize = normalize,n_classes = n_classes).to(device)
        
    if(not resume):
        model = SemanticMapPredictorYCB(predictor=underlying_model, mode=mode, lr=lr, k_type=k_type, k=k,
                                 total_steps=total_steps,n_classes = n_classes,
                                update_frequency=update_frequency, recursive=recursive,normalize = normalize,max_alpha=max_alpha)
    else:

        run = wandb.init(project='eval')

        
        artifact = run.use_artifact(checkpoint_to_resume, type='model')

        checkpoint = artifact.download()+'/model.ckpt'
        run.finish(quiet = True)
        model = SemanticMapPredictorYCB.load_from_checkpoint(checkpoint,k=k,lr=lr,k_type='fixed')

    # compiled_model = torch.compile(model,mode='reduce-overhead')
    # print('\n\ncompiled model!\n\n')
    # prof = SimpleProfiler(dirpath= './debugging_logs/',filename = 'debug_log2.txt')
    prof = None
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=50, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", every_n_train_steps= int(val_check_interval*len(train_loader)+1))
    trainer = L.Trainer(precision="32-true", enable_progress_bar=True,
                        logger=wandb_logger, max_epochs=max_epochs,
                        callbacks=[early_stop_callback, checkpoint_callback],
                        log_every_n_steps=5,val_check_interval=val_check_interval,
                        num_sanity_val_steps=0, use_distributed_sampler=True, accelerator="gpu",)
    # trainer.validate(model=model,dataloaders = val_loader)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)



