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
from shelf_gym.utils.learning_utils.losses import focused_evidential_semantic_crossentropy,evidential_semantic_crossentropy,focused_evidential_occupancy_crossentropy
from shelf_gym.utils.learning_utils.datasets import PushDatasetH5pyNPY
from shelf_gym.utils.learning_utils.metrics import map_accuracy,semantic_accuracy
from shelf_gym.utils.learning_utils.data_preprocessing import DataPrepper
# from my_calibration import Calibration_calc_3D
from shelf_gym.utils.models.UNet import PushSemanticUNet,SemanticUNet
from tqdm import tqdm
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
import pdb
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler
import gc
from shelf_gym.utils.learning_utils.data_preprocessing import PushDataPrepper,to_channels_last
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import MulticlassJaccardIndex
import argparse
import wandb
import multiprocessing


class PushPredictor(L.LightningModule):
    def __init__(self, map_predictor_dir,map_predictor_name,push_predictor, lr=0.001, n_classes=7, k_type='fixed', k=0,
                 total_steps=1000,w2=1,w3=1,focus_factor = 100,max_alpha = 50,normalize = False):
        super().__init__()

        self.lr = lr
        self.n_classes = n_classes
        self.total_steps = total_steps
        self.map_predictor_dir = map_predictor_dir
        self.map_predictor_name = map_predictor_name
        self.map_predictor_checkpoint = self.map_predictor_dir + self.map_predictor_name
        self.push_predictor = push_predictor
        self.normalize = normalize
        assert k_type in ['fixed',
                          'variable'], 'selected k_type {} is NOT a valid k_type. Please select a mode that is in [fixed,variable]'
        self.k_type = k_type
        self.k = k
        self.max_alpha = max_alpha
        self.dp = PushDataPrepper(self.map_predictor_checkpoint,max_alpha = self.max_alpha)
        self.w2 = w2
        self.w3 = w3
        self.w4 = 1/1000
        self.focus_factor = focus_factor
        self.f1 = BinaryF1Score().to('cuda')
        self.iou = MulticlassJaccardIndex(num_classes = self.n_classes).to('cuda')
        self.MSE = nn.L1Loss(reduce = 'mean')
        # save hyper-parameters to self.hparams (auto-logged by W&B)
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
        tmp = self.dp.get_model_inputs(batch)
        torch.cuda.empty_cache()

        map_and_push = tmp['map_and_push']
        semantic_map = tmp['semantic_map']
        
        occupancy_map,semantic_map,pred_difference,_ = self.push_predictor(map_and_push,semantic_map)
        difference = batch['difference'].to(self.device)
        gt_semantics = batch['post_push_gt_semantics'].to(self.device)
        pre_push_gt_semantics = batch['gt_semantics']
        gt_3d = batch['post_push_gt_3d'].to(self.device)
        k = self.get_mixing_coefficient()
        loss1= focused_evidential_semantic_crossentropy(semantic_map,gt_semantics,difference,focus_factor = self.focus_factor,n_classes = self.n_classes,k=k)
        loss2 = focused_evidential_occupancy_crossentropy(occupancy_map,gt_3d,difference,focus_factor= self.focus_factor,k=k)
        loss3 = evidential_semantic_crossentropy(pred_difference,difference,k=k)
        #implementation of the consistency loss
        input_occupancy = tmp['input_occupancy_map']
        input_beta = input_occupancy[:,::2,:,:]
        input_alpha = input_occupancy[:,1::2,:,:]
        not_difference = torch.logical_not(difference)
        input_map = torch.stack([input_beta,input_alpha],dim = 1).permute(0,3,4,2,1)[not_difference].reshape(-1,2)
        input_semantic_map = tmp['input_semantic_map'].permute(0,2,3,1)[not_difference]
        reshaped_occupancy_map = occupancy_map.permute(0,3,4,2,1)[not_difference].reshape(-1,2)
        reshaped_semantic_map = semantic_map.permute(0,2,3,1)[not_difference]
        semantic_consistency_loss =  self.MSE(reshaped_semantic_map,input_semantic_map)
        occupancy_consistency_loss = self.MSE(reshaped_occupancy_map,input_map)
        consistency_loss = self.w4*(occupancy_consistency_loss + semantic_consistency_loss)
        loss = loss1+self.w2*loss2+self.w3*loss3 + consistency_loss

        acc = map_accuracy(occupancy_map.detach(), gt_3d).cpu().numpy().tolist()
        sem_acc = semantic_accuracy(semantic_map.detach(), gt_semantics).cpu().numpy().tolist()

        sem_iou = self.iou(semantic_map.detach().argmax(axis = 1),gt_semantics)
        diff_probs = (pred_difference/pred_difference.sum(axis = 1,keepdims = True))[:,1,:,:].detach()
        difference_f1 = self.f1(diff_probs,difference)
        # if(batch_idx%2 == 1):
        #     tmp = diff_probs.detach().cpu().numpy()
        #     tmp2 = difference.cpu().numpy()
        #     tmp3 = np.concatenate((tmp[0],tmp2[0]),axis = 1)
        #     # plt.imshow(tmp3)
        #     # plt.show()
        #     cv2.imshow('differences',tmp3)
        #     key = cv2.waitKey(100)
        #     if(key&0xFF == ord('d')):
        #         pdb.set_trace()

        
        # print(step_accuracies[0],semantic_accuracies[0],step_accuracies[0].shape,semantic_accuracies[0].shape,type(step_accuracies[0]),type(semantic_accuracies[0]),)
        split = 'train'
        self.log_dict({'{}_loss'.format(split): loss, '{}_occupancy_accuracy'.format(split): acc,
                       '{}_semantic_accuracy'.format(split): sem_acc,
                       '{}_semantic_loss'.format(split): loss1.detach().cpu().numpy().tolist(), 
                       '{}_occupancy_loss'.format(split): loss2.detach().cpu().numpy().tolist(),
                       '{}_differences_loss'.format(split):loss3.detach().cpu().numpy().tolist(),
                       '{}_semantic_iou'.format(split):sem_iou,
                       '{}_difference_f1'.format(split):difference_f1,
                       '{}_consistency_loss'.format(split):consistency_loss.detach().cpu().numpy().tolist()
                       })
        torch.cuda.empty_cache()
        gc.collect()
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            torch.cuda.empty_cache()
            tmp = self.dp.get_model_inputs(batch)
            torch.cuda.empty_cache()

            map_and_push = tmp['map_and_push']
            semantic_map = tmp['semantic_map']
            
            occupancy_map,semantic_map,pred_difference,_ = self.push_predictor(map_and_push,semantic_map)
            
            difference = batch['difference'].to(self.device)
            gt_semantics = batch['post_push_gt_semantics'].to(self.device)
            gt_3d = batch['post_push_gt_3d'].to(self.device)
            if (self.k_type == 'fixed'):
                k = self.get_mixing_coefficient()
            else:
                k = max(1,self.k)
            loss1= focused_evidential_semantic_crossentropy(semantic_map,gt_semantics,difference,focus_factor = self.focus_factor,n_classes = n_classes,k=k)
            loss2 = focused_evidential_occupancy_crossentropy(occupancy_map,gt_3d,difference,focus_factor = self.focus_factor,k=k)
            loss3 = evidential_semantic_crossentropy(pred_difference,difference,k=k)
            input_occupancy = tmp['input_occupancy_map']
            input_beta = input_occupancy[:,::2,:,:]
            input_alpha = input_occupancy[:,1::2,:,:]
            not_difference = torch.logical_not(difference)
            input_map = torch.stack([input_beta,input_alpha],dim = 1).permute(0,3,4,2,1)[not_difference].reshape(-1,2)
            input_semantic_map = tmp['input_semantic_map'].permute(0,2,3,1)[not_difference]
            reshaped_occupancy_map = occupancy_map.permute(0,3,4,2,1)[not_difference].reshape(-1,2)
            reshaped_semantic_map = semantic_map.permute(0,2,3,1)[not_difference]
            semantic_consistency_loss =  self.MSE(reshaped_semantic_map,input_semantic_map)
            occupancy_consistency_loss = self.MSE(reshaped_occupancy_map,input_map)
            consistency_loss = self.w4*(occupancy_consistency_loss + semantic_consistency_loss)
            
            loss = loss1+self.w2*loss2+self.w3*loss3 + consistency_loss



            loss = loss1+self.w2*loss2+self.w3*loss3 + consistency_loss

            acc = map_accuracy(occupancy_map, gt_3d).cpu().numpy().tolist()
            sem_acc = semantic_accuracy(semantic_map, gt_semantics).cpu().numpy().tolist()
            diff_acc = map_accuracy(pred_difference,difference).cpu().numpy().tolist()

            sem_iou = self.iou(semantic_map.argmax(axis = 1),gt_semantics)
            diff_probs = (pred_difference/pred_difference.sum(axis = 1,keepdims = True))[:,1,:,:]
            difference_f1 = self.f1(diff_probs,difference)

            # print(step_accuracies[0],semantic_accuracies[0],step_accuracies[0].shape,semantic_accuracies[0].shape,type(step_accuracies[0]),type(semantic_accuracies[0]),)
            split = 'val'
            self.log_dict({'{}_loss'.format(split): loss, '{}_occupancy_accuracy'.format(split): acc,
                        '{}_semantic_accuracy'.format(split): sem_acc,
                        '{}_semantic_loss'.format(split): loss1, '{}_occupancy_loss'.format(split): loss2,
                        '{}_differences_loss'.format(split):loss3,
                        '{}_semantic_iou'.format(split):sem_iou,
                        '{}_difference_f1'.format(split):difference_f1,
                        '{}_consistency_loss'.format(split):consistency_loss.cpu().numpy().tolist()
                        })
            torch.cuda.empty_cache()
            gc.collect()
        return loss

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            torch.cuda.empty_cache()
            tmp = self.dp.get_model_inputs(batch)
            torch.cuda.empty_cache()

            map_and_push = tmp['map_and_push']
            semantic_map = tmp['semantic_map']
            
            occupancy_map,semantic_map,pred_difference,_ = self.push_predictor(map_and_push,semantic_map)
            
            difference = batch['difference'].to(self.device)
            gt_semantics = batch['post_push_gt_semantics'].to(self.device)
            gt_3d = batch['post_push_gt_3d'].to(self.device)

            loss1= focused_evidential_semantic_crossentropy(semantic_map,gt_semantics,difference,focus_factor = self.focus_factor,n_classes = self.n_classes)
            loss2 = focused_evidential_occupancy_crossentropy(occupancy_map,gt_3d,difference,focus_factor = self.focus_factor)
            loss3 = evidential_semantic_crossentropy(pred_difference,difference)

            input_occupancy = tmp['input_occupancy_map']
            input_beta = input_occupancy[:,::2,:,:]
            input_alpha = input_occupancy[:,1::2,:,:]
            not_difference = torch.logical_not(difference)
            input_map = torch.stack([input_beta,input_alpha],dim = 1).permute(0,3,4,2,1)[not_difference].reshape(-1,2)
            input_semantic_map = tmp['input_semantic_map'].permute(0,2,3,1)[not_difference]
            reshaped_occupancy_map = occupancy_map.permute(0,3,4,2,1)[not_difference].reshape(-1,2)
            reshaped_semantic_map = semantic_map.permute(0,2,3,1)[not_difference]
            semantic_consistency_loss =  self.MSE(reshaped_semantic_map,input_semantic_map)
            occupancy_consistency_loss = self.MSE(reshaped_occupancy_map,input_map)
            consistency_loss = self.w4*(occupancy_consistency_loss + semantic_consistency_loss)
            loss = loss1+self.w2*loss2+self.w3*loss3 + consistency_loss

            acc = map_accuracy(occupancy_map, gt_3d).cpu().numpy().tolist()
            sem_acc = semantic_accuracy(semantic_map, gt_semantics).cpu().numpy().tolist()
            sem_iou = self.iou(semantic_map.argmax(axis = 1),gt_semantics)
            diff_probs = (pred_difference/pred_difference.sum(axis = 1,keepdims = True))[:,1,:,:]
            difference_f1 = self.f1(diff_probs,difference)
            # print(step_accuracies[0],semantic_accuracies[0],step_accuracies[0].shape,semantic_accuracies[0].shape,type(step_accuracies[0]),type(semantic_accuracies[0]),)
            split = 'test'
            self.log_dict({'{}_loss'.format(split): loss, '{}_occupancy_accuracy'.format(split): acc,
                        '{}_semantic_accuracy'.format(split): sem_acc,
                        '{}_semantic_loss'.format(split): loss1, '{}_occupancy_loss'.format(split): loss2,
                        '{}_differences_loss'.format(split):loss3,
                        '{}_semantic_iou'.format(split):sem_iou,
                        '{}_difference_f1'.format(split):difference_f1,
                        '{}_consistency_loss'.format(split):consistency_loss.cpu().numpy().tolist()
                        })
            torch.cuda.empty_cache()
            gc.collect()
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.push_predictor.parameters(), lr=self.lr)

    def get_outputs(self, batch,intermediates = False,max_obs = None,normalize = False):
        self.push_predictor.eval()
        with torch.no_grad():
            torch.cuda.empty_cache()
            tmp = self.dp.get_model_inputs(batch,max_obs = max_obs,all_steps = intermediates)
            torch.cuda.empty_cache()
            # pdb.set_trace()

            map_and_push = tmp['map_and_push']
            semantic_map = tmp['semantic_map']
            if(not intermediates):
                occupancy_map,semantic_map,pred_difference,_ = self.push_predictor(map_and_push,semantic_map)
            else:
                orig_shape = map_and_push.shape[:2]
                semantic_map_input = torch.clone(semantic_map).cpu().numpy()
                occupancy_map,semantic_map,pred_difference,_ = self.push_predictor(map_and_push.flatten(0,1),semantic_map.flatten(0,1))
                if(normalize):
                    # pdb.set_trace()
                    occupancy_map = occupancy_map/occupancy_map.sum(axis = 1,keepdims = True)
                    semantic_map = semantic_map/semantic_map.sum(axis =1,keepdims = True)
                    pred_difference = pred_difference/pred_difference.sum(axis = 1,keepdims = True)
                occupancy_map = to_channels_last(occupancy_map)
                semantic_map = to_channels_last(semantic_map)
                pred_difference = to_channels_last(pred_difference)
                occupancy_map = torch.unflatten(occupancy_map,0,orig_shape).cpu().numpy()
                semantic_map = torch.unflatten(semantic_map,0,orig_shape).cpu().numpy()
                pred_difference = torch.unflatten(pred_difference,0,orig_shape).cpu().numpy()
                torch.cuda.empty_cache()

            difference = batch['difference'].to(self.device)
            gt_semantics = batch['post_push_gt_semantics'].to(self.device)
            gt_3d = batch['post_push_gt_3d'].to(self.device)
            torch.cuda.empty_cache()
            gc.collect()
        self.push_predictor.train()
        return {'occupancy_map':occupancy_map,'semantic_map':semantic_map,
                'pred_difference':pred_difference,'difference_gt':difference,
                'gt_semantics':gt_semantics,'occupancy_gt':gt_3d,
                'map_and_push':map_and_push.cpu().numpy(),'semantic_map_input':semantic_map_input,'pre_push_semantic_gt':batch['gt_semantics']}


if (__name__ == '__main__'):
    multiprocessing.set_start_method('spawn')



    parser = argparse.ArgumentParser()
    parser.add_argument('--k_type',default = 'variable')
    parser.add_argument('--k', type=float, default=1.0,
                        help="""max (or actual) mixing coefficient between loss and regularization""")
    parser.add_argument('--max_alpha', type=float, default=200.0,
                        help="""Max value for the normalization of the dirichlet (beta) distributions as inputs""")
    parser.add_argument('--normalize',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--checkpoint',default = None)
    parser.add_argument('--lr', type=float, default=0.0005,
                        help="""Learning Rate""")
    parser.add_argument('--focus_factor', type=float, default=1.0,
                        help="""Loss focus factor""")
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    # torch._dynamo.config.guard_nn_modules = True
    # np.random.seed(42)
    generator = torch.Generator().manual_seed(42)
    skip = 1
    # dataset_dir = '/home/motion/pybullet_shelf_gym/shelf_gym/tasks/hallucination/data/map_completion/'
    # dataset_dir = 'push_prediction_ycb.hdf5'
    # dataset_dir = '/tmp/push_prediction_ycb.hdf5'
    # dataset_dir = 'push_completion_ycb_new_depth.hdf5'
    dataset_dir = 'unbiased_push_dataset.hdf5'
    # dataset_dir = '/scratch/bdds/jcorreiamarques/pybullet_shelf_gym/shelf_gym/tasks/hallucination/model_training/push_prediction.hdf5'
    dataset = PushDatasetH5pyNPY(dataset_dir, max_samples=10,camera_params_dir='../evaluation/camera_matrices.npz')    # val_set = MapDatasetH5py(val_dir,max_samples = 10,skip = skip)
    # pdb.set_trace()
    # train_set,val_set,test_set = torch.utils.data.random_split(dataset,[0.7,0.2,0.1],generator=generator)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1], generator=generator)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True,
                              num_workers=4, prefetch_factor=1, pin_memory=True, pin_memory_device='cuda',
                              persistent_workers=True, worker_init_fn=PushDatasetH5pyNPY.worker_init_fn)
    val_loader = DataLoader(val_set, batch_size=10, shuffle=False,
                            num_workers=4, prefetch_factor=1, pin_memory=True, pin_memory_device='cuda',
                            persistent_workers=True, worker_init_fn=PushDatasetH5pyNPY.worker_init_fn)
    # test_loader = DataLoader(test_set,batch_size = 30,shuffle = False,num_workers= 5,prefetch_factor = 1,pin_memory= True, pin_memory_device = 'cuda',persistent_workers = True)


    k_type = args.k_type
    k = args.k
    normalize = args.normalize
    resume = args.resume
    checkpoint_to_resume = args.checkpoint
    max_alpha = args.max_alpha
    lr = args.lr
    focus_factor = args.focus_factor

    k_type = 'variable'
    n_classes = 15
    max_epochs = 1000
    total_steps = 1000
    if(not resume):
        val_check_interval = 0.33
    else:
        val_check_interval = 0.25
    # map_predictor_checkpoint = '/home/jmc/pybullet_shelf_gym/shelf_gym/tasks/hallucination/model_training/artifacts/model-ma6zvwu4:v9/model.ckpt'
    # map_predictor_name = 'model-hfxhxvla:v52.ckpt'
    # map_predictor_name = 'model-5dburcae:v4.ckpt'
    map_predictor_name = 'model-9c3i82rt:v4.ckpt'
    map_predictor_dir = './artifacts/models/'


    # wandb_logger = None
    if(not resume):
        name = 'Raycasting Depth Push Prediction_FF_{}_lr_{}_normalize_{}_max_alpha_{}'.format(focus_factor,lr,normalize,max_alpha).replace('.','_')
    else:
        name = 'Resume_{}_LR_{}_salt_noise_{}'.format(checkpoint_to_resume.split('/')[-1],focus_factor,lr).replace('.','_')
    wandb_logger = WandbLogger(name,project = 'YCB Push Prediction',log_model ='all')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    underlying_model = PushSemanticUNet(308, 204, n_semantic_channel_in=n_classes,
                                        n_classes = n_classes, do_dropout=False,
                                        normalize = normalize).to('cuda')


    if(not resume):
        model = PushPredictor(push_predictor=underlying_model,map_predictor_dir= map_predictor_dir,
                          map_predictor_name = map_predictor_name, lr=lr, k_type=k_type, k=k,
                            total_steps=total_steps,focus_factor = focus_factor,normalize = normalize,
                            n_classes = n_classes,max_alpha = max_alpha)
    else:
        run = wandb.init(project='eval')

        
        artifact = run.use_artifact(checkpoint_to_resume, type='model')

        checkpoint = artifact.download()+'/model.ckpt'
        run.finish(quiet = True)

        model = PushPredictor.load_from_checkpoint(checkpoint,map_predictor_dir = './artifacts/models/',map_predictor_name =map_predictor_name,lr = lr)


    # compiled_model = torch.compile(model,mode='reduce-overhead')
    # print('\n\ncompiled model!\n\n')
    # prof = SimpleProfiler(dirpath= './debugging_logs/',filename = 'debug_log2.txt')
    prof = None
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=20, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", every_n_train_steps = int(val_check_interval*len(train_loader)+1))
    trainer = L.Trainer(precision="32", enable_progress_bar=True,
                        logger=wandb_logger, max_epochs=max_epochs,
                        callbacks=[early_stop_callback, checkpoint_callback],
                        log_every_n_steps=5, val_check_interval=val_check_interval,
                        num_sanity_val_steps=0, use_distributed_sampler=False)
    # trainer.validate(model=model,dataloaders = val_loader)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)



