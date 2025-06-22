import numpy as np
import torch
import torch.nn.functional as F 
from matplotlib import pyplot as plt
import pdb
import sys
l2_loss = torch.nn.MSELoss(reduction = 'none')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DataPrepper:
    def __init__(self,device,n_classes,mode,normalize,max_alpha = 50):
        self.device = device
        self.n_classes = n_classes
        self.mode = mode
        self.normalize = normalize
        self.max_alpha = max_alpha
        self.a = 2/(self.max_alpha-1)
        self.b = -1-self.a
    def data_prep(self,data):
        permuted_occupied = data['occupied'].permute(1,0,2,3,4).to(self.device)
        permuted_free = data['free'].permute(1,0,2,3,4).to(self.device)
        gt_3d = data['gt_3d'].to(self.device)
        permuted_semantics = data['semantic_hms'].permute(1,0,2,3).to(self.device)
        permuted_semantics =  torch.nn.functional.one_hot(permuted_semantics.to(torch.int64),num_classes = self.n_classes+1).permute(0,1,4,2,3)
        gt_semantics = data['gt_semantics'].to(self.device)
        return gt_3d,permuted_free,permuted_occupied,permuted_semantics,gt_semantics

    def get_model_input(self,occupied,free,previous_map,previous_semantic_map,semantics):
    # free_map 
        if (self.mode in ['3D_all_inputs','3D_semantic_augmented_all_inputs']):
            if(self.normalize):
                # we transform the input to be between -1 and 1 without destroying information
                previous_map = torch.clip(previous_map,1+0.0001,self.max_alpha)
                previous_semantic_map = torch.clip(previous_semantic_map,1+0.0001,self.max_alpha)
                previous_map = self.a*previous_map + self.b
                previous_semantic_map = self.a*previous_semantic_map + self.b
                occupied = -1 + 2*occupied
                free = -1 + 2*free
                # total = previous_map[:,1::2,:,:] + previous_map[:,::2,:,:]
                # previous_map[:,1::2,:,:] = previous_map[:,1::2,:,:]/total
                # previous_map[:,::2,:,:] = previous_map[:,::2,:,:]/total
                # previous_semantic_map = previous_semantic_map/previous_semantic_map.sum(axis =1, keepdims = True)
            # import pdb
            # pdb.set_trace()
            previous_semantic_map = previous_semantic_map.to("cuda")
            previous_map = previous_map.to("cuda")
            previous_semantic_map = torch.cat((previous_semantic_map, semantics),dim = 1)

            if(self.mode == '3D_semantic_augmented_all_inputs'):
                model_input = torch.cat((previous_map,occupied,free,previous_semantic_map),dim = 1)

            else:
                model_input = torch.cat((previous_map,occupied,free),dim = 1)

        elif((self.mode == '3D_denoising') or (self.mode=='3D_augmented_denoising') or (self.mode=='3D_semantic_augmented_denoising')):
            model_input = previous_map
            model_input[:,1::2,:,:] += occupied
            model_input[:,::2,:,:] += free
            observed = torch.any(torch.logical_or(occupied,free),axis = 1).unsqueeze(dim=1)
            semantics=semantics[:,:-1,:,:]
            previous_semantic_map[observed.expand(previous_semantic_map.shape)] += semantics[observed.expand(previous_semantic_map.shape)]
            if(self.normalize):
                # we transform the input to be between -1 and 1 without destroying information
                model_input = torch.clip(model_input,1,self.max_alpha)
                previous_semantic_map = torch.clip(previous_semantic_map,1,self.max_alpha)
                model_input = self.a*model_input + self.b
                previous_semantic_map = self.a*previous_semantic_map + self.b
                observed = -1 + 2*observed
            if(self.mode == '3D_augmented_denoising'):
                model_input = torch.cat((model_input,observed),dim = 1)
                # previous_semantic_map = torch.cat((previous_semantic_map,observed),dim = 1)
            if(self.mode == '3D_semantic_augmented_denoising'):
                model_input = torch.cat((model_input,observed,previous_semantic_map),dim = 1)
                # previous_semantic_map = torch.cat((previous_semantic_map,observed),dim = 1)



        return model_input.float(),previous_semantic_map.float()

    def get_initial_map(self,permuted_free):
        previous_map = torch.ones((permuted_free.shape[1],204,120,200)).to(self.device)
        previous_semantics = torch.ones((permuted_free.shape[1],self.n_classes,permuted_free.shape[3],permuted_free.shape[4])).to(self.device)
        return previous_map.float(),previous_semantics.float()


class NonEvidentialDataPrepper:
    def __init__(self,device,n_classes,mode,normalize):
        self.device = device
        self.n_classes = n_classes
        self.mode = mode
        self.normalize = normalize

    def data_prep(self,data):
        permuted_occupied = data['occupied'].permute(1,0,2,3,4).to(self.device)
        permuted_free = data['free'].permute(1,0,2,3,4).to(self.device)
        gt_3d = data['gt_3d'].to(self.device)
        permuted_semantics = data['semantic_hms'].permute(1,0,2,3).to(self.device)
        permuted_semantics =  torch.nn.functional.one_hot(permuted_semantics.to(torch.int64),num_classes = self.n_classes+1).permute(0,1,4,2,3)
        gt_semantics = data['gt_semantics'].to(self.device)
        return gt_3d,permuted_free,permuted_occupied,permuted_semantics,gt_semantics

    def get_model_input(self,occupied,free,previous_map,previous_semantic_map,semantics):
    # free_map 
        if (self.mode in ['3D_all_inputs','3D_semantic_augmented_all_inputs']):
            if(self.normalize):
                # we transform the input to be between -1 and 1 without destroying information
                previous_map = torch.clip(previous_map,1,self.max_alpha)
                previous_semantic_map = torch.clip(previous_semantic_map,1,self.max_alpha)
                free = -1+2*free
                occupied = -1+2*occupied
                # total = previous_map[:,1::2,:,:] + previous_map[:,::2,:,:]
                # previous_map[:,1::2,:,:] = previous_map[:,1::2,:,:]/total
                # previous_map[:,::2,:,:] = previous_map[:,::2,:,:]/total
                # previous_semantic_map = previous_semantic_map/previous_semantic_map.sum(axis =1, keepdims = True)
            previous_semantic_map = torch.cat((previous_semantic_map,semantics),dim = 1)

            if(self.mode == '3D_semantic_augmented_all_inputs'):
                model_input = torch.cat((previous_map,occupied,free,previous_semantic_map),dim = 1)

            else:
                model_input = torch.cat((previous_map,occupied,free),dim = 1)

        elif((self.mode == '3D_denoising') or (self.mode=='3D_augmented_denoising') or (self.mode=='3D_semantic_augmented_denoising')):
            model_input = previous_map
            neg_input = 1-previous_map
            model_input[occupied] *=0.8
            neg_input[occupied] *= 0.2
            neg_input[free]*=0.8 
            model_input[free] *= 0.2
            model_input = model_input/(neg_input+model_input)
            observed = torch.any(torch.logical_or(occupied,free),axis = 1).unsqueeze(dim=1)
            semantics = semantics[:,:-1,:,:]
            previous_semantic_map[observed.expand(previous_semantic_map.shape)] += semantics[observed.expand(previous_semantic_map.shape)]
            previous_semantic_map = previous_semantic_map/previous_semantic_map.sum(axis = 1,keepdims = True)
            if(self.normalize):
                # we transform the input to be between -1 and 1 without destroying information
                model_input = torch.clip(model_input,1,self.max_alpha)
                previous_semantic_map = torch.clip(previous_semantic_map,1,self.max_alpha)
                model_input = self.a*model_input + self.b
                previous_semantic_map = self.a*previous_semantic_map + self.b
                observed = -1 + 2*observed
            if(self.mode == '3D_augmented_denoising'):
                model_input = torch.cat((model_input,observed),dim = 1)
                # previous_semantic_map = torch.cat((previous_semantic_map,observed),dim = 1)
            if(self.mode == '3D_semantic_augmented_denoising'):
                model_input = torch.cat((model_input,observed,previous_semantic_map),dim = 1)
                # previous_semantic_map = torch.cat((previous_semantic_map,observed),dim = 1)



        return model_input.float(),previous_semantic_map.float()

    def get_initial_map(self,permuted_free):
        previous_map = torch.zeros((permuted_free.shape[1],102,120,200)).to(self.device)
        previous_semantics = torch.ones((permuted_free.shape[1],self.n_classes,permuted_free.shape[3],permuted_free.shape[4])).to(self.device)
        return previous_map.float(),previous_semantics.float()

class PushDataPrepper:
    def __init__(self,mapper_checkpoint,max_obs = 10,max_alpha = 50):
        from shelf_gym.scripts.model_training.occupancy_lightning_training import SemanticMapPredictor
        import shelf_gym.utils.models.UNet
        sys.modules['UNet'] = shelf_gym.utils.models.UNet
        self.mapper_checkpoint = mapper_checkpoint
        self.max_obs = max_obs
        self.max_alpha = max_alpha
        self.a = 2/(self.max_alpha-1)
        self.b = -1-self.a
        print(max_alpha)
        self.map_completion_model = SemanticMapPredictor.load_from_checkpoint(self.mapper_checkpoint).eval()
        del sys.modules['UNet']

        self.rng = np.random.default_rng(seed = 42)

    def get_model_inputs(self,batch,max_obs = None,all_steps = False):
        if(max_obs is None):
            max_obs_before_manipulation = self.rng.choice(np.arange(self.max_obs))
            max_obs_before_manipulation = max(1,max_obs_before_manipulation)
        else:
            max_obs_before_manipulation = max_obs


        with torch.no_grad():
            outputs = self.map_completion_model.get_outputs(batch,intermediates = all_steps,
                                                            max_obs = max_obs_before_manipulation,normalize = False)

            if(not all_steps):

                op = outputs['occupancy_probs'][0]
                sp = outputs['semantic_probs'][0]            
                sp = sp.permute(0,3,1,2)

            else:
                op = torch.from_numpy(np.stack(outputs['occupancy_probs'])).to('cuda')
                sp = torch.from_numpy(np.stack(outputs['semantic_probs'])).to('cuda')
                sp = sp.permute(0,1,4,2,3)
            sv = batch['swept_volume'].to('cuda')
            pp = batch['push_parametrization'].to('cuda')
            # pdb.set_trace()
            torch.cuda.empty_cache()
        return self.add_push_features(op,sp,sv,pp,all_steps,outputs)

    def add_push_features(self,op,sp,sv,pp,all_steps,outputs):
            # pdb.set_trace()

            op = op*self.a + self.b
            sp = sp*self.a + self.b
            if(not all_steps):
                map_and_push = torch.concatenate([op[:,:,:,:,0],op[:,:,:,:,1],sv],dim = 1)
                original_shape = list(map_and_push.shape)


            else:
                map_and_push = torch.concatenate([op[:,:,:,:,:,0],op[:,:,:,:,:,1],sv.unsqueeze(0).expand(op.shape[0],-1,-1,-1,-1)],dim = 2)
                original_shape = list(map_and_push.shape)[1:]

            original_shape[1] = 2
            push_parametrization = torch.zeros(original_shape).to('cuda')
            x1 = pp[:,0].long()
            y1 = 200-pp[:,1].long()
            x2 = pp[:,2].long()
            y2 = 200-pp[:,3].long()
            x1 = torch.clamp(x1,0,119)
            x2 = torch.clamp(x2,0,119)
            y1 = torch.clamp(y1,0,199)
            y2 = torch.clamp(y2,0,199)
            # push_parametrization[]
            # print(x1,x2,y1,y2)
            push_parametrization[np.arange(x1.shape[0]),0,x1,y1] = 1
            push_parametrization[np.arange(x2.shape[0]),1,x2,y2] = 1

            if(not all_steps):
                map_and_push = torch.concatenate([map_and_push,push_parametrization],dim = 1)
            else:
                map_and_push = torch.concatenate([map_and_push,push_parametrization.unsqueeze(0).expand(op.shape[0],-1,-1,-1,-1)],dim = 2)

            return {'map_and_push':map_and_push,'semantic_map':sp,'input_occupancy_map':outputs['occupancy_map'],'input_semantic_map':outputs['semantic_map']}
        
    
def to_channels_last(tmp):
    tmp = torch.movedim(tmp,1,-1)
    return tmp