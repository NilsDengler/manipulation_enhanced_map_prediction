import cv2
import numpy as np
import torch
import cupy as cp
import seaborn as sns
import open3d as o3d
import os 
from shelf_gym.utils.scaling_utils import scale_semantic_probs

def get_my_cmap(n_classes = 7):
    my_cmap = (255*np.array(sns.color_palette("husl",n_classes))).astype(np.uint8)
    my_cmap[-1,:] = 0
    return my_cmap

def put_text(img,text):
    img = cv2.resize(img,(0,0),fx = 2,fy =2,interpolation = cv2.INTER_LINEAR)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 0.8
    color = (255,255,255) 
    thickness = 1
    # get boundary of this text
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    # get coords based on boundary
    textX = (img.shape[1] - textsize[0]) // 2
    return cv2.putText(img,text,(textX, 30), font,fontScale, color, thickness, cv2.LINE_AA)


def create_pcd_from_voxel_grid(vg,thold = 0.7):
    pcd = o3d.geometry.PointCloud()
    point_coords = np.where(vg>thold)
    point_coords = np.stack(point_coords).transpose()
    pcd.points = o3d.utility.Vector3dVector(point_coords)
    return pcd


def disentangle_and_preprocess_map_and_push(mnp):
    occupancy_input_beta = mnp[:,:,:102,:,:]
    occupancy_input_alpha = mnp[:,:,102:204,:,:]
    occupancy_input_prob = occupancy_input_alpha/(occupancy_input_alpha+occupancy_input_beta)
    occupancy_input = occupancy_input_prob[:,:,10:,:,:].max(axis = 2)
    swept_volume = mnp[:,:,-104:-2,:,:].sum(axis = 2)
    return occupancy_input,swept_volume


def get_result_videos_from_batch(model,batch,video_dir,n_classes = 15):
    my_cmap = get_my_cmap(n_classes)
    outputs = model.get_outputs(batch,normalize = True,intermediates = True)
    semantic_gt = outputs['semantic_gt']
    occupancy_probs = outputs['occupancy_probs']

    semantic_probs = outputs['semantic_probs']
    semantic_probs = scale_semantic_probs(semantic_probs)
    fused_occupancy = []
    fused_semantics = []
    gt_3d,permuted_free,permuted_occupied,permuted_semantics,gt_semantics = model.dp.data_prep(batch)
    previous_map,previous_semantic_map = model.dp.get_initial_map(permuted_free)
    previous_semantic_map[:,-1,:,:]+=0.01
    original_model_mode = model.dp.mode
    original_model_normalize = model.dp.normalize
    for occupied,free,semantics in zip(permuted_occupied,permuted_free,permuted_semantics):
        model.dp.mode = '3D_denoising'
        model.dp.normalize = False
        previous_map,previous_semantic_map = model.dp.get_model_input(occupied,free,previous_map,previous_semantic_map,semantics)            
        fused_semantics.append(previous_semantic_map.cpu().numpy())
        alpha = previous_map[:,::2,:,:]
        beta = previous_map[:,1::2,:,:]
        fused_prob = beta/(alpha+beta)
        prob_2d = fused_prob[:,10:,:,:].cpu().numpy().max(axis = 1)
        fused_occupancy.append(prob_2d)
    model.dp.mode = original_model_mode
    model.dp.normalize = original_model_normalize
    num_batches,_,height,width,_ = occupancy_probs[0].shape
    os.makedirs(video_dir,exist_ok = True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    for datapoint in range(num_batches):
        summary_video = cv2.VideoWriter(video_dir+'/run_summary_{}.mp4'.format(datapoint),fourcc,1,(3*2*width,2*2*height))
        for step in range(10):
            this_occ = (255*occupancy_probs[step][datapoint,15:,:,:,1].max(axis = 0)).astype(np.uint8)
            this_occ = np.repeat(this_occ[:,:,np.newaxis],repeats = 3,axis =2)
            this_occ = put_text(this_occ,'Occupancy Prediction')
            this_semantic = my_cmap[semantic_probs[step][datapoint].argmax(axis = 2)]
            this_semantic = put_text(this_semantic,'Semantic Prediction')
            this_semantic_conf = (255*semantic_probs[step][datapoint].max(axis = 2)).astype(np.uint8)
            this_semantic_conf = np.repeat(this_semantic_conf[:,:,np.newaxis],repeats = 3,axis =2)
            this_semantic_conf = put_text(this_semantic_conf,'Semantic Confidence')
            this_semantic_fused = put_text(my_cmap[fused_semantics[step][datapoint].argmax(axis = 0)],'Naive Semantic Map')
            this_occupancy_fused = (255*np.repeat(fused_occupancy[step][datapoint][:,:,np.newaxis],repeats = 3,axis =2)).astype(np.uint8)
            this_occupancy_fused = put_text(this_occupancy_fused,'Naive Occupancy Map')
        #     this_semantic_summary = np.concatenate((this_semantic,this_semantic_conf),axis = 0)
            gt = my_cmap[cp.asnumpy(semantic_gt[datapoint])]
            gt = put_text(gt,'Ground Truth')
            upper = np.concatenate([gt,this_semantic_fused,this_occupancy_fused],axis = 1)
            lower = np.concatenate([this_semantic,this_semantic_conf,this_occ],axis = 1)
            full_summary = np.concatenate((upper,lower),axis = 0)
            summary_video.write(full_summary)
        summary_video.release()

def get_push_result_videos_from_batch(model,batch,video_dir,offset = 0,n_classes = 15):
    my_cmap = get_my_cmap(n_classes)
    outputs = model.get_outputs(batch,intermediates = True,max_obs = 10,normalize = True)

    post_push_semantic_gt = my_cmap[outputs['gt_semantics'].cpu().numpy()]
    pre_push_semantic_gt = my_cmap[outputs['pre_push_semantic_gt'].cpu().numpy()]
    occupancy_probs = outputs['occupancy_map']
    occupancy_probs = occupancy_probs/occupancy_probs.sum(axis = -1,keepdims = True)
    occupancy_probs_2d = (255*occupancy_probs[:,:,10,:,:,1]).astype(np.uint8)

    semantic_probs = outputs['semantic_map']
    semantic_probs = semantic_probs/semantic_probs.sum(axis = -1,keepdims = True)
    semantic_color = my_cmap[semantic_probs.argmax(axis =-1)]
    semantic_conf = semantic_probs.max(axis = -1)
    semantic_color_conf = (semantic_color.astype(np.float32)*(semantic_conf[:,:,:,:,np.newaxis])+(1-semantic_conf[:,:,:,:,np.newaxis])*255).astype(np.uint8)
    map_and_push = outputs['map_and_push']
    map_and_push[:,:,:204] = (map_and_push[:,:,:204]-model.dp.b)/model.dp.a

    occupancy_input_2d,swept_volume = disentangle_and_preprocess_map_and_push(map_and_push)
    swept_volume = (255*(swept_volume/swept_volume.max())).astype(np.uint8)

    occupancy_input_2d = (occupancy_input_2d*255).astype(np.uint8)
    semantic_map_input = outputs['semantic_map_input']
    semantic_map_input = (semantic_map_input-model.dp.b)/model.dp.a
    semantic_map_input_probs = semantic_map_input/semantic_map_input.sum(axis = 2,keepdims = True)
    semantic_map_input_color = my_cmap[semantic_map_input_probs.argmax(axis = 2)]
    semantic_map_input_conf = semantic_map_input_probs.max(axis = 2)
    semantic_map_input_color = (semantic_map_input_color.astype(np.float32)*semantic_map_input_conf[:,:,:,:,np.newaxis]+(1-semantic_map_input_conf[:,:,:,:,np.newaxis]*255)).astype(np.uint8)
    pred_difference = outputs['pred_difference']
    pred_difference = (255*(pred_difference/pred_difference.sum(axis = -1,keepdims = True))[:,:,:,:,1]).astype(np.uint8)
    difference_gt = 255*(outputs['difference_gt'].cpu().numpy().astype(np.uint8))
    num_batches,_,height,width,_ = occupancy_probs[0].shape
    os.makedirs(video_dir,exist_ok = True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    for datapoint in range(num_batches):
        summary_video = cv2.VideoWriter(video_dir+'/run_summary_{}.mp4'.format(datapoint+offset),fourcc,1,(3*2*width,3*2*height))
        for step in range(10):
            this_occ_pred =  put_text(occupancy_probs_2d[step,datapoint],'Occupancy Pred')
            this_semantic_pred = put_text(semantic_color_conf[step,datapoint],'Semantic Pred')
            this_pre_semantic_gt = put_text(pre_push_semantic_gt[datapoint],'Pre Push GT')
            this_post_semantic_gt = put_text(post_push_semantic_gt[datapoint],'Post Push GT')
            this_semantic_map_input = put_text(semantic_map_input_color[step,datapoint],'Semantic Input')
            this_occupancy_map_input = put_text(occupancy_input_2d[step,datapoint],'Occupancy Input')
            this_difference_gt = put_text(difference_gt[datapoint],'Difference GT')
            this_difference_pred = put_text(pred_difference[step,datapoint],'Predicted Difference')
            this_swept_volume = put_text(swept_volume[step,datapoint],'Swept Volume')
            first_row = np.concatenate([this_pre_semantic_gt,this_post_semantic_gt,
                                np.repeat(this_difference_gt[:,:,np.newaxis],repeats= 3,axis =2)],axis = 1)
            mid_row = np.concatenate([np.repeat(this_occupancy_map_input[:,:,np.newaxis],repeats =3,axis =2),
                                    this_semantic_map_input,
                                    np.repeat(this_swept_volume[:,:,np.newaxis],repeats = 3,axis =2)],axis =1)
            last_row = np.concatenate([np.repeat(this_occ_pred[:,:,np.newaxis],repeats = 3,axis =2),
                                    this_semantic_pred,
                                    np.repeat(this_difference_pred[:,:,np.newaxis],repeats =3,axis =2)],axis =1)
            frame = np.concatenate([first_row,mid_row,last_row],axis = 0)
            summary_video.write(frame)
        summary_video.release()



def get_non_evidential_result_videos_from_batch(model,batch,video_dir,n_classes = 15):
    my_cmap = get_my_cmap(n_classes)
    outputs = model.get_outputs(batch,normalize = True,intermediates = True)
    semantic_gt = outputs['semantic_gt']
    occupancy_probs = outputs['occupancy_probs']

    semantic_probs = outputs['semantic_probs']
    semantic_probs = scale_semantic_probs(semantic_probs)
    fused_occupancy = []
    fused_semantics = []
    gt_3d,permuted_free,permuted_occupied,permuted_semantics,gt_semantics = model.dp.data_prep(batch)
    previous_map,previous_semantic_map = model.dp.get_initial_map(permuted_free)
    previous_map = model.predictor.sigmoid(previous_map)
    previous_semantic_map = model.softmax(previous_semantic_map)
    previous_semantic_map[:,-1,:,:]+=0.01
    original_model_mode = model.dp.mode
    original_model_normalize = model.dp.normalize
    for occupied,free,semantics in zip(permuted_occupied,permuted_free,permuted_semantics):
        model.dp.mode = '3D_denoising'
        model.dp.normalize = False
        previous_map,previous_semantic_map = model.dp.get_model_input(occupied,free,previous_map,previous_semantic_map,semantics)            
        fused_semantics.append(previous_semantic_map.cpu().numpy())

        fused_prob = previous_map
        prob_2d = fused_prob[:,10:,:,:].cpu().numpy().max(axis = 1)

        fused_occupancy.append(prob_2d)
    model.dp.mode = original_model_mode
    model.dp.normalize = original_model_normalize
    num_batches,_,height,width = occupancy_probs[0].shape
    os.makedirs(video_dir,exist_ok = True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    for datapoint in range(num_batches):
        summary_video = cv2.VideoWriter(video_dir+'/run_summary_{}.mp4'.format(datapoint),fourcc,1,(3*2*width,2*2*height))
        for step in range(10):
            this_occ = (255*occupancy_probs[step][datapoint,15:,:,:].max(axis = 0)).astype(np.uint8)
            this_occ = np.repeat(this_occ[:,:,np.newaxis],repeats = 3,axis =2)
            this_occ = put_text(this_occ,'Occupancy Prediction')
            this_semantic = my_cmap[semantic_probs[step][datapoint].argmax(axis = 2)]
            this_semantic = put_text(this_semantic,'Semantic Prediction')
            this_semantic_conf = (255*semantic_probs[step][datapoint].max(axis = 2)).astype(np.uint8)
            this_semantic_conf = np.repeat(this_semantic_conf[:,:,np.newaxis],repeats = 3,axis =2)
            this_semantic_conf = put_text(this_semantic_conf,'Semantic Confidence')
            this_semantic_fused = put_text(my_cmap[fused_semantics[step][datapoint].argmax(axis = 0)],'Naive Semantic Map')
            this_occupancy_fused = (255*np.repeat(fused_occupancy[step][datapoint][:,:,np.newaxis],repeats = 3,axis =2)).astype(np.uint8)
            this_occupancy_fused = put_text(this_occupancy_fused,'Naive Occupancy Map')
            gt = my_cmap[cp.asnumpy(semantic_gt[datapoint])]
            gt = put_text(gt,'Ground Truth')
            upper = np.concatenate([gt,this_semantic_fused,this_occupancy_fused],axis = 1)
            lower = np.concatenate([this_semantic,this_semantic_conf,this_occ],axis = 1)
            full_summary = np.concatenate((upper,lower),axis = 0)

            summary_video.write(full_summary)
        summary_video.release()


def add_push_to_image(scene,path,path_annotation,original_image):
    mp = scene.linear_interpolate_motion_klampt_joint_traj(path,traj_annotation = path_annotation,imagined = True,verbose= False)

    wp = np.array(mp)

    swept_volume = np.zeros((140,200,102))

    relevant_wps = wp[:,-3:].reshape(-1,3)

    points = np.asarray(scene.mapping.hg.world_point_to_map_point(relevant_wps).astype(int).reshape(-1,3,3))
    points[:,:,:] = points[:,:,[1,0,2]]

    map_size = np.asarray(swept_volume.shape[:2])
    min_index = points.reshape(-1,3).min(axis = 0)[:2]
    max_index = points.reshape(-1,3).max(axis = 0)[:2]
    min_index = np.minimum(np.array([0,0]),min_index)
    max_index = np.maximum(map_size,max_index)
    image_shape = max_index-min_index
    image = np.zeros(image_shape.tolist()+[3],dtype = np.uint8)
    print(image.shape)

    # we overlay the original image:
    image[-min_index[0]:-min_index[0]+map_size[0],-min_index[1]:-min_index[1]+map_size[1]] = original_image
    points[:,:,:2] -=min_index
    layer = image.copy()
    drawn_points = points[::3]
    for i,triangle in enumerate(drawn_points):
        triangle[:,[0,1]] = triangle[:,[1,0]]
        all_drawn = drawn_points.shape[0]
        color = np.array([255,255,0])*(i/(all_drawn-1)) + np.array([0,0,0])*((all_drawn-i)/(all_drawn-1))
        cv2.drawContours(layer,[triangle[:,:2]],0,color.tolist(),1)
        layer = np.asarray(layer)
        this_swept = layer[-min_index[0]:-min_index[0]+map_size[0],-min_index[1]:-min_index[1]+map_size[1]]
        this_swept = this_swept[:,::-1]
    

    return this_swept