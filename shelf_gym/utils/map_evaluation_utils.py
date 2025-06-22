import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from shelf_gym.utils.map_calibration_utils import Cumulative_mIoU,mECE_Calibration_calc_3D
import cupy as cp
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import os
from shelf_gym.utils.scaling_utils import scale_semantic_probs


def produce_mECE_evaluation_v_time_plots(model,dataloader,experiment_name,n_classes = 7,collection_name = 'YCB-completion'):
    palette = sns.color_palette("rocket_r",n_colors = 10)

    occ_iou = []
    sem_iou = []
    semantic_mECE = []
    occupancy_mECE = []
    for i in range(10):
        occ_iou.append(Cumulative_mIoU(n_classes = 2))
        sem_iou.append(Cumulative_mIoU(n_classes = n_classes))
        semantic_mECE.append(mECE_Calibration_calc_3D(no_void = False, one_hot = False,n_classes = n_classes))
        occupancy_mECE.append(mECE_Calibration_calc_3D(no_void = False, one_hot = False,n_classes = 2))

    for batch in tqdm(dataloader):
        outputs = model.get_outputs(batch,normalize = True,intermediates = True)
        occupancy_gt = outputs['occupancy_gt']
        semantic_gt = outputs['semantic_gt']
        occupancy_probs = outputs['occupancy_probs']
        semantic_probs = outputs['semantic_probs']
        semantic_probs = scale_semantic_probs(semantic_probs)
        occupancy_gt_cp = cp.asarray(occupancy_gt[:,:,40:-20,20:-20].cpu().numpy()).reshape(-1)
        semantic_gt_cp = cp.asarray(semantic_gt[:,40:-20,20:-20].cpu().numpy()).reshape(-1)
        

        for i,data in enumerate(zip(occupancy_probs,semantic_probs)):
            occupancy,semantics = data
            occupancy = occupancy[:,:,40:-20,20:-20,:]
            semantics = semantics[:,40:-20,20:-20,:]
            occupancy_flat = cp.asarray(occupancy.astype(float)).reshape(-1,2)
            semantics_flat = cp.asarray(semantics.astype(float)).reshape(-1,n_classes)
            occ_iou[i].update_counts(occupancy_flat.argmax(axis = -1),occupancy_gt_cp)
            sem_iou[i].update_counts(semantics_flat.argmax(axis = -1),semantic_gt_cp)
            occupancy_mECE[i].update_bins(occupancy_flat,occupancy_gt_cp)
            semantic_mECE[i].update_bins(semantics_flat,semantic_gt_cp)
        
        o_meces = []
        s_meces = []
        s_ious = []
        o_ious = []
        steps = []
        for step,(o_mece,s_mece,o_iou,s_iou) in enumerate(zip(occupancy_mECE,semantic_mECE,occ_iou,sem_iou)):
            o_meces.append(o_mece.get_mECE().tolist())
            s_meces.append(s_mece.get_mECE().tolist())
            s_ious.append(s_iou.get_IoUs().mean().tolist())
            o_ious.append(o_iou.get_IoUs().mean().tolist())
            steps.append(step)

    results_df = pd.DataFrame({'semantic_miou':s_ious,'occupancy_miou':o_ious,'semantic_mece':s_meces,'occupancy_mece':o_meces,'samples':steps})
    os.makedirs('./results/{}/{}/plots/reliability_diagrams/'.format(collection_name,experiment_name),exist_ok = True)


    create_reliability_diagrams(occupancy_mECE,experiment_name,'Occupancy',task_type = collection_name)
    create_reliability_diagrams(semantic_mECE,experiment_name,'Semantic',task_type = collection_name)
    fig = plt.figure()

    ax = sns.scatterplot(data = results_df,x = 'semantic_mece',y = 'semantic_miou',hue = 'samples',palette = palette)
    sns.move_legend(ax,"upper left",bbox_to_anchor = (1,1))
    plt.title('Semantic mECEs vs mIoUs vs sample_number for {}'.format(experiment_name))
    plt.savefig('./results/{}/{}/plots/semantic_mece_v_miou.png'.format(collection_name,experiment_name),bbox_inches='tight')
    plt.close(fig)
    fig = plt.figure()

    ax = sns.scatterplot(data = results_df,x = 'occupancy_mece',y = 'occupancy_miou',hue = 'samples',palette = palette)
    sns.move_legend(ax,"upper left",bbox_to_anchor = (1,1))
    plt.title('mECEs vs mIoUs vs sample_number for {}'.format(experiment_name))
    plt.savefig('./results/{}/{}/plots/mece_v_miou.png'.format(collection_name,experiment_name),bbox_inches='tight')
    plt.close(fig)


def create_reliability_diagrams(ece_container,experiment_name,ece_type,task_type = 'completion'):
    for step in range(len(ece_container)):
        cc = ece_container[step]
        tmp = cc.return_calibration_results()
        for i in tmp.keys():
            fig,ax = plt.subplots(1,1)
            cal,conf,lims,total_bin_members,ECE = tmp[i]
            cal = np.nan_to_num(cp.asnumpy(cal),0)
            conf = np.nan_to_num(cp.asnumpy(conf),0)
            lims = cp.asnumpy(lims)-0.05

            fig.set_size_inches(8,8)
            p1 = ax.bar(x= lims, height = cal,  width = 0.8*(lims[1]-lims[0]),color = 'b',alpha = 0.5)
            ax.bar(x= lims, height = conf, width = 0.8*(lims[1]-lims[0]),color = 'r',alpha = 0.2)
            ax.plot(np.arange(11)/10,np.arange(11)/10)
            membership = (total_bin_members/total_bin_members.sum()*100)
            membership = np.round(membership,decimals = 1)
            membership = ['(' +str(i) + '%)' for i in membership]

            ax.bar_label(p1, labels = membership,label_type='edge')
            title = '{} RD - Step {} - Class {}'.format(ece_type,step,i)
            ax.set_title(title + ' - {} ECE = {:.3f}'.format(ece_type,ECE))
            ax.set_xlabel('Upper Confidence')
            ax.set_ylabel('Empirical Accuracy within bin (total pixel %)')
            plt.savefig('./results/{}/{}/plots/reliability_diagrams/'.format(task_type,experiment_name)+title+'.png',bbox_inches = 'tight') 
            plt.close(fig)


def produce_mECE_evaluation_v_time_plots_push(model,dataloader,experiment_name,n_classes = 7,reliability_diagrams = False):
    palette = sns.color_palette("rocket_r",n_colors = 10)

    occ_iou = []
    sem_iou = []
    diff_iou = []
    semantic_mECE = []
    occupancy_mECE = []
    diff_mECE = []
    for i in range(10):
        occ_iou.append(Cumulative_mIoU(n_classes = 2))
        diff_iou.append(Cumulative_mIoU(n_classes = 2))
        sem_iou.append(Cumulative_mIoU(n_classes = n_classes))
        semantic_mECE.append(mECE_Calibration_calc_3D(no_void = False, one_hot = False,n_classes = n_classes))
        occupancy_mECE.append(mECE_Calibration_calc_3D(no_void = False, one_hot = False,n_classes = 2))
        diff_mECE.append(mECE_Calibration_calc_3D(no_void = False, one_hot = False,n_classes = 2))


    for j,batch in enumerate(tqdm(dataloader)):
        outputs = model.get_outputs(batch,intermediates = True,max_obs = 10,normalize = True)

        occupancy_gt = outputs['occupancy_gt']
        semantic_gt = outputs['gt_semantics']
        difference_gt = outputs['difference_gt']
        pred_difference = outputs['pred_difference']
        occupancy_probs = outputs['occupancy_map']
        semantic_probs = outputs['semantic_map']
        semantic_probs = scale_semantic_probs(semantic_probs)

        occupancy_gt_cp = cp.asarray(occupancy_gt[:,:90,40:-20,20:-20].cpu().numpy()).reshape(-1)
        semantic_gt_cp = cp.asarray(semantic_gt[:,40:-20,20:-20].cpu().numpy()).reshape(-1)
        difference_gt_cp = cp.asarray(difference_gt.cpu().numpy()).reshape(-1)

        for i,data in enumerate(zip(occupancy_probs,semantic_probs,pred_difference)):
            occupancy,semantics,diff = data
            occupancy = occupancy[:,:90,40:-20,20:-20,:]
            semantics = semantics[:,40:-20,20:-20,:]
            occupancy_flat = cp.asarray(occupancy.astype(float)).reshape(-1,2)
            occupancy_flat = occupancy_flat/occupancy_flat.sum(axis = 1,keepdims = True)
            semantics_flat = cp.asarray(semantics.astype(float)).reshape(-1,n_classes)
            semantics_flat = semantics_flat/semantics_flat.sum(axis = 1,keepdims = True)
            pred_difference_flat = cp.asarray(diff.reshape(-1,2))
            pred_difference_flat = pred_difference_flat/pred_difference_flat.sum(axis = 1,keepdims = True)

            occ_iou[i].update_counts(occupancy_flat.argmax(axis = -1),occupancy_gt_cp)
            sem_iou[i].update_counts(semantics_flat.argmax(axis = -1),semantic_gt_cp)
            diff_iou[i].update_counts(pred_difference_flat.argmax(axis = -1),difference_gt_cp)
            occupancy_mECE[i].update_bins(occupancy_flat,occupancy_gt_cp)
            semantic_mECE[i].update_bins(semantics_flat,semantic_gt_cp)
            diff_mECE[i].update_bins(pred_difference_flat,difference_gt_cp)
    o_meces = []
    s_meces = []
    d_meces =[]
    s_ious = []
    o_ious = []
    d_ious = []
    steps = []
    for step,(o_mece,s_mece,d_mece,o_iou,s_iou,d_iou) in enumerate(zip(occupancy_mECE,semantic_mECE,diff_mECE,occ_iou,sem_iou,diff_iou)):
        o_meces.append(o_mece.get_mECE().tolist())
        s_meces.append(s_mece.get_mECE().tolist())
        d_meces.append(d_mece.get_mECE().tolist())
        s_ious.append(s_iou.get_IoUs().mean().tolist())
        o_ious.append(o_iou.get_IoUs().mean().tolist())
        d_ious.append(d_iou.get_IoUs().mean().tolist())
        steps.append(step)

    results_df = pd.DataFrame({'semantic_miou':s_ious,'occupancy_miou':o_ious,'difference_miou':d_ious,
                            'semantic_mece':s_meces,'occupancy_mece':o_meces,'difference_mece':d_meces,
                            'samples':steps})
    

    os.makedirs('./results/push/{}/plots/reliability_diagrams/'.format(experiment_name),exist_ok = True)
    if(reliability_diagrams):
        create_reliability_diagrams(occupancy_mECE,experiment_name,'Occupancy','push')
        create_reliability_diagrams(semantic_mECE,experiment_name,'Semantic','push')
        create_reliability_diagrams(diff_mECE,experiment_name,'Differences','push')

    fig = plt.figure()

    ax = sns.scatterplot(data = results_df,x = 'semantic_mece',y = 'semantic_miou',hue = 'samples',palette = palette)
    sns.move_legend(ax,"upper left",bbox_to_anchor = (1,1))
    plt.title('Semantic mECEs vs mIoUs vs sample_number for {}'.format(experiment_name))
    plt.savefig('./results/push/{}/plots/semantic_mece_v_miou.png'.format(experiment_name),bbox_inches='tight')
    plt.close(fig)
    fig = plt.figure()

    ax = sns.scatterplot(data = results_df,x = 'occupancy_mece',y = 'occupancy_miou',hue = 'samples',palette = palette)
    sns.move_legend(ax,"upper left",bbox_to_anchor = (1,1))
    plt.title('mECEs vs mIoUs vs sample_number for {}'.format(experiment_name))
    plt.savefig('./results/push/{}/plots/mece_v_miou.png'.format(experiment_name),bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = sns.scatterplot(data = results_df,x = 'difference_mece',y = 'difference_miou',hue = 'samples',palette = palette)
    sns.move_legend(ax,"upper left",bbox_to_anchor = (1,1))
    plt.title('Differences mECEs vs mIoUs vs sample_number for {}'.format(experiment_name))
    plt.savefig('./results/push/{}/plots/difference_mece_v_miou.png'.format(experiment_name),bbox_inches='tight')
    plt.close(fig)



def produce_non_evidential_mECE_evaluation_v_time_plots(model,dataloader,experiment_name,n_classes = 7,collection_name = 'YCB-completion'):
    palette = sns.color_palette("rocket_r",n_colors = 10)

    occ_iou = []
    sem_iou = []
    semantic_mECE = []
    occupancy_mECE = []
    for i in range(10):
        occ_iou.append(Cumulative_mIoU(n_classes = 2))
        sem_iou.append(Cumulative_mIoU(n_classes = n_classes))
        semantic_mECE.append(mECE_Calibration_calc_3D(no_void = False, one_hot = False,n_classes = n_classes))
        occupancy_mECE.append(mECE_Calibration_calc_3D(no_void = False, one_hot = False,n_classes = 2))

    for batch in tqdm(dataloader):
        outputs = model.get_outputs(batch,normalize = True,intermediates = True)
        occupancy_gt = outputs['occupancy_gt']
        semantic_gt = outputs['semantic_gt']
        occupancy_probs = outputs['occupancy_probs']
        semantic_probs = outputs['semantic_probs']
        semantic_probs = scale_semantic_probs(semantic_probs)

        occupancy_gt_cp = cp.asarray(occupancy_gt[:,:,40:-20,20:-20].cpu().numpy()).reshape(-1)
        semantic_gt_cp = cp.asarray(semantic_gt[:,40:-20,20:-20].cpu().numpy()).reshape(-1)
        

        for i,data in enumerate(zip(occupancy_probs,semantic_probs)):
            occupancy,semantics = data

            occupancy = occupancy[:,:,40:-20,20:-20]
            semantics = semantics[:,40:-20,20:-20,:]

            neg_occupancy = 1-occupancy
            occupancy =  np.stack((neg_occupancy,occupancy),axis = -1)

            occupancy_flat = cp.asarray(occupancy.astype(float)).reshape(-1,2)
            semantics_flat = cp.asarray(semantics.astype(float)).reshape(-1,n_classes)

            occ_iou[i].update_counts(occupancy_flat.argmax(axis = -1),occupancy_gt_cp)
            sem_iou[i].update_counts(semantics_flat.argmax(axis = -1),semantic_gt_cp)
            occupancy_mECE[i].update_bins(occupancy_flat,occupancy_gt_cp)
            semantic_mECE[i].update_bins(semantics_flat,semantic_gt_cp)
        
        o_meces = []
        s_meces = []
        s_ious = []
        o_ious = []
        steps = []
        for step,(o_mece,s_mece,o_iou,s_iou) in enumerate(zip(occupancy_mECE,semantic_mECE,occ_iou,sem_iou)):
            o_meces.append(o_mece.get_mECE().tolist())
            s_meces.append(s_mece.get_mECE().tolist())
            s_ious.append(s_iou.get_IoUs().mean().tolist())
            o_ious.append(o_iou.get_IoUs().mean().tolist())
            steps.append(step)

    results_df = pd.DataFrame({'semantic_miou':s_ious,'occupancy_miou':o_ious,'semantic_mece':s_meces,'occupancy_mece':o_meces,'samples':steps})

    os.makedirs('./results/{}/{}/plots/reliability_diagrams/'.format(collection_name,experiment_name),exist_ok = True)


    create_reliability_diagrams(occupancy_mECE,experiment_name,'Occupancy',task_type = collection_name)
    create_reliability_diagrams(semantic_mECE,experiment_name,'Semantic',task_type = collection_name)
    fig = plt.figure()
    ax = sns.scatterplot(data = results_df,x = 'semantic_mece',y = 'semantic_miou',hue = 'samples',palette = palette)
    sns.move_legend(ax,"upper left",bbox_to_anchor = (1,1))
    plt.title('Semantic mECEs vs mIoUs vs sample_number for {}'.format(experiment_name))
    plt.savefig('./results/{}/{}/plots/semantic_mece_v_miou.png'.format(collection_name,experiment_name),bbox_inches='tight')
    plt.close(fig)
    fig = plt.figure()

    ax = sns.scatterplot(data = results_df,x = 'occupancy_mece',y = 'occupancy_miou',hue = 'samples',palette = palette)
    sns.move_legend(ax,"upper left",bbox_to_anchor = (1,1))
    plt.title('mECEs vs mIoUs vs sample_number for {}'.format(experiment_name))
    plt.savefig('./results/{}/{}/plots/mece_v_miou.png'.format(collection_name,experiment_name),bbox_inches='tight')
    plt.close(fig)