
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from os.path import join, exists

import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from models.unet import Unet
from utils import path_utils

import json

import pprint

from eval_utils import inference_funcs, load_eval_modules



def get_confusion_matrix_binary(label, pred, size, num_class, ignore=-1):
    """
    Modification of get_confusion_matrix: the input is considered to be binary thresholded already
    """

    #output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    #seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    
    seg_pred = np.asarray(pred.cpu().numpy() , dtype=np.uint8)
    # print('seg_pred shape:', seg_pred.shape)
    
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix



def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    source: https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/pytorch-v1.1/lib/utils/utils.py
    """
    
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    #print('output shape:', output.shape)
    
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    
    #print('seg_pred shape:', seg_pred.shape)
    
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix



def metrics_fg_only(predicted, labels):

    pred_fg = torch.eq(predicted, torch.ones(1).long().cuda())
    gt_fg = torch.eq( labels, torch.ones(1).long().cuda()  )
    
    eps = 1e-9
    
    intersection_now = torch.sum( pred_fg * gt_fg  )
    union_now = torch.sum(pred_fg) + torch.sum(gt_fg) - intersection_now
    
    iou= (intersection_now.float()) / ( union_now.float() + eps )
    
    gt_n = torch.sum(gt_fg)
    acc = torch.sum(intersection_now) / (gt_n.float() + eps)
    
    return iou, acc



def segmentation_metrics(predicted, labels):
    # normalize the prediction
    predicted = F.softmax(predicted, dim=1)     # for training, PyTorch expects the prediction to be unnormalized

    p = torch.argmax(predicted, dim=1)

    iou = torch.zeros(predicted.shape[1]).cuda()  # per class iou
    acc = torch.zeros(predicted.shape[1]).cuda()  # per class accuracy

    gt_n = torch.zeros(predicted.shape[1]).cuda()  # to save number of true pixels for every class. needed for fwIou

    eps = 1e-9

    for k in range(predicted.shape[1]):
        pred_now = torch.eq(p, k*torch.ones(1).long().cuda())
        gt_now = torch.eq( labels, k*torch.ones(1).long().cuda()  )

        intersection_now = torch.sum( pred_now * gt_now  )
        union_now = torch.sum(pred_now) + torch.sum(gt_now) - intersection_now

        iou[k] = (intersection_now.float()) / ( union_now.float() + eps )
        gt_n[k] = torch.sum(gt_now)

        # calculate accuracy
        acc[k] = torch.sum(intersection_now) / (gt_n[k].float() + eps)

            # resume from here

    mIoU = torch.mean(iou)
    fwIoU = torch.sum(iou * gt_n) / torch.sum(gt_n)

    total_pixels = predicted.shape[0]*predicted.shape[1]*predicted.shape[2]*predicted.shape[3]
    #accuracy = (100.0*correct_n) / float(total_pixels) #float(predicted.shape[1])
    m_accuracy = 100.0*torch.mean(acc)

    return iou, mIoU, fwIoU, acc, m_accuracy, gt_n



def evaluate_metrics(args, datasets, dataloaders, model):
    
    out_dir = path_utils.get_eval_out_dir(args)
    best_threshold = load_eval_modules.load_best_threshold(args, dataloaders, model)
    
    num_classes = 2
    data_loader = dataloaders[args['phase']]
    
    per_class_iou = torch.zeros(num_classes).cuda()
    per_class_acc = torch.zeros(num_classes).cuda()
    iou = torch.zeros(1).cuda()
    fwIoU = torch.zeros(1).cuda()
    accuracy = torch.zeros(1).cuda()
    acc_sinkhole = torch.zeros(1).cuda()

    iou_threshold = torch.zeros(1).cuda()
    acc_threshold = torch.zeros(1).cuda()

    # counting number of pixels
    gt_n = torch.zeros(num_classes).cuda()

    confusion_matrix = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for idx, (data, labels) in enumerate(data_loader):

            data = data.cuda()
            labels = labels.cuda()
            
            predictions = model(data)
            
            assert labels.shape[-2:] == predictions.shape[-2:], f"idx: {idx}, labels.shape: {labels.shape[-2:]}, predictions.shape[1:], {predictions.shape[-2:]}"
            
            startx, endx, starty, endy = get_patch_idxs(idx, data_loader)
            
            predictions = predictions[:,:,starty:endy,startx:endx]
            
            labels = labels[:,starty:endy,startx:endx]

            iou_b, miou_b, fwiou_b, acc_b, m_acc_b, n_b = segmentation_metrics(predictions, labels)

            pred_final = torch.softmax(predictions, dim=1)
            
            pred_final = (pred_final[:,1,:,:] > best_threshold).long()

            iou_b_threshold, acc_b_threshold = metrics_fg_only(pred_final, labels)

            iou_threshold += iou_b_threshold
            acc_threshold += acc_b_threshold

            per_class_iou += iou_b
            iou += miou_b
            fwIoU += fwiou_b
            per_class_acc += acc_b
            accuracy += m_acc_b       # mean accuracy
            gt_n += n_b

            
            confusion_matrix += get_confusion_matrix_binary(label=labels, pred=pred_final,size=(args['cutout_size'], args['cutout_size']),num_class=num_classes)
            
    # compute averages
    print('length of dataloader: ', len(data_loader))
    
    per_class_iou /= float(len(data_loader))
    iou /= float(len(data_loader))
    fwIoU /= float(len(data_loader))
    per_class_acc /= float(len(data_loader))
    accuracy /= float(len(data_loader))
    gt_n /= float(len(data_loader))

    iou_threshold /= float(len(data_loader))
    acc_threshold /= float(len(data_loader))

    confusion_matrix /= float(len(data_loader))


    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    print('\n\n\n\n NEW metrics...')
    print('pixel acc: ', pixel_acc)
    print('mean acc: ', mean_acc)
    print('IoU full: ', IoU_array)
    print('m IoU: ', mean_IoU)
    
    area_pred_raw  = inference_funcs.gen_inference_over_area(args, dataloaders, model, best_threshold)
    area_labels = gen_pasted_area_labels(args, dataloaders)
    
    ##save test result images JZ 10/31/2022
    save_eval_images(out_dir,area_pred_raw,area_labels,best_threshold,args, datasets)
 
    
    ## PR Curve

    y_true = area_labels.view(-1).detach().cpu().numpy() #np.array([0, 0, 1, 1])
    y_scores = area_pred_raw.view(-1).detach().cpu().numpy() #np.array([0.1, 0.4, 0.35, 0.8])
    
    average_precision = average_precision_score(y_true, y_scores)

    # AUC: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
    auc = roc_auc_score(y_true, y_scores)

    print('average_precision: ', average_precision)
    print('AUC: ', auc)

    # Saving logs
    # This is the set of results used to generate the tables.
    name_string = f"metrics_{args['phase']}.json"
    fname = os.path.join(out_dir, name_string)
    
    with open(fname, 'w') as result_file:
        results = {
            
            'auc' : float(auc),
            'avg_precision' : float(average_precision),
            'new_metrics_pixel_acc' : float(pixel_acc),
            'new_metrics_mean_acc' : float(mean_acc),
            'new_metrics_mean_iou' : float(mean_IoU),
            'true_feature_iou_full' : float(IoU_array[1]),
        }
        
        pprint.pprint(results)
        
        # 6/21/21 https://www.geeksforgeeks.org/json-dump-in-python/
        json.dump(results, result_file)
        # end code
        print(f'Writing these results to {fname}')
    
    return results
        
    
def find_best_threshold(args, phase, dataloaders, model):
    
    out_dir = path_utils.get_eval_out_dir(args)
    num_classes = 2
    
    data_loader = dataloaders[phase] # Importantly, this is NOT data_mode.
    
    best_iou = torch.zeros(1).cuda()
    best_acc = torch.zeros(1).cuda()
    best_threshold = torch.zeros(1).cuda()

    acc_log = []
    iou_log = []
    thresh_log = []

    iou_new = []
    acc_new = []

    thresholds = np.linspace(start=0, stop=1, num=21)
    for j in range(thresholds.shape[0]):
        threshold = thresholds[j]
        sinkhole_iou = torch.zeros(1).cuda()
        sinkhole_acc = torch.zeros(1).cuda()

        confusion_matrix = np.zeros((num_classes, num_classes))

        with torch.no_grad():
            for idx, (data, labels) in enumerate(data_loader):
                
                data = data.cuda()
                labels = labels.cuda()
                
                predictions = model(data) # Classifier will separate the data as needed.
                
                predictions = torch.softmax(predictions, dim=1)
                
                startx, endx, starty, endy = get_patch_idxs(idx, data_loader)
                
                predictions = predictions[:,:,starty:endy,startx:endx] 
                
                labels = labels[:,starty:endy,startx:endx]

                pred_final = (predictions[:,1,:,:] > threshold).long()
                

                iou_b, acc = metrics_fg_only(pred_final, labels)

                sinkhole_iou += iou_b
                sinkhole_acc += acc

                confusion_matrix += get_confusion_matrix_binary(label=labels, pred=pred_final,size=(args['cutout_size'], args['cutout_size']),num_class=num_classes)

        # compute averages
        sinkhole_iou /= float(len(data_loader))
        sinkhole_acc /= float(len(data_loader))

        acc_log.append(sinkhole_acc.item())
        iou_log.append(sinkhole_iou.item())
        thresh_log.append(threshold)

#        if sinkhole_iou > best_iou:
#            best_iou = sinkhole_iou
#            best_threshold = threshold

        # new method

        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        pixel_acc = tp.sum()/pos.sum()
        mean_acc = (tp/np.maximum(1.0, pos)).mean()
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()

        iou_new.append(IoU_array[1]*100.0)
        acc_new.append(mean_acc)

        ##use IOU calculated from the new method above
        new_sink_iou=torch.tensor(IoU_array[1]*100.0)
        if new_sink_iou > best_iou:
            best_iou = new_sink_iou
            best_threshold = threshold        

    print(f'finished analysis for phase {phase}==== best iou: ', best_iou.item(), ' threshold: ', best_threshold)

    plt.figure()
    #plt.plot(thresh_log, acc_new)
    plt.plot(thresh_log, iou_new)
    #plt.legend(['accuracy', 'IoU'])
    #plt.title('Sinkhole metrics only (no BG)')
    plt.ylabel('Sinkhole IoU (%)')
    plt.xlabel('threshold')
    name_string = f'threshold_metrics_new_{phase}.png'
    fname = os.path.join(out_dir, name_string)
    plt.savefig(fname)
    plt.close()
    
    return best_threshold


def gen_pasted_area_labels(args, dataloaders):

    this_phase = args['phase']
    
    this_data_loader = dataloaders[this_phase]
    dset = this_data_loader.dataset
    
    size_x, size_y = this_data_loader.dataset.get_pred_area_dims_pixels()
    
    true_label = torch.zeros(size_y, size_x)

    with torch.no_grad():

        for idx, (_, labels) in enumerate(this_data_loader):
            
            startx, endx, starty, endy = get_patch_idxs(idx, this_data_loader)
            
            # Cut out the padding
            labels = labels[:,starty:endy,startx:endx]
            left, upper, right, lower = dset.get_rc_from_index_no_pad(idx)
            
            # Reassign to the larger image.
            true_label[upper:lower, left:right] = labels
            
    return true_label # The real return statement, above is just for debugging


def get_patch_idxs(idx, loader):
    
    left, upper, right, lower = loader.dataset.get_rc_relative_to_pad(idx) 
    
    # Use original naming conventions
    
    startx, endx, starty, endy = left, right, upper, lower
    
    return startx, endx, starty, endy

#save evaluation image results JZ 10/31/2022
#fix image size JZ 03/14/2023
def save_eval_images(out_dir,area_pred_raw,area_labels,best_threshold,args, datasets):
    
    this_phase = args['phase']
    this_data = datasets[this_phase]

    if (args['input_type'] == 'dem_ddxy' or args['input_type'] == 'spp'):
        size_l, size_w = this_data.original_shape
    else:
        size_w, size_l = this_data.original_shape      
    
    #plt options
    eva_dpi=600
    eva_color = "Blues"
    ##save soft prediction
    cropped_area = area_pred_raw[0:size_l, 0:size_w]
    area_for_save = cropped_area.detach().cpu().numpy()
    plt_test_soft_pred_path = os.path.join (out_dir,'plt_test_soft_pred.png')
    plt.figure(dpi=eva_dpi)
    plt.set_cmap(eva_color)
    plt.imshow(area_for_save)
    plt.axis('off')
    plt.savefig(plt_test_soft_pred_path,bbox_inches='tight',pad_inches = 0)
    plt.close()

    ##save test result with best threshold 
    pred_best_threshold= 1.0 * (area_for_save > best_threshold)
    plt_test_pred_path = os.path.join (out_dir,'plt_test_pred_best_threshold.png')
    plt.figure(dpi=eva_dpi)
    plt.set_cmap(eva_color)
    plt.imshow(pred_best_threshold)
    plt.axis('off')
    plt.savefig(plt_test_pred_path,bbox_inches='tight',pad_inches = 0)
    plt.close()

    ##save test label image
    cropped_area = area_labels[0:size_l, 0:size_w]
    area_for_save = cropped_area.detach().cpu().numpy()
    plt_test_label_path = os.path.join (out_dir,'plt_test_label.png')
    plt.figure(dpi=eva_dpi)
    plt.set_cmap(eva_color)
    plt.imshow(area_for_save)
    plt.axis('off')
    plt.savefig(plt_test_label_path,bbox_inches='tight',pad_inches = 0)
    plt.close() 
    
    print('Saved test result images to: ', out_dir)
    
