import os
import gc
import cv2
import csv
import PIL
import time
import torch
import sklearn.metrics
import numpy as np
import pandas as pd
import torchvision.transforms as T

from torchvision         import models 
from collections         import defaultdict
from glob                import glob
from tqdm.auto           import tqdm
from matplotlib          import pyplot as plt

from utils          import isnotebook, dice, hypothesis_test
from configurations import THRESHOLDS, EXPERIMENT_NAME, TEST_RESOLUTION, OVERLAP_PERCENTAGES, FOLDS

TEST_VID_H, TEST_VID_W = TEST_RESOLUTION 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def evaluate_img(img_path, model, frame_annotations, nerve, median_areas, decision_thresh=None): 
    
    dice_scores = defaultdict(list) 
    hypo_tests  = defaultdict(list) 
    
    if type(decision_thresh) is float:
        thresholds  = [decision_thresh]  
    else:
        thresholds  = THRESHOLDS    
    
    pil_image = PIL.Image.open(img_path) # Dont use cv2.imread - BGR 
    image = T.Resize(TEST_RESOLUTION)(pil_image)  
    image = T.ToTensor()(image)
    image = torch.unsqueeze(image,0)
    inputs = image.to(device)    

    outputs = model(inputs)     
    y_pred = outputs['out'].data.cpu().numpy()
    
    mask = np.zeros((pil_image.size[1],pil_image.size[0]), dtype=np.float32)
    if frame_annotations is not None:     
        bounding_boxes = frame_annotations['bounding_boxes']
        bounding_boxes = eval(bounding_boxes)            
        if (bounding_boxes is not None) and (len(bounding_boxes) > 0):
            for box in bounding_boxes:
                (x, y, w, h) = [int(v) for v in box]
                mask[y:y+h, x:x+w] = 255.  

    gt = cv2.resize(mask, (TEST_VID_W, TEST_VID_H))  # Resizing operation may induce non-binary   ########     
    gt = (gt  > 217).astype(int)        # values around edges. Just to safeguard.     
    
    fp_area_thresh = int(0.1*median_areas[nerve])
    y_pred = y_pred.squeeze()
    for i in range(len(thresholds)):   
        thresh  = thresholds [i]   
        pred  = (y_pred  > thresh ).astype(int)  
        dice_score  = dice(gt , pred )    
        dice_scores[nerve].append(dice_score) 
        for pct in OVERLAP_PERCENTAGES:
            hypo_tests[nerve+'_pct_' +str(pct)].append(hypothesis_test(gt , pred , pct/100, fp_area_thresh )) 

    return dice_scores, hypo_tests, pil_image, pred, gt 
    
    
def frame_to_video_evaluations(frames, dice_scores, hypo_tests, nerve, joint = False) :
    
    if joint == True:
        nerves = ['sc' , 'isc']
    else:
        nerves = [nerve]

    vid_tests = defaultdict(list) 
    vid_dice  = defaultdict(list)     
        
    for nerve in nerves:
        for thresh in range(len(THRESHOLDS)): 
            #print(nerve, thresh, frames)
            for frame in frames: 
                frame_dice = dice_scores[frame][nerve][thresh]
                vid_dice[nerve + '_th_' + str(thresh)].append(frame_dice)                
                for pct in OVERLAP_PERCENTAGES: 
                    frame_test = hypo_tests[frame][nerve + '_pct_' +str(pct)][thresh]                    
                    vid_tests[nerve + '_pct_' +str(pct)+'_th_' + str(thresh)].append(frame_test) 

    avg_dice = defaultdict(list)    
    tests = defaultdict(list)
    precision_series   = defaultdict(list)
    recall_series      = defaultdict(list)
    specificity_series = defaultdict(list)
    f_score_series     = defaultdict(list)  
    
    for nerve in nerves:
        for thresh in range(len(THRESHOLDS)): 
            
            threshold_avg_dice = sum(vid_dice[nerve + '_th_' + str(thresh)])/len(vid_dice[nerve + '_th_' + str(thresh)])
            avg_dice[nerve].append(threshold_avg_dice) 
            
            for pct in OVERLAP_PERCENTAGES: 
                tp = vid_tests[nerve +'_pct_'+str(pct)+'_th_' + str(thresh)].count('tp') 
                tn = vid_tests[nerve +'_pct_'+str(pct)+'_th_' + str(thresh)].count('tn')
                fp = vid_tests[nerve +'_pct_'+str(pct)+'_th_' + str(thresh)].count('fp')
                fn = vid_tests[nerve +'_pct_'+str(pct)+'_th_' + str(thresh)].count('fn') 
                
                tests[nerve+'_pct_'+str(pct)+'_tp'].append(tp)
                tests[nerve+'_pct_'+str(pct)+'_tn'].append(tn)
                tests[nerve+'_pct_'+str(pct)+'_fp'].append(fp)
                tests[nerve+'_pct_'+str(pct)+'_fn'].append(fn) 
                
                try:
                    precision = tp/(tp + fp)
                except:
                    if len(precision_series[nerve+ '_pct_'+str(pct)]) > 0 : 
                        if precision_series[nerve+ '_pct_'+str(pct)][-1] is not None:
                            if precision_series[nerve+ '_pct_'+str(pct)][-1] > 0.99:
                                precision = 1.0
                            else: precision = None
                        else: precision = None
                    else: precision = None
                precision_series[nerve+'_pct_'+str(pct)].append(precision)

                try:
                    recall = tp/(tp + fn)
                except: recall = None
                recall_series[nerve+'_pct_'+str(pct)].append(recall)

                try:
                    specificity = tn/(tn + fp)
                except: specificity = None
                specificity_series[nerve +'_pct_'+str(pct)].append(specificity)

                try:
                    f_score = 2*tp/(2*tp + fp + fn)
                except: f_score = None
                f_score_series[nerve +'_pct_'+str(pct)].append(f_score)  
                
    df = pd.DataFrame({'threshold': THRESHOLDS}) 
    for nerve in nerves:   
        df['Dice_' + nerve] = avg_dice[nerve]        
        
    for nerve in nerves:    
        for pct in OVERLAP_PERCENTAGES: 
            df[nerve+'_pct_'+str(pct)+'_tp'] = tests[nerve+'_pct_'+str(pct)+'_tp']
            df[nerve+'_pct_'+str(pct)+'_tn'] = tests[nerve+'_pct_'+str(pct)+'_tn']
            df[nerve+'_pct_'+str(pct)+'_fp'] = tests[nerve+'_pct_'+str(pct)+'_fp']
            df[nerve+'_pct_'+str(pct)+'_fn'] = tests[nerve+'_pct_'+str(pct)+'_fn']
            df[nerve+'_pct_'+str(pct)+'_precision'] = precision_series[nerve+'_pct_'+str(pct)] 
            df[nerve+'_pct_'+str(pct)+'_recall'] = recall_series[nerve+'_pct_'+str(pct)] 
            df[nerve+'_pct_'+str(pct)+'_specificity'] = specificity_series[nerve+'_pct_'+str(pct)] 
            df[nerve+'_pct_'+str(pct)+'_f_score'] = f_score_series[nerve+'_pct_'+str(pct)]  
    return df 
    
    
def evaluate_vid(video_name, model, eval_folder, all_annotations, 
                 images_dir, median_areas, patient_info, joint = False): 
    video_name = str(video_name)
    frames = []
    dice_scores= {}  
    hypo_tests = {} 
    
    nerve = 'sc' #patient_info.loc[patient_info.vid_name == int(video_name),'nerve'].iloc[0]
    img_addresses = glob(os.path.join(images_dir, '**', video_name + '*.jpg'), recursive=True)
    img_addresses = sorted(img_addresses) 

    tqdm_txt = 'Images of Video: '+ video_name
    for i,img_path in tqdm(enumerate(img_addresses), tqdm_txt, position=2, leave=False):         
        frame = str(int(os.path.basename(img_path)[-7:-4])) 
        if frame in all_annotations[video_name]:
            frame_annotations = all_annotations[video_name][frame] #########################
        else: frame_annotations = None 
        
        im_dice, im_tests, _,_,_ = evaluate_img(img_path, model, frame_annotations, nerve, median_areas) 
        frames.append(frame)
        dice_scores[frame] = im_dice  
        hypo_tests[frame]  = im_tests     
          
    vid_eval = frame_to_video_evaluations(frames, dice_scores, hypo_tests, nerve)            
    eval_csv_path = os.path.join(eval_folder, video_name + '.csv') 
    vid_eval.to_csv(eval_csv_path, index=False) 
        
    return 
    
    
def evaluate_external_videos(data_dir, out_folder, models_folder):
    best_model = os.path.join(models_folder, EXPERIMENT_NAME+'_best_of_fold_1.pt')

    eval_folder = os.path.join(out_folder, 'new_video_evals/') 
    if not os.path.exists(eval_folder):
        print('Creating a folder for new video evaluations at: ', eval_folder)
        os.makedirs(eval_folder, exist_ok=False)
    else:
        print('New Video evaluations folder : ', eval_folder)  

    test_vids = os.listdir(os.path.join(data_dir, 'new_videos'))

    checkpoint = torch.load(best_model) 
    model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=1)
    model.load_state_dict(checkpoint['state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()     

    images_dir = os.path.join(data_dir,'new_images') 

    for video_name in test_vids: 
        evaluate_vid(video_name[:-4], model, eval_folder, new_annotations, images_dir, median_areas, patient_info) 

    model.cpu()
    del model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()     

    return eval_folder 
    
    
def evaluate_all_videos(data_dir, out_folder, models_folder, vid_splits, all_annotations, median_areas, patient_info):
    model_test_vid_map = {} 
    best_models = glob(os.path.join(models_folder, EXPERIMENT_NAME+'_best_of_fold_' + '*.pt')) 
    best_models.sort()
    if len(best_models) != FOLDS:
        print('Number of best models ({}) should be equal to the number of folds ({})'.format(len(best_models),FOLDS)) 
        # assert

    eval_folder = os.path.join(out_folder, 'video_evals/') 
    if not os.path.exists(eval_folder):
        print('Creating a folder for video evaluations at: ', eval_folder)
        os.makedirs(eval_folder, exist_ok=False)
    else:
        print('Video evaluations folder : ', eval_folder)  

    evaluated_vids = os.listdir(eval_folder)
    if len(evaluated_vids) > 0:
        print('Found {} files in evaluations folder. Skipping these names ...'.format(len(evaluated_vids)) )
        evaluated_vids = [int(vid[:4]) for vid in evaluated_vids]    

    for fold_num, vid_split in enumerate(vid_splits):
        model_test_vid_map[best_models[fold_num]] = vid_split['test'] 

    for fold_num, model_path in tqdm(enumerate(best_models), "Folds", position = 0):  
        test_vids = model_test_vid_map[model_path] 

        checkpoint = torch.load(model_path) 
        model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=1)
        model.load_state_dict(checkpoint['state_dict'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()     

        images_dir = os.path.join(data_dir, 'us_images') 

        tqdm_txt = 'Videos of fold#'+str(fold_num+1)
        for video_name in tqdm(test_vids, tqdm_txt , position = 1, leave=False): 
            if video_name not in evaluated_vids:
                evaluate_vid(video_name, model, eval_folder, all_annotations, 
                             images_dir, median_areas, patient_info) 

        model.cpu()
        del model, checkpoint
        gc.collect()
        torch.cuda.empty_cache()     

    return eval_folder 
    
    
def consolidate_vid_evals(eval_folder):
    evaluated_vids = glob(os.path.join(eval_folder, '**', '*.csv'), recursive=True)
    all_dfs = [pd.read_csv(eval_path) for eval_path in evaluated_vids]
    df_concat = pd.concat(all_dfs)
    df_mean = df_concat.groupby(df_concat.index).mean() 
    df_min  = df_concat.groupby(df_concat.index).min()
    df_max  = df_concat.groupby(df_concat.index).max() 
    df_std  = df_concat.groupby(df_concat.index).std()
    columns_to_drop = [col for col in df_mean.columns if col[-2:] in ['tp','tn','fp','fn'] ]
    df_mean = df_mean.drop(columns= columns_to_drop) 
    df_min = df_min.drop(columns= columns_to_drop) 
    df_max = df_max.drop(columns= columns_to_drop)
    df_std = df_std.drop(columns= columns_to_drop)   
    
    return df_mean, df_min, df_max, df_std   
    
    
def save_eval_files(out_folder, df_mean, df_min, df_max, df_std):
    eval_csv_path = os.path.join(out_folder, 'mean_evaluations.csv') 
    df_mean.to_csv(eval_csv_path, index=False) 
    eval_csv_path = os.path.join(out_folder, 'min_evaluations.csv') 
    df_min.to_csv(eval_csv_path, index=False) 
    eval_csv_path = os.path.join(out_folder, 'max_evaluations.csv') 
    df_max.to_csv(eval_csv_path, index=False) 
    eval_csv_path = os.path.join(out_folder, 'std_evaluations.csv') 
    df_std.to_csv(eval_csv_path, index=False) 
    
    
def get_evaluation_metrics(df_mean, df_min, df_max, df_std, nerve, joint=False):
    
    if joint == True:
        nerves = ['sc','isc']
    else: nerves = [nerve]
    
    f_scores     = defaultdict(list)
    precisions   = defaultdict(list)
    recalls      = defaultdict(list)
    specificitys = defaultdict(list)

    f_scores_l     = defaultdict(list)
    precisions_l   = defaultdict(list)
    recalls_l      = defaultdict(list)
    specificitys_l = defaultdict(list)

    f_scores_u     = defaultdict(list)
    precisions_u   = defaultdict(list)
    recalls_u      = defaultdict(list)
    specificitys_u = defaultdict(list)
    
    f_scores_std     = defaultdict(list)
    precisions_std   = defaultdict(list)
    recalls_std      = defaultdict(list)
    specificitys_std = defaultdict(list)    

    for nerve in nerves:
        for pct in OVERLAP_PERCENTAGES:
            col_f = nerve + '_pct_' + str(pct) + '_f_score'
            col_p = nerve + '_pct_' + str(pct) + '_precision'
            col_r = nerve + '_pct_' + str(pct) + '_recall'
            col_s = nerve + '_pct_' + str(pct) + '_specificity'
            max_f_score = max(df_mean[col_f])
            precision   = df_mean.loc[df_mean[col_f] == max_f_score, col_p].iloc[0]
            recall      = df_mean.loc[df_mean[col_f] == max_f_score, col_r].iloc[0]
            specificity = df_mean.loc[df_mean[col_f] == max_f_score, col_s].iloc[0]

            f_scores[nerve].append(max_f_score) 
            precisions[nerve].append(precision) 
            recalls[nerve].append(recall) 
            specificitys[nerve].append(specificity)         

            max_f_index = df_mean[col_f].idxmax()
            f_score_std     = df_std[col_f][max_f_index] 
            precision_std   = df_std[col_p][max_f_index]
            recall_std      = df_std[col_r][max_f_index]
            specificity_std = df_std[col_s][max_f_index]

            f_scores_std[nerve].append(f_score_std) 
            precisions_std[nerve].append(precision_std) 
            recalls_std[nerve].append(recall_std) 
            specificitys_std[nerve].append(specificity_std) 

            if max_f_score - f_score_std < 0: f_score_l = -max_f_score
            else: f_score_l = f_score_std

            if precision - precision_std < 0: precision_l = -precision
            else: precision_l = precision_std

            if recall - recall_std < 0: recall_l = -recall
            else: recall_l = recall_std

            if specificity - specificity_std < 0: specificity_l = -specificity
            else: specificity_l = specificity_std                
            
            f_scores_l[nerve].append(f_score_l) 
            precisions_l[nerve].append(precision_l) 
            recalls_l[nerve].append(recall_l) 
            specificitys_l[nerve].append(specificity_l)             
            
            if max_f_score + f_score_std > 1: f_score_u = 1-max_f_score
            else: f_score_u = f_score_std

            if precision + precision_std > 1: precision_u = 1-precision
            else: precision_u = precision_std

            if recall + recall_std > 1: recall_u = 1-recall
            else: recall_u = recall_std

            if specificity + specificity_std > 1: specificity_u = 1-specificity
            else: specificity_u = specificity_std            

            f_scores_u[nerve].append(f_score_u) 
            precisions_u[nerve].append(precision_u) 
            recalls_u[nerve].append(recall_u) 
            specificitys_u[nerve].append(specificity_u) 
            
    f = {'f_scores':f_scores, 'std': f_scores_std, 'low': f_scores_l, 'high': f_scores_u}
    p = {'precisions':precisions, 'std': precisions_std, 'low': precisions_l, 'high': precisions_u}
    r = {'recalls':recalls, 'std': recalls_std, 'low': recalls_l, 'high': recalls_u}
    s = {'specificitys':specificitys, 'std': specificitys_std, 'low': specificitys_l, 'high': specificitys_u}
    
    return f,p,r,s
    
    
def plot_p_r_f(p,r,f,title):
    ax=[]
    fig = plt.figure(figsize=(12,5))
    fig.suptitle(title, fontsize= 15)
    ax.append(fig.add_subplot(111))
    x_labels = [str(pct)+'%' for pct in OVERLAP_PERCENTAGES] 
    X = np.arange(4)

    nerve = 'sc'

    precision_err = [p['low'][nerve],p['high'][nerve]]
    recall_err    = [r['low'][nerve],r['high'][nerve]]
    f_score_err   = [f['low'][nerve],f['high'][nerve]]

    ax[0].bar(X-0.15, p['precisions'][nerve], yerr=precision_err, color = 'b', width = 0.15, label = 'Precision') 
    ax[0].bar(X+0.00, r['recalls'   ][nerve], yerr=recall_err   , color = 'g', width = 0.15, label = 'Recall') 
    ax[0].bar(X+0.15, f['f_scores'  ][nerve], yerr=f_score_err  , color = 'tab:orange', width = 0.15, label = 'F-score') 

    ax[0].set_xticks([0,1,2,3])
    ax[0].set_xticklabels([str(pct)+'%' for pct in OVERLAP_PERCENTAGES]) 
    ax[0].set_ylim([0,1.05])
    rects = ax[0].patches
    labels = [str(round(val,2))    for val in p['precisions'][nerve]] + \
                [str(round(val,2)) for val in r['recalls'   ][nerve]] + \
                [str(round(val,2)) for val in f['f_scores'  ][nerve]] 
    for j, (rect, label) in enumerate(zip(rects, labels)):
        height = rect.get_height()
        ax[0].text(rect.get_x() + rect.get_width() / 2+0.05 , height+0.02, label, ha="center")

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',ncol=4)

    plt.show()
    
    
def rectify_recall_series(y):
    z = []
    z.append(y[0])    
    for i in range(len(y)-1):
        if (z[i] ==1.) and np.isnan(y[i+1]): 
            z.append(1.)
        else:
            z.append(y[i+1])
    return np.array(z) 
    
    
def pr_curve(precisions, recalls):

    monotonic_p = [precisions[0] ]
    for i,s in enumerate(precisions[:-1]): 
        if precisions[i+1] > monotonic_p[i] :
            monotonic_p.append( precisions[i+1] ) 
        else:
            monotonic_p.append( monotonic_p[i] ) 
    
    precisions = np.array(monotonic_p)     
        
    decreasing_precisions = np.fliplr([precisions])[0]
    inv_recalls    = np.fliplr([recalls])[0]    
    
    inv_recalls = rectify_recall_series(inv_recalls)

    monotonic_r = [inv_recalls[0] ]
    for i,s in enumerate(inv_recalls[:-1]): 
        if inv_recalls[i+1] > monotonic_r[i] :
            monotonic_r.append( inv_recalls[i+1] ) 
        else:
            monotonic_r.append( monotonic_r[i] ) 
    
    inv_recalls = np.array(monotonic_r)     
    
    
    zero_added = 1

    if inv_recalls[0] > 0:
        inv_recalls = np.concatenate((np.array([0]), inv_recalls ))
        p0 = decreasing_precisions[0]
        decreasing_precisions = np.concatenate((np.array([p0]), decreasing_precisions ))
        zero_added = 0

    auc = sklearn.metrics.auc(decreasing_precisions, inv_recalls)
    
    return decreasing_precisions, inv_recalls, auc, zero_added   
    
    
def plot_pr_curve(nerve, title, df_mean, f_scores):
    pct_colors    = ['b','g','r','tab:orange'] 
    auc_txt_loc   = [(0.02,0.97),(0.02,0.91),(0.02,0.79),(0.02,0.53)]

    ax=[]
    fig = plt.figure(figsize=(6,6))
    ax.append(fig.add_subplot(111))
    fig.suptitle(title, fontsize= 15)
    for i,pct in enumerate(OVERLAP_PERCENTAGES):
        key = nerve + '_' + str(pct)
        prec = df_mean[nerve + '_pct_' + str(pct) + '_precision'] 
        reca = df_mean[nerve + '_pct_' + str(pct) + '_recall'] 
        fsco = list(df_mean[nerve + '_pct_' + str(pct) + '_f_score'])

        p_, r_, auc, zero_added = pr_curve(prec, reca)     

        max_f_score = f_scores['f_scores'][nerve][i]
        index_max = len(THRESHOLDS)-fsco.index(max_f_score) - zero_added 

        formatting = pct_colors[i]  
        label = 'iou@' + str(OVERLAP_PERCENTAGES[i]) + '%'
        ax[0].scatter(r_, p_,c=formatting,s=5) 
        ax[0].plot(r_, p_,c=formatting, linestyle=':', linewidth=1,label=label)    

    ax[0].legend(loc= 'lower left') 
    ax[0].set_xlabel("Recall")
    ax[0].set_ylabel("Precision")   
    ax[0].set_xlim([0,1.05])  
    ax[0].set_ylim([0,1.05])

    plt.show() 
