import os
import cv2
import csv
import glob
import json
import random 

import numpy as np
import pandas as pd

from collections         import defaultdict
from glob                import glob
from itertools           import zip_longest
from matplotlib          import pyplot as plt
from matplotlib.ticker   import MaxNLocator

import configurations
from configurations import VAL_PROPORTION, EXPERIMENT_NAME, FOLDS

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
        
def live_plot(data_dict, disp_text, disp_chart):
    
    ax=[]
    f = plt.figure(figsize=(12,5))
    ax.append(f.add_subplot(121))
    ax.append(f.add_subplot(122))
    
    for label,data in data_dict.items(): 
        line_style = ''
        if 'Train' in label:
            line_style = '--'
        
        if 'Loss' not in label:
            plot_num = 1
            if 'ISC' in label:
                if 'Dice' in label: color='g'
                #else: color='c'
            elif 'SC' in label:
                if 'Dice' in label: color='b'
                #else: color='m'
            else:
                color='y'
        else:
            plot_num = 0
            if 'BCE' in label: 
                color='g'
                plot_num = -1                
            else: color='r'
            
        formatting = color+line_style
        x_label = 'Epoch'

        if plot_num >= 0:
            ax[plot_num].plot(data, formatting, label=label)

    for axi in ax:            
        axi.grid(True)  
        axi.legend(loc='best') 
        axi.set_xlabel(x_label)
        axi.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Score')

    metrics_df = pd.DataFrame.from_dict(data_dict)
    disp_text.update(metrics_df)  # metrics_df.iloc[::-1]    
    disp_chart.update(f)
    plt.close()   # .close prevents the normal figure display at end of cell execution   
    #plt.show() 

def break_into_images(video_path, images_dir):
    base_primary     = os.path.splitext(os.path.basename(video_path))[0] 
    vs = cv2.VideoCapture(video_path)
    while vs.isOpened():
        ret, frame = vs.read()
        if ret==True:
            current_frame_num = int(vs.get(cv2.CAP_PROP_POS_FRAMES))-1
            out_path = os.path.join(images_dir, base_primary + '_' + str(current_frame_num).zfill(3) + '.jpg')  
            cv2.imwrite(out_path, frame)
        else:
            vs.release() 
            
def prepare_folders(data_dir) :

    out_folder = os.path.join(data_dir, 'output/')
    if not os.path.exists(out_folder):
        print('Creating an output folder at: ', out_folder)
        os.makedirs(out_folder, exist_ok=False)
    else:
        print('Output folder : ', out_folder)
                
    test_vids_dir = os.path.join(out_folder,'test_videos/')      
    if not os.path.exists(test_vids_dir):
        print('Creating folder for test videos at: ', test_vids_dir)
        os.makedirs(test_vids_dir, exist_ok=False)
    else:
        print('Folder for test videos at : ', test_vids_dir)    

    images_dir = os.path.join(data_dir,'us_images') 
    if not os.path.exists(images_dir):
        print('Creating a folder for video frames at: ', images_dir)
        os.makedirs(images_dir, exist_ok=False)

        videos_folder = os.path.join(data_dir, 'us_videos')
        videos = glob(videos_folder + "/*.mp4")
        print ('Breaking down %d videos into frames' % len(videos) )
        for video_path in videos:
            break_into_images(video_path, images_dir)
        print ('Broken down %d videos into %d frames' % (len(videos), len(os.listdir(images_dir)) ))
    else:
        print('Located %d files in images folder at: %s'  % (len(os.listdir(images_dir)),images_dir)) 
        
    ex_images_dir = os.path.join(data_dir,'new_images') 
    if not os.path.exists(ex_images_dir):
        print('Creating a folder for external video frames at: ', ex_images_dir)
        os.makedirs(ex_images_dir, exist_ok=False)

        videos_folder = os.path.join(data_dir, 'new_videos')
        videos = glob(videos_folder + "/*.mp4")
        print ('Breaking down %d external videos into frames' % len(videos) )
        for video_path in videos:
            break_into_images(video_path, ex_images_dir)
        print ('Broken down %d external videos into %d frames' % (len(videos), len(os.listdir(ex_images_dir)) ))
    else:
        print('Located %d files in images folder at: %s'  % (len(os.listdir(ex_images_dir)),ex_images_dir))         

    return images_dir, out_folder, test_vids_dir  
    
    
def get_annotations(annotations_dir):
    annotations_dir   = os.path.join(annotations_dir, 'bb_annotations')
    annotations_paths = glob(os.path.join(annotations_dir, '**', '*.txt'), recursive=True)
    print('# annotation txt files = ', len(annotations_paths))

    all_annotations = defaultdict(dict)
    for a_path in annotations_paths:
        vid_name = os.path.basename(a_path)[:-4]
        annotations={}
        input_file = open(a_path, 'r')
        for line in input_file:
            json_decode = json.loads(line)
            for item in json_decode:
                all_annotations[vid_name][item] = json_decode[item]
        input_file.close() 

    return all_annotations    
    
    
def get_nerve_areas(all_annotations, patient_info):
    areas_sc  = []
    areas_isc = []
    box_counts_sc  = []
    box_counts_isc = []

    for video_name in all_annotations:
        nerve = patient_info.loc[patient_info.vid_name == int(video_name),'nerve'].iloc[0]
        for frame_num in all_annotations[video_name]:
            tracker        = all_annotations[video_name][frame_num]['tracker']
            bounding_boxes = all_annotations[video_name][frame_num]['bounding_boxes']

            if tracker is not None:        
                bounding_boxes = eval(bounding_boxes)  
                if (bounding_boxes is not None) and (len(bounding_boxes) > 0):
                    mask = np.zeros((542, 562), dtype=np.float32)
                    for box in bounding_boxes:
                        (x, y, w, h) = [int(v) for v in box]
                        mask[y:y+h, x:x+w] = 255.        

                    mask = cv2.resize(mask, (256, 256)) 
                    area = np.count_nonzero(mask == 255.)

                    if nerve == 'sc':
                        areas_sc.append(area)
                        box_counts_sc.append(len(bounding_boxes))
                    elif nerve == 'isc':
                        areas_isc.append(area)
                        box_counts_isc.append(len(bounding_boxes)) 
                        
    sc_median_area  = np.median(areas_sc) 
    isc_median_area = np.median(areas_isc)  
    
    print ('Median Area of union of sc nerve bounding boxes  =' , sc_median_area) 
    #print ('Median Area of union of isc nerve bounding boxes =' , isc_median_area) 
    '''
    if isnotebook():
        fig = plt.figure(figsize=(12,5))
        arr = np.array([areas_sc], dtype=object)
        #arr = np.array([areas_sc,areas_isc], dtype=object)
        plt.hist(arr, bins = 100)  
        plt.xlabel('Area captured by nerve')
        plt.ylabel('Frames count')  
        plt.title('Area Histogram')
        plt.grid()
        plt.show()    
    '''
    return {'sc': sc_median_area }     
    
    
def train_val_test_splits(data_dir, images_dir, nerve):
    patient_data_csv_path = os.path.join(data_dir, 'patient_data.csv') 
    all_patient_info = pd.read_csv(patient_data_csv_path) 
    
    if nerve in ['sc','isc']:
        patient_info  = all_patient_info[all_patient_info.nerve == nerve]
    else: patient_info = all_patient_info
    
    if EXPERIMENT_NAME is None :
        folds_csv_path = os.path.join(data_dir, 'output', 'folds.csv') 
    else:
        folds_csv_path = os.path.join(data_dir, 'output', EXPERIMENT_NAME + '_folds.csv') 

    vid_splits= []
    if os.path.exists(folds_csv_path) :
        print('Existing folds and train/val/test splits found at :', folds_csv_path) 
        data = pd.read_csv(folds_csv_path) 
        for fold_num in range(1,FOLDS+1):  
            train = list(data['fold_'+ str(fold_num) +'_train'].dropna().astype(int))
            val   = list(data['fold_'+ str(fold_num) +'_val'  ].dropna().astype(int))
            test  = list(data['fold_'+ str(fold_num) +'_test' ].dropna().astype(int))
            vid_splits.append({'train':train, 'val': val, 'test': test })         
    else:
        negative_vids = list(patient_info[patient_info.nerve == 'neg'].vid_name)
        to_split  = patient_info[patient_info.nerve != 'neg']
        print(len(to_split))
        random.seed(7)
        for i in range(FOLDS):
            folds_left = FOLDS-i
            test_proportion = 1/folds_left

            grps = to_split.groupby(['left_right','gain','gender'], group_keys=False)
            test = grps.apply(lambda x: x.sample(max(round(len(x)*test_proportion), random.choice([0,1]))))
            # Remaining is train + Val
            train_val = patient_info[~patient_info.vid_name.isin(set(test.vid_name)|set(negative_vids))]
            grps  = train_val.groupby(['left_right','gain','gender'], group_keys=False)
            val   = grps.apply(lambda x: x.sample(max(round(len(x)*VAL_PROPORTION), random.choice([0,1]))))
            train = train_val[~train_val.vid_name.isin(set(val.vid_name))]    
            to_split = to_split[~to_split.vid_name.isin(set(test.vid_name))]  # for next fold 

            vid_splits.append({'train':list(train.vid_name) + negative_vids, 
                        'val'  :list(val.vid_name), 'test' :list(test.vid_name ) })

        print('Generated %d fold stratified video splits of train/val/test data' % len(vid_splits)) 
        
        header_row = []
        splits_in_folds = []
        for fold_num,split in enumerate(vid_splits):
            for key in ['train','val','test']:
                splits_in_folds.append(split[key])
                header_row.append('fold_' + str(fold_num+1) + '_' + key)

        folds_columns = list(zip_longest(*splits_in_folds)) 
        
        with open(folds_csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header_row)
            writer.writerows(folds_columns) 
        print('Generated folds and train/val/test splits csv at :', folds_csv_path) 
        
    img_splits = []
    all_images = glob(os.path.join(images_dir, '**', '*.jpg'), recursive=True)

    for split in vid_splits:
        train_vids = split['train']
        val_vids   = split['val']
        test_vids  = split['test']
        train_images = [f for f in all_images if int(os.path.basename(f)[:4]) in train_vids]        
        val_images   = [f for f in all_images if int(os.path.basename(f)[:4]) in val_vids]        
        test_images  = [f for f in all_images if int(os.path.basename(f)[:4]) in test_vids]

        img_splits.append({'train':train_images, 'val' :val_images, 'test':test_images})

        text = 'Train/Val/Test split of images is {} / {} / {}'.format(
                                        len(train_images),len(val_images),len(test_images))
        #print (text)

    return vid_splits, img_splits, all_patient_info, all_images 
    
    
def dice(y_true, y_pred):
    smooth = 1.
    #print(y_true.shape, y_pred.shape)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return score

def batch_dice(y_true, y_pred):
    dice_sum = 0
    batch_size = y_true.shape[0]
    for batch_num in range(batch_size):        
        dice_sum +=  dice(y_true[batch_num,:,:], y_pred[batch_num,:,:])    
    return dice_sum
    
    
def hypothesis_test(y_true, y_pred, iou_threshold=0.5, area_threshold=225):
    smooth = 1.
    test = None    
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    pred_area = np.sum(y_pred_f)
    gt_area   = np.sum(y_true_f)
    
    intersection = np.sum(y_true_f * y_pred_f) 
    union        = gt_area + pred_area - intersection
    
    iou = (intersection + smooth)/(union + smooth)    
    
    if pred_area < area_threshold :    # too small prediction
        if gt_area ==  0: test = 'tn'  # Negative Data (Nothing in Ground Truth) 
        else: test = 'fn'              # Miss
    else:                              # Significant prediction
        if iou >= iou_threshold: test = 'tp'
        else: test = 'fp' 
                                                
    return test 
    
    
def evaluate(y_true, y_pred, joint = False): 
    y_true[y_true>=0.9] = 1
    y_true[y_true< 0.9] = 0                            
    y_pred[y_pred> 0] = 1   
    y_pred[y_pred<=0] = 0

    if joint == True:
        y_true_sc = y_true[:,0,:,:]
        y_pred_sc = y_pred[:,0,:,:]
        y_true_isc = y_true[:,1,:,:]
        y_pred_isc = y_pred[:,1,:,:]

        batch_dice_sc  = batch_dice(y_true_sc, y_pred_sc)
        batch_dice_isc = batch_dice(y_true_isc, y_pred_isc) 

        return batch_dice_sc, batch_dice_isc 
    else:
        return batch_dice(y_true, y_pred)
        
        
    
