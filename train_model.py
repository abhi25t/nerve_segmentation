import os
import cv2
import csv
import PIL
import time
import copy
import torch
import random

import numpy as np
import pandas as pd
import torchvision.transforms as T

from torch.nn.functional import binary_cross_entropy_with_logits as BCE_logit_Loss
from torch.utils.data    import Dataset, DataLoader
from torchvision         import models
from collections         import defaultdict
from tqdm.auto           import tqdm
from glob                import glob
from matplotlib          import pyplot as plt

from utils          import isnotebook, evaluate, live_plot 
from configurations import EXPERIMENT_NAME, DATA_DIR, BATCH_SIZE, NUM_WORKERS, NUM_EPOCHS, FOLDS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class NerveSegDataset(Dataset):
    """Segmentation Dataset"""
    def __init__(self, image_paths, all_annotations, patient_info, train=False):
        self.image_paths     = image_paths
        self.train           = train
        self.all_annotations = all_annotations
        self.patient_info    = patient_info

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]  
        image = PIL.Image.open(img_path)  
        
        video_name = os.path.basename(img_path)[:4]
        frame_num = str(int(os.path.basename(img_path)[5:8]))
        if frame_num in self.all_annotations[video_name]:
            frame_annotations = self.all_annotations[video_name][frame_num]
            tracker = frame_annotations['tracker']
            bounding_boxes = frame_annotations['bounding_boxes']
        else:
            tracker, bounding_boxes = None, None
        mask = np.zeros((image.size[1],image.size[0]), dtype=np.float32)
        if tracker is not None:        
            bounding_boxes = eval(bounding_boxes)            
            if (bounding_boxes is not None) and (len(bounding_boxes) > 0):
                for box in bounding_boxes:
                    (x, y, w, h) = [int(v) for v in box]
                    mask[y:y+h, x:x+w] = 255.        
        
        mask = cv2.resize(mask, (256, 256))          
        mask = PIL.Image.fromarray(mask)
        image = T.Resize((256,256))(image)  
        
        if self.train == True:    # Augmentations

            if random.random()<0.5:   # Horizontal Flip
                image = T.functional.hflip(image)
                mask  = T.functional.hflip(mask)
 
            if random.random()<0.25:  # Rotation
                rotation_angle = random.randrange(-10,11)
                image = T.functional.rotate(image,angle = rotation_angle)
                mask  = T.functional.rotate(mask ,angle = rotation_angle)
   
            if random.random()<0.3:   # Gamma Intensity
                gain = self.patient_info.loc[self.patient_info.vid_name == int(video_name),'gain'].iloc[0]
                if gain == 'H':
                    adj_gamma=random.uniform(1.5,2.0)
                elif gain == 'L':                    
                    adj_gamma=random.uniform(0.5,0.75)
                elif gain == 'M':
                    adj_gamma=random.uniform(0.75,1.33)
                image = T.functional.adjust_gamma(image,gamma=adj_gamma,gain=1)
     
        nerve = self.patient_info.loc[self.patient_info.vid_name == int(video_name),'nerve'].iloc[0]         
        mask = np.array(mask)       
        
        image = T.ToTensor()(image)             
        mask  = T.ToTensor()(mask) 

        return {'image': image, 'mask': mask, 'nerve': nerve}  
        
        
def check_models_folder(models_folder): 
    
    if not os.path.exists(models_folder):
        print('Creating folder for models at: ', models_folder)
        os.makedirs(models_folder, exist_ok=False)
    else:
        print('Models folder : ', models_folder)     
    
    models_found = glob(os.path.join(models_folder, EXPERIMENT_NAME + '_best_of_fold_' + '*.pt')) 
    if len(models_found) == 0:
        print('No previous model found. Starting fresh') 
        return 1, 1, None
    
    last_model = os.path.join(models_folder, EXPERIMENT_NAME+'_last_model.pt')
    checkpoint = torch.load(last_model) 
    last_epoch = int(checkpoint["epoch"]) 
    
    models_found = glob(os.path.join(models_folder, EXPERIMENT_NAME + '_best_of_fold_' + '*.pt'))
    folds_found = [os.path.basename(f)[-4:-3] for f in models_found] 
    folds_found = [int(m) for m in folds_found]
    fold_num = max(folds_found)

    if (fold_num<FOLDS) and (last_epoch == NUM_EPOCHS):
        print('Resuming after Fold {}; Epoch {}'.format(fold_num,last_epoch))
        starting_fold = fold_num+1
        starting_epoch = 1
    elif last_epoch < NUM_EPOCHS:
        print('Resuming after Fold {}, Epoch {}'.format(fold_num,last_epoch))
        starting_fold = fold_num
        starting_epoch = last_epoch+1
    elif (fold_num==FOLDS) and (last_epoch == NUM_EPOCHS): 
        print('All {} folds completed'.format(fold_num))
        return None, None, None

    return starting_fold, starting_epoch, last_model 

        
def run_epoch(model, optimizer, positive_weights, metrics, best_loss, epoch, 
              fold_num, models_folder, model_time, metrics_csv_path, disp_epoch, 
              disp_phase, train_dataloader, val_dataloader, pos_weight=None):
    
    disp_epoch.update('Epoch {}/{}'.format(epoch, NUM_EPOCHS)) 
    for phase in ['Train', 'Val']:
        disp_phase.update('Phase: '+phase)

        epoch_loss    = 0.
        best_epoch    = 0
        total_samples = 0
        epoch_dice_sc  = 0

        if phase == 'Train':
            model.train()  # Set model to training mode
            dataloader = train_dataloader
        else:
            model.eval()   # Set model to evaluation mode
            dataloader = val_dataloader
        #num_batches = len(dataloader)      
        for sample in tqdm(iter(dataloader), leave=False):  # Iterate over batches.
            inputs = sample['image'].to(device)
            masks  = sample['mask'].clamp(min=0.,max=1.).to(device) 

            optimizer.zero_grad() 
            samples = inputs.shape[0]

            if samples >1:  # Due to a bug, training not possible with just one sample
                total_samples += samples 

                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = BCE_logit_Loss( outputs['out'].clamp(min=0.,max=1.), 
                                    # github.com/pytorch/pytorch/issues/2866#issuecomment-393242524
                                    masks, # .clamp(min=0.,max=1.) done already while reading
                                    reduction="mean", pos_weight=positive_weights)

                    epoch_loss += loss.item() 

                    y_true = masks.data.cpu().numpy()                                
                    y_pred = outputs['out'].data.cpu().numpy()

                    batch_dice_sc= evaluate(y_true, y_pred)
                    epoch_dice_sc  += batch_dice_sc 

                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

        dice_sc  = epoch_dice_sc /total_samples

        bce_loss = epoch_loss/total_samples 
        avg_dice = dice_sc 
        avg_loss = (bce_loss+(1-avg_dice))/2         
        metrics[phase +'_BCE_Loss'].append(bce_loss)
        metrics[phase +'_Loss'].append(avg_loss)

        if phase == 'Val':
            metrics['SC_Dice' ].append(dice_sc )
            if avg_loss < best_loss : 
                best_loss  = avg_loss
                best_model_wts = copy.deepcopy(model.state_dict())  

                if fold_num == -1:
                    model_name = EXPERIMENT_NAME + '_best_model.pt'
                else: model_name = EXPERIMENT_NAME + '_best_of_fold_' + str(fold_num) +".pt"
                
                model_path = os.path.join(models_folder, model_name)
                save = {'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best_loss': best_loss,
                        'loss': avg_loss,
                        'opt_dict': optimizer.state_dict() }
                torch.save(save, model_path)                    

    # Saving model after every epoch -- Deactivated
    #if fold_num == -1:
    #    model_name = EXPERIMENT_NAME + '_epoch_' + str(epoch) + model_time +".pt"
    #else: model_name = EXPERIMENT_NAME +'_fold_' + str(fold_num) + '_epoch_' + str(epoch) + model_time +".pt"

    model_name = EXPERIMENT_NAME + '_last_model.pt'  # Saving last model    
    model_path = os.path.join(models_folder, model_name)
    save = {'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss, 
            'loss': avg_loss,
            'opt_dict': optimizer.state_dict()  }
    torch.save(save, model_path)              
    metrics_df = pd.DataFrame.from_dict(metrics)  
    metrics_df.to_csv(metrics_csv_path, index=False)   
            
    return metrics, best_loss 
    
           
def k_fold_crossval(data_dir, vid_splits, img_splits, patient_info, all_annotations, median_areas, joint=False): 
    
    models_folder = os.path.join(data_dir, 'output' , 'models')       
    starting_fold, starting_epoch, last_model = check_models_folder(models_folder) 
    
    if starting_fold is None: 
        print('Ready for evaluations.')
        return models_folder
    
    start_time = time.time()

    for fold_num in range(starting_fold, FOLDS+1):
        img_split = img_splits[fold_num-1]
        vid_split = vid_splits[fold_num-1]
        print('=' * 25) 
        print('Fold {}/{}'.format(fold_num, len(img_splits)))  
        train_vids, val_vids, test_vids = vid_split['train'], vid_split['val'], vid_split['test'] 
        text = 'Train/Val/Test split of videos is {} / {} / {}'.format(
                                        len(train_vids),len(val_vids),len(test_vids))
        print (text)                
        train_images, val_images, test_images = img_split['train'], img_split['val'], img_split['test']
        text = 'Train/Val/Test split of images is {} / {} / {}'.format(
                                        len(train_images),len(val_images),len(test_images))
        print (text)        
        train_dataset = NerveSegDataset(train_images, all_annotations, patient_info, train=True)
        val_dataset   = NerveSegDataset(val_images  , all_annotations, patient_info) 

        train_dataloader = DataLoader(train_dataset,
                                    batch_size =BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_dataloader   = DataLoader(val_dataset, 
                                    batch_size =BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)        
        
        model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=1) 
        model.to(device)        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.33)
        
        if starting_epoch==1 : 
            best_loss      = 1e10   
        else:
            checkpoint = torch.load(last_model)
            model.load_state_dict(checkpoint['state_dict'])        
            optimizer.load_state_dict(checkpoint['opt_dict'])  # https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/2
            best_loss = checkpoint["best_loss"]   
            
        fold_start_time = time.time()
        model_time      = time.strftime('_%Y%m%d_%H%M', time.localtime())

        best_model_wts = copy.deepcopy(model.state_dict())

        if joint == True:
            positive_weights = torch.FloatTensor([1, 50]).reshape(1, 2, 1, 1) 
            positive_weights = positive_weights.to(device)
        else: positive_weights = None
        
        disp_text = display('', display_id=True)  
        fig = plt.figure(figsize=(12,5))
        disp_chart = display(fig, display_id=True)
        disp_epoch = display('', display_id=True)
        disp_phase = display('', display_id=True) 
        metrics_csv_path = os.path.join(data_dir, 'output', EXPERIMENT_NAME+'_fold_'+ str(fold_num) +'_metrics.csv') 
        
        if os.path.exists(metrics_csv_path):
            metrics_df = pd.read_csv(metrics_csv_path) 
            rowws = len(metrics_df) 
            dicto = metrics_df.to_dict()  
            metrics = {k: [g[x] for x in range(rowws)] for k,g in dicto.items() }  
        else: metrics = defaultdict(list)         
        best_epoch = 0
        for epoch in range(starting_epoch, NUM_EPOCHS+1) : 
            previous_best_loss = best_loss
            metrics, best_loss = run_epoch(model, optimizer, 
                        positive_weights, metrics, best_loss, epoch, fold_num, 
                        models_folder, model_time, metrics_csv_path, disp_epoch, 
                        disp_phase, train_dataloader, val_dataloader,pos_weight=positive_weights)
            if best_loss < previous_best_loss : 
                best_epoch = str(epoch)
            
            if isnotebook():    
                live_plot(metrics, disp_text, disp_chart)                 
            # Epoch ends        
        time_elapsed = (time.time() - fold_start_time)/60
        print('-'*30)
        print('Fold# {} completed in {:.0f} hr {:.0f} min'.format(fold_num, time_elapsed // 60, time_elapsed % 60))
        print('Lowest Loss in Epoch#{}: {:.4f}'.format(best_epoch, best_loss)) 
        starting_epoch = 1
        # Fold ends
    time_elapsed = (time.time() - start_time)/60
    print('='*50)
    print('k-fold completed in {:.0f} hr {:.0f} min'.format(time_elapsed//60, time_elapsed%60))  
    
    return models_folder 



    
