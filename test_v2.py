from PIL import Image
import cv2, os, argparse, random
import numpy as np
import torch
from torch.nn import Linear,Sequential,Dropout
import albumentations
from VideoFrameDataset import ImglistOrdictToTensor
from torchvision import transforms
from models import *
import time
#from firenet import FireNet,build_FireNet
#from firenetV2 import FireNetV2


##### CODICE PROFESSORE #####
def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--videos", type=str, default='foo_videos/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results/', help="Results folder")
    args = parser.parse_args()
    return args
##### FINE CODICE PROFESSORE #####

args = init_parameter()


### TODO: # Here you should initialize your method
WEIGHT_PATH = 'ResNet50_exp10_2000epoch_5fold_5segment_1frampersegment_batchsize32/fold_0_best_model.pth'
MIN_DURATION = 10
THRESHOLD = 0.5


#####  MODEL CREATION #####


### TODO: caricare il modello per il test !!!!!!!!!!!!!!!!!!!!
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = build_ResNet50(1)
model.load_state_dict(torch.load(WEIGHT_PATH,map_location=device))
#model = build_FireNet()



model = model.cuda() if torch.cuda.is_available() else model
model.eval()

print('Videos folder: {}'.format(args.videos),"Current working directory: ", os.getcwd())
num_videos: int = len(os.listdir(args.videos))
print('Numero di video: {}'.format(num_videos))

total_frames: int = 0
total_time = 0
num_frames: int = 0
num_imgs: int = 0
start_frame_fire = None
fire_duration: int = 0
video_prediction = None
################################################



### CODICE PROFESSORE
for video in os.listdir(args.videos):
    print(video)
    ret = True
    cap = cv2.VideoCapture(os.path.join(args.videos, video))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    while ret:
        ret, img = cap.read()
### FINE CODICE PROFESSORE
        start_time = time.time()
        if ret == True:
            if num_frames % fps == 0: # Prendo un frame ogni secondo
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                ##### PREPROCESSING IMAGES #####
                #TODO: dai video si estraggono le immagini 
                #### Firenet
                # transform = albumentations.Compose([
                #     albumentations.Resize(height=64, width=64, interpolation=1, always_apply=True),
                #     albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                #                             std=[0.229, 0.224, 0.225],
                #                             max_pixel_value=255.,
                #                             always_apply=True),
                # ])
                preprocessing = albumentations.Sequential([
                    albumentations.Resize(height=224, width=224, interpolation=1, always_apply=True),
                    albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225],
                                 max_pixel_value=255.,
                                 always_apply=True),
                ])
                ################################
                img = transform(image=img)["image"]
                
                with torch.no_grad():
                    
                    input_batch = transforms.functional.to_tensor(img).unsqueeze(0) # create a mini-batch as expected by the model        
                    if torch.cuda.is_available():
                        input_batch = input_batch.to('cuda')
                        
                
                    output = model(input_batch)
                
                ###################### 
                #prediction = FireNetV2.compute_output(output[0])
                prediction = torch.nn.functional.sigmoid(output[0]) 
                ######################
               

                if prediction >= THRESHOLD: # Fire detected in the current frame
                    #print("prediction >= THRESHOLD: ", num_imgs)
                    if start_frame_fire is None:
                        start_frame_fire = num_imgs
                    fire_duration += 1
                    #print("prediction >= THRESHOLD. fire_duration: ",fire_duration, "start_frame_fire: ", start_frame_fire)
                else: # No fire detected in the current frame
                    #print("prediction < THRESHOLD. prediction: ",prediction)
                    fire_duration = 0
                    start_frame_fire = None

                if fire_duration >= MIN_DURATION: # Fire detected for at least MIN_DURATION seconds
                    #print("Fire detected for at least MIN_DURATION seconds. start_frame_fire: ",start_frame_fire,"num_imgs" ,num_imgs)

                    #### INSERIRE CODICE PER SCRIVERE I RISULTATI ####
                    img = None
                    num_imgs += 1
                    video_prediction = 1
                    break
                        
                num_imgs += 1
            num_frames += 1
            img = None
            
        


    end_time = time.time()
    ### CODICE PROFESSORE
    cap.release()
    f = open(args.results+video+".txt", "w")
    ### FINE CODICE PROFESSORE 

    #print("video",video,"num_imgs",num_imgs)
    ### CODICE AGGIUNTO
    # Here you should add your code for writing the results
    if video_prediction is None and start_frame_fire == 0 and prediction >= THRESHOLD: 
        # If the fire video is shorter than MIN_DURATION, 
        # we consider it as a fire video if: the first frame is classified as fire (start_frame_fire == 0 ) and the last frame is also classified as fire. 
        # This means that all frames of the video are classified as fire.
        video_prediction = 1
    



    if video_prediction:
        t = int(start_frame_fire)
        f.write(str(t))

    #print(torch.cuda.memory_summary(device=None, abbreviated=False))
    
    total_frames += num_imgs
    total_time += end_time-start_time
    end_time = 0
    start_time = 0

    num_frames = 0
    num_imgs = 0
    start_frame_fire = None
    fire_duration = 0
    prediction = None
    video_prediction = None
    ########################################################
    f.close()

print("Total frames: ", total_frames)
print("Total time: ", total_time)



