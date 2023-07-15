from PIL import Image
import cv2, os, argparse, random
import numpy as np
import torch
from torch.nn import Linear,Sequential,Dropout
import albumentations
from VideoFrameDataset import ImglistOrdictToTensor
from torchvision import transforms

def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--videos", type=str, default='foo_videos/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results/', help="Results folder")
    args = parser.parse_args()
    return args


### TODO: CODICE AGGIUNTO
def classify_video(predictions, threshold=0.5, min_duration=10):
  video_length = len(predictions)
  start_frame = None
  fire_duration = 0

  for frame_idx in range(video_length):
    if predictions[frame_idx] >= threshold:
      if start_frame is None:
        start_frame = frame_idx
      fire_duration += 1
    else:
      if fire_duration >= min_duration:
        return 1, start_frame  

      start_frame = None
      fire_duration = 0

  if fire_duration >= min_duration:
    return 1, start_frame  
  else:
    return 0, None  
################################################

args = init_parameter()

### TODO: CODICE AGGIUNTO 
# Here you should initialize your method
WEIGHT_PATH = 'MobileNetV2_exp3_1000epoch_10fold/fold_4_best_model.pth'
MIN_DURATION = 5
THRESHOLD = 0.5

##### CREAZIONE DEL MODELLO #####
### TODO: caricare il modello scelto da noi !!!!!!!!!!!!!!!!!!!!!
def build_MobileNet(num_outputs=1):
  model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
  model.classifier = Sequential(Dropout(p=0.2, inplace=False),Linear(in_features=1280, out_features=num_outputs, bias=True))
  return model

model = build_MobileNet()
model.load_state_dict(torch.load(WEIGHT_PATH))
model = model.cuda() if torch.cuda.is_available() else model
model.eval()
########### FINE CODICE NOSTRO AGGIUNTO

print('Videos folder: {}'.format(args.videos),"Current working directory: ", os.getcwd())
num_videos: int = len(os.listdir(args.videos))
print('Numero di video: {}'.format(num_videos))

num_frames: int = 0
frames = []
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

        ### CODICE AGGIUNTO
        # Here you should add your code for applying your method
        if ret == True:
          if num_frames % fps == 0: # Prendo un frame ogni secondo
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            ##### PREPROCESSING IMAGES #####
            #TODO: dai video si estraggono le immagini 
            transform = albumentations.Compose([
                albumentations.Resize(height=224, width=224, interpolation=1, always_apply=True),
                albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225],
                                        max_pixel_value=255.,
                                        always_apply=True),
            ])
            img = transform(image=img)["image"]

            ########################################################

            frames.append(img)
            img = None
          num_frames += 1
        else:
          #
          frames_old = frames
          #
          frames_tensor = ImglistOrdictToTensor.forward(frames)
          frames_tensor = frames_tensor.cuda() if torch.cuda.is_available() else frames_tensor
          frames = []
          print("frames_tensor.size()",frames_tensor.size())
        ########################################################



    ### CODICE PROFESSORE
    cap.release()
    f = open(args.results+video+".txt", "w")
    ### FINE CODICE PROFESSORE 


    ### CODICE AGGIUNTO
    # Here you should add your code for writing the results
    frames_predictions = {}
    with torch.no_grad():
      model.eval()
      frame_predictions = model(frames_tensor)

      for (id,frame) in enumerate(frames_old):
        #print(id,frame)
        input_batch = transforms.functional.to_tensor(frame).unsqueeze(0) # create a mini-batch as expected by the model
        if torch.cuda.is_available():
          input_batch = input_batch.to('cuda')
          model.to('cuda')
        frames_predictions[id] = model(input_batch)

      for (id,_) in enumerate(frames_old):
        print(frame_predictions[id] ,frames_predictions[id])
        
      #print(type(frame_predictions))
      #print("frame_predictions: ",frame_predictions)


    min_duration = MIN_DURATION if len(frames_predictions) >= MIN_DURATION else frame_predictions.size(0)
    prediction, start_frame = classify_video(frames_predictions, threshold=THRESHOLD, min_duration=min_duration)
    #min_duration = MIN_DURATION if frame_predictions.size(0) >= MIN_DURATION else frame_predictions.size(0)
    #prediction, start_frame = classify_video(frame_predictions, threshold=THRESHOLD, min_duration=min_duration)

    if prediction:
        t = int(start_frame)
        f.write(str(t))

    torch.cuda.memory_summary(device=None, abbreviated=False)
    num_frames = 0
    frames = []
    frames_tensor = None
    prediction = None
    ########################################################
    f.close()



