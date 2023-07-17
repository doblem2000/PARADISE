import os
os.system('pip install striprtf ')
os.system('pip install torchinfo ')

import torchinfo

import os
import os.path
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from typing import List, Union, Tuple, Any
from striprtf.striprtf import rtf_to_text
import albumentations


# ha le informazioni legate ad  ogni video, i metadati.
class VideoRecord(object):
    """
    Helper class for class VideoFrameDataset. This class
    represents a video sample's metadata.

    Args:
        root_datapath: the system path to the root folder of the videos.
        row: A list with four or more elements where
             1) The first element is the path to the video sample's frames excluding
             the root_datapath prefix
             2) The  second element is the starting frame id of the video
             3) The third element is the inclusive ending frame id of the video
             4) The fourth element is the label index.
             5) any following elements are labels in the case of multi-label classification
    """
    def __init__(self, row, root_datapath):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])

    @property
    def path(self) -> str:
        return self._path

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1  # +1 because end frame is inclusive

    @property
    def start_frame(self) -> int:
        return int(self._data[1])

    @property
    def end_frame(self) -> int:
        return int(self._data[2])

    @property
    def label(self) -> Union[int, List[int]]:
        # just one label_id
        if len(self._data) == 4:
            return int(self._data[3])
        # sample associated with multiple labels
        else:
            return [int(label_id) for label_id in self._data[3:]]

# Il parametro test_mode serve per rendere non aleatoria l'estrazione dei frame dal segmento, ovvero prendere sempre gli stessi frame serve per la validation
class VideoFrameDataset(torch.utils.data.Dataset):
    r"""
    A highly efficient and adaptable dataset class for videos.
    Instead of loading every frame of a video,
    loads x RGB frames of a video (sparse temporal sampling) and evenly
    chooses those frames from start to end of the video, returning
    a list of x PIL images or ``FRAMES x CHANNELS x HEIGHT x WIDTH``
    tensors.

    More specifically, the frame range [START_FRAME, END_FRAME] is divided into NUM_SEGMENTS
    segments and FRAMES_PER_SEGMENT consecutive frames are taken from each segment.

    Note:
        A demonstration of using this class can be seen
        in ``demo.py``
        https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch

    Note:
        This dataset broadly corresponds to the frame sampling technique
        introduced in ``Temporal Segment Networks`` at ECCV2016
        https://arxiv.org/abs/1608.00859.

    Args:
        root_path: The root path in which video folders lie.
                   this is ROOT_DATA from the description above.
        num_segments: The number of segments the video should
                      be divided into to sample frames from.
        frames_per_segment: The number of frames that should
                            be loaded per segment. For each segment's
                            frame-range, a random start index or the
                            center is chosen, from which frames_per_segment
                            consecutive frames are loaded.
        imagefile_template: The image filename template that video frame files
                            have inside of their video folders as described above.
        transform: Transform pipeline that receives a list of numpy images/frames.
        test_mode: If True, frames are taken from the center of each
                   segment, instead of a random location in each segment.

    """
    def __init__(self,
                 root_path: str,
                 num_segments: int = 3,
                 frames_per_segment: int = 1,
                 imagefile_template: str='{:05d}.jpg',
                 transform=None,
                 totensor=True,
                 test_mode: bool = False):
        super(VideoFrameDataset, self).__init__()

        self.root_path = root_path
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template
        self.test_mode = test_mode

        if transform is None:
            self.transform = None
        else:
            additional_targets = {}
            for i in range(self.num_segments * self.frames_per_segment - 1):
                additional_targets["image%d" % i] = "image"
            self.transform = albumentations.Compose([transform],
                                                    additional_targets=additional_targets,
                                                    p=1)
        self.totensor = totensor
        self.totensor_transform = ImglistOrdictToTensor()

        self._parse_annotationfile()
        self._sanity_check_samples()

    def _load_image(self, directory: str, idx: int) -> Image.Image:
        return np.asarray(Image.open(os.path.join(directory, self.imagefile_template.format(idx))).convert('RGB'))

    def _parse_annotationfile(self):
        self.video_list = []
        for class_name in os.listdir(self.root_path):
            for video_name in os.listdir(os.path.join(self.root_path, class_name)):
                frames_dir = os.path.join(self.root_path, class_name, video_name)
                if os.path.isdir(frames_dir):
                    frame_path = os.path.join(class_name, video_name)
                    end_frame = len(os.listdir(frames_dir))

                    annotation_path = frames_dir\
                        .replace("\\", "/") \
                        .replace("FRAMES/", "GT/") \
                        .replace(".mp4", ".rtf")

                    with open(annotation_path, 'r') as file:
                        text = rtf_to_text(file.read())
                    if len(text):
                        label = 1
                        start_frame = int(text.split(",")[0])
                        if start_frame == 0:
                          start_frame = 1
                    else:
                        label = 0
                        start_frame = 1

                    self.video_list.append(VideoRecord(
                        [frame_path, start_frame, end_frame, label],
                        self.root_path))

    def _sanity_check_samples(self):
        for record in self.video_list:
            if record.num_frames <= 0 or record.start_frame == record.end_frame:
                print(f"\nDataset Warning: video {record.path} seems to have zero RGB frames on disk!\n")

            elif record.num_frames < (self.num_segments * self.frames_per_segment):
                print(f"\nDataset Warning: video {record.path} has {record.num_frames} frames "
                      f"but the dataloader is set up to load "
                      f"(num_segments={self.num_segments})*(frames_per_segment={self.frames_per_segment})"
                      f"={self.num_segments * self.frames_per_segment} frames. Dataloader will throw an "
                      f"error when trying to load this video.\n")

    def _get_start_indices(self, record: VideoRecord) -> 'np.ndarray[int]':
        """
        For each segment, choose a start index from where frames
        are to be loaded from.

        Args:
            record: VideoRecord denoting a video sample.
        Returns:
            List of indices of where the frames of each
            segment are to be loaded from.
        """
        # choose start indices that are perfectly evenly spread across the video frames.
        if self.test_mode:
            distance_between_indices = (record.num_frames - self.frames_per_segment + 1) / float(self.num_segments)

            start_indices = np.array([int(distance_between_indices / 2.0 + distance_between_indices * x)
                                      for x in range(self.num_segments)])
        # randomly sample start indices that are approximately evenly spread across the video frames.
        else:
            max_valid_start_index = (record.num_frames - self.frames_per_segment + 1) // self.num_segments

            start_indices = np.multiply(list(range(self.num_segments)), max_valid_start_index) + \
                      np.random.randint(max_valid_start_index, size=self.num_segments)

        return start_indices

    def __getitem__(self, idx: int) -> Union[
        Tuple[List[Image.Image], Union[int, List[int]]],
        Tuple['torch.Tensor[num_frames, channels, height, width]', Union[int, List[int]]],
        Tuple[Any, Union[int, List[int]]],
        ]:
        """
        For video with id idx, loads self.NUM_SEGMENTS * self.FRAMES_PER_SEGMENT
        frames from evenly chosen locations across the video.

        Args:
            idx: Video sample index.
        Returns:
            A tuple of (video, label). Label is either a single
            integer or a list of integers in the case of multiple labels.
            Video is either 1) a list of PIL images if no transform is used
            2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1]
            if the transform "ImglistToTensor" is used
            3) or anything else if a custom transform is used.
        """
        record: VideoRecord = self.video_list[idx]

        frame_start_indices: 'np.ndarray[int]' = self._get_start_indices(record)

        return self._get(record, frame_start_indices)

    def _get(self, record: VideoRecord, frame_start_indices: 'np.ndarray[int]') -> Union[
        Tuple[List[Image.Image], Union[int, List[int]]],
        Tuple['torch.Tensor[num_frames, channels, height, width]', Union[int, List[int]]],
        Tuple[Any, Union[int, List[int]]],
        ]:
        """
        Loads the frames of a video at the corresponding
        indices.

        Args:
            record: VideoRecord denoting a video sample.
            frame_start_indices: Indices from which to load consecutive frames from.
        Returns:
            A tuple of (video, label). Label is either a single
            integer or a list of integers in the case of multiple labels.
            Video is either 1) a list of PIL images if no transform is used
            2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1]
            if the transform "ImglistToTensor" is used
            3) or anything else if a custom transform is used.
        """

        frame_start_indices = frame_start_indices + record.start_frame
        images = list()

        # from each start_index, load self.frames_per_segment
        # consecutive frames
        for start_index in frame_start_indices:
            frame_index = int(start_index)

            # load self.frames_per_segment consecutive frames
            for _ in range(self.frames_per_segment):
                image = self._load_image(record.path, frame_index)
                images.append(image)

                if frame_index < record.end_frame:
                    frame_index += 1

        if self.transform is not None:
            transform_input = {"image": images[0]}
            for i, image in enumerate(images[1:]):
                transform_input["image%d" % i] = image
            images = self.transform(**transform_input)

        if self.totensor:
            images = self.totensor_transform(images)
        return images, record.label

    def __len__(self):
        return len(self.video_list)


class ImglistOrdictToTensor(torch.nn.Module):
    """
    Converts a list or a dict of numpy images to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH).
    Can be used as first transform for ``VideoFrameDataset``.
    """
    @staticmethod
    def forward(img_list_or_dict):
        """
        Converts each numpy image in a list or a dict to
        a torch Tensor and stacks them into a single tensor.

        Args:
            img_list_or_dict: list or dict of numpy images.
        Returns:
            tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        if isinstance(img_list_or_dict, list):
            return torch.stack([transforms.functional.to_tensor(img)
                                for img in img_list_or_dict])
        else:
            return torch.stack([transforms.functional.to_tensor(img_list_or_dict[k])
                                for k in img_list_or_dict.keys()])
from torch.utils.data import Subset, DataLoader

# Function for the K-fold Cross Validation
def cross_val_dataloaders(train_dataset, val_dataset=None, K=10, batch_size=32):
  if val_dataset is None:
    val_dataset = train_dataset

  indexes = torch.randperm(len(train_dataset)) % K

  dataloader_params = {"batch_size": batch_size, "num_workers": 2, "pin_memory": True}

  train_folds, val_folds = [], []
  for k in range(K):

      val_fold   = Subset(val_dataset,   (indexes==k).nonzero().squeeze())
      train_fold = Subset(train_dataset, (indexes!=k).nonzero().squeeze())

      #print("train_fold: ", len(train_fold))

      val_fold   = DataLoader(val_fold,   shuffle=False, **dataloader_params)
      train_fold = DataLoader(train_fold, shuffle=True,  **dataloader_params)

      val_folds.append(val_fold)
      train_folds.append(train_fold)
  return train_folds, val_folds, indexes

from torch.utils.tensorboard import SummaryWriter
from tensorboard import notebook

def start_tensorboard(log_dir):
  writer = SummaryWriter(os.path.join("runs", log_dir))

  # run tensorboard in background
  
  os.system('killall tensorboard ')
  #%load_ext tensorboard ################################################
  #%tensorboard --logdir ./runs###############################################

  notebook.list() # View open TensorBoard instances

  return writer

from torchvision.utils import make_grid
from tqdm import tqdm

def one_epoch(model, lossFunction, output_activation, optimizer, train_loader, val_loader, writer, epoch_num, device):
  model.train()

  i_start = epoch_num * len(train_loader)
  for i, (X, y) in tqdm(enumerate(train_loader), desc="epoch {} - train ".format(epoch_num)):
  #for i, (X, y) in enumerate(train_loader):
    (batch_size, frames, channels, width, height) = X.shape
    #print("\nX prima: ",X.shape)
    #print("y prima: ",y.shape)
    #pippo=X[0][0]
    #pippo2=X[0][1]
    #pippo3=X[0][2]
    #print(X[0][0])

    X = X.view(-1,channels, width, height)
    #y = y.repeat(3).long()#.float()   # oppure y.repeat_interleave(frames).long()
    y = y.repeat_interleave(frames)#.float()#.long()  #TODO: Capire .float() e .long()
    #print("X dopo: ",X.shape)
    #print("y dopo: ",y.shape)
    #print(X[0])
    #print("Primo tensore")
    #print(pippo==X[0])
    #print("Secondo tensore")
    #print(pippo2==X[1])
    #print("Terzo tensore")
    #print(pippo3==X[2])

    if i == 0:
      writer.add_image('first_batch', make_grid(X))

    X = X.cuda()#.to(device)###################################
    y = y.cuda().float()#.long()##############################

    optimizer.zero_grad()
    o = model(X)
    o = output_activation(o).squeeze()
    #print("o=",o,"shape o", o.shape, "y=", y, "shape y", y.shape)
    l = lossFunction(o, y)

    l.backward()
    optimizer.step()

    #print("o.detach()", o.detach())
    #print("y.detach()", y.detach())
    acc = ((o.detach() > .5) == y.detach()).float().mean()
    print("acc", acc)
    #acc = (o.detach().argmax(-1) == y.detach()).float().mean()

    print("- batch loss and accuracy : {:.7f}\t{:.4f}".format(l.detach().item(), acc))
    writer.add_scalar('train/loss', l.detach().item(), i_start+i)
    writer.add_scalar('train/acc', acc, i_start+i)

  model.eval()
  print("\nVALIDATION FASE\n")
  #print("val_loader", val_loader)

  with torch.no_grad():
    val_loss = []
    val_corr_pred = []
    for X, y in tqdm(val_loader, desc="epoch {} - validation".format(epoch_num)):
    #for X, y in val_loader:
      #print("X:", X," y:", y)
      (batch_size, frames, channels, width, height) = X.shape
      X = X.view(-1,channels, width, height)
      y = y.repeat_interleave(frames).float()#.long()
      #y = y.repeat(3).long()#.float()
      #print("X:", X," y:", y)

      X = X.cuda()#.to(device)#cuda() ####################
      y = y.cuda()#.to(device)#cuda().float()#######################

      o = model(X)
      o = output_activation(o).squeeze()
      val_loss.append(lossFunction(o, y))
      #print("o=",o," y=", y)
      val_corr_pred.append((o > .5) == y)
      #val_corr_pred.append(o.argmax(-1) == y)

    val_loss = torch.stack(val_loss).mean().item()
    val_accuracy = torch.concatenate(val_corr_pred).float().mean().item()

    print("Validation loss and accuracy : {:.7f}\t{:.4f}".format(val_loss, val_accuracy))
    writer.add_scalar('val/loss', val_loss, i_start+i)
    writer.add_scalar('val/acc', val_accuracy, i_start+i)
  return val_loss, val_accuracy



from torch.nn import Linear,Sequential,Dropout
from models import *

model = build_MobileNetV3Small(1)
model=network_parameters_MobileNetV3Small(model)

print(torchinfo.summary(model, ####################################################32 batch size da mettere a run time
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
))


preprocessing = albumentations.Sequential([
        albumentations.Resize(height=224, width=224, interpolation=1, always_apply=True),
        albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225],
                                 max_pixel_value=255.,
                                 always_apply=True),
    ])

augmentation = albumentations.OneOf([
        albumentations.HorizontalFlip(p=1.),
        ], p=.5)

dataset = VideoFrameDataset(root_path="FRAMES/TRAINING_SET/",
                                num_segments=3,
                                frames_per_segment=1,
                                transform=albumentations.Compose([
                                    preprocessing,
                                    augmentation],
                                    p=1.,
                                )
                                )

# Creazione K fold e relativi dataloader
K_cross_val = 5
batch_size = 32
train_folds, val_folds, indexes = cross_val_dataloaders(dataset, dataset, K_cross_val, batch_size)

# Define loss and optimizer
output_activation=torch.nn.Sigmoid()  # Sostituire con la relu
#output_activation=torch.nn.ReLU()  #########################################################
lossFunction = torch.nn.BCELoss()    # Nel caso di output_size del modello == 1
#lossFunction = torch.nn.CrossEntropyLoss()  # Nel caso di output_size del modello > 1
lr=0.001
momentum = 0.9
lambda_reg = 0

epochs = 1000
early_stopping_patience = 40

# create output directory and logger
experiment_name = "MobileNetV3Small_exp12_1000epoch_5fold_3segment_1frampersegment_batchsize32"

optimizer_config = optimizer_settings_MobileNetV3Small(model, lr, lambda_reg, momentum)

dirs = os.listdir()

if experiment_name not in dirs:
  os.makedirs(experiment_name)

writer = start_tensorboard(experiment_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Start the timer
from timeit import default_timer as timer
#start_time = timer()

# Setup training and save the results
#####################################################
torch.save(indexes, os.path.join(experiment_name, "cross-val-indexes.pt"))

val_losses = torch.zeros(epochs, K_cross_val)
val_accuracies = torch.zeros(epochs, K_cross_val)

for k in range(K_cross_val):
  model = build_MobileNetV3Small(1)
  model = network_parameters_MobileNetV3Small(model)
  model.cuda()#.cpu() ######################################
  # dataloader, network, optimizer for each fold
  train_loader, val_loader = train_folds[k], val_folds[k]
  optimizer = optimizer_config
  #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  # early stopping and best model saving
  early_stopping_counter = early_stopping_patience
  min_val_loss = 1e10

  # training and validation
  for e in range(epochs):
    print("FOLD {} - EPOCH {}".format(k, e))

    val_loss, val_accuracy = one_epoch(model, lossFunction, output_activation, optimizer, train_loader, val_loader, writer, e, device)

    # store the validation metrics
    val_losses[e, k] = val_loss
    val_accuracies[e, k] = val_accuracy
    torch.save(val_losses, os.path.join(experiment_name,'val_losses.pth'))
    torch.save(val_accuracies, os.path.join(experiment_name,'val_accuracies.pth'))

    # save the best model and check the early stopping criteria
    if val_loss < min_val_loss: # save the best model
      min_val_loss = val_loss
      early_stopping_counter = early_stopping_patience # reset early stopping counter
      torch.save(model.state_dict(), os.path.join(experiment_name,'fold_{}_best_model.pth'.format(k)))
      print("- saved best model with val_loss =", val_loss, "and val_accuracy =", val_accuracy)

    if e>0: # early stopping counter update
      if val_losses[e, k] > val_losses[e-1, k]:
          early_stopping_counter -= 1 # update early stopping counter
      else:
          early_stopping_counter = early_stopping_patience # reset early stopping counter
    if early_stopping_counter == 0: # early stopping
        break

# End the timer and print out how long it took
#end_time = timer()
#print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")