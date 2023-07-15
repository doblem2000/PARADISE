import torch

def build_MobileNetV3Small(num_outputs=1):
  model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_small', pretrained=True)
  model.classifier = torch.nn.Sequential(
    torch.nn.Linear(in_features=576,
                    out_features=1024, # same number of output units as our number of classes
                    bias=True),
    torch.nn.Hardswish(inplace=True),
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1024,
                    out_features=num_outputs, # same number of output units as our number of classes
                    bias=True))
  return model

def build_MobileNetV2(num_outputs=1):
  model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
  model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.2, inplace=False),
                                         torch.nn.Linear(in_features=1280, out_features=num_outputs, bias=True))
  return model