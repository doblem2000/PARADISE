import os
import numpy as np
import pandas as pd
import torch
import csv

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def print_data(path, type='train/loss'):
    event_acc = EventAccumulator(path)
    print(event_acc.Reload())
    print(event_acc.Tags())
    for e in event_acc.Scalars(type):
        print(e.step, e.value)


def write_csv(dir,fold,headers,data):
    print("Writing fold",fold,"to csv in: ",dir)
    with open(dir+"fold"+str(fold)+".csv", 'w') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)

if __name__ == '__main__':
    steps_per_epoch=9
    experiment=["MobileNetV3Small_exp14_1000epoch_10fold_3segment_1frampersegment_batchsize32_optSGD",
                "ResNet50_exp13_1000epoch_10fold_3segment_1frampersegment_batchsize32_optSGD",
                "ResNet18_exp17_1000epoch_10fold_3segment_1frampersegment_batchsize32_optSGD"]
    path = ["runs/MobileNetV3Small_exp14_1000epoch_10fold_3segment_1frampersegment_batchsize32_optSGD/events.out.tfevents.1689690865.MICHELE-DELL.178023.0",
            "runs/ResNet50_exp13_1000epoch_10fold_3segment_1frampersegment_batchsize32_optSGD/events.out.tfevents.1689689667.6f949c9a8046.2006388.0",
            "runs/ResNet18_exp17_1000epoch_10fold_3segment_1frampersegment_batchsize32_optSGD/events.out.tfevents.1689699687.MICHELE-DELL.294463.0"]
    #print_data(path)
    fields = ['Step', 'Value']
    graph_types = ['train/loss', 'train/acc', 'val/loss', 'val/acc']
    
    event_acc = EventAccumulator(path)
    print(event_acc.Reload())
    print(event_acc.Tags())
    
    
    
    for i in range(len(path)):
        for graph_type in graph_types:
            dir = "csv/"+ experiment[i] +"/"+ graph_type + "/"
            os.makedirs(dir,exist_ok=True)
            #train_loss = create_csv(path[i],dir,fields,graph_type,steps_per_epoch=9)
            flag = False
            train_loss = list()
            fold = -1
            if type == 'val/loss' or type == 'val/acc':
                #print(type)
                flag = True
            
            for e in event_acc.Scalars(type):
                step = e.step
                value = e.value
                # if flag:
                #     print(e.step, e.value)
                if step == 0 or (flag == True and step == (steps_per_epoch-1)):
                    fold += 1
                    if fold != 0:
                        write_csv(dir,fold,fields,train_loss)
                        train_loss.clear()
        
                train_loss.append({fields[0]: step, fields[1]: value})
            write_csv(dir,fold+1,fields,train_loss)
    