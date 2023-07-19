import os
import numpy as np
import pandas as pd
import torch
import csv

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def print_data(path):
    event_acc = EventAccumulator(path)
    print(event_acc.Reload())
    print(event_acc.Tags())
    for e in event_acc.Scalars('train/loss'):
        print(e.step, e.value)


def write_csv(dir,fold,headers,data):
    with open(dir+"fold"+str(fold), 'w') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)

def create_csv(path,dir,fields,type='train/loss'):
    event_acc = EventAccumulator(path)
    print(event_acc.Reload())
    print(event_acc.Tags())
    train_loss = list()
    fold = -1
    #train_loss.update({'fold{fold}':list()})
    for e in event_acc.Scalars('train/loss'):
        step = e.step
        value = e.value
        #print(e.step, e.value)
        if step == 0:
            fold += 1
            if fold != 0:
                write_csv(dir,fold,fields,train_loss)
                train_loss = list()
            
        train_loss.append({fields[0]: step, fields[1]: value})
    

if __name__ == '__main__':
    path = "runs/MobileNetV3Small_exp18_10epoch_10fold_3segment_1frampersegment_batchsize32_optSGD/events.out.tfevents.1689777887.MICHELE-DELL.3357625.0"
    #print_data(path)
    fields = ['Step', 'Value']
    graph_types = ['train/loss','train/accuracy','val/loss','val/accuracy']
    for graph_type in graph_types:
        dir = "csv/MobileNetV3Small_exp18_10epoch_10fold_3segment_1frampersegment_batchsize32_optSGD/" + graph_type + "/"
        os.makedirs(dir,exist_ok=True)
        train_loss = create_csv(path,dir,fields,graph_type)
    
    #print(train_loss)
    #print(train_loss.keys(),len(train_loss['fold0']),len(train_loss['fold1']),len(train_loss['fold2']))
    #to_csv(path)