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
    with open(dir+"fold"+str(fold)+".csv", 'w') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)

def create_csv(path,dir,fields,type='train/loss',epoch_per_folds=100,steps_per_epoch=9):
    event_acc = EventAccumulator(path)
    print(event_acc.Reload())
    print(event_acc.Tags())
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
        write_csv(dir,fold,fields,train_loss)
    

if __name__ == '__main__':
    path = "events.out.tfevents.1689777887.MICHELE-DELL.3357625.0"
    #print_data(path)
    fields = ['Step', 'Value']
    graph_types = ['train/loss', 'train/acc', 'val/loss', 'val/acc']
    #print_data(path,graph_types[2])
    #print_data(path,graph_types[3])
    for graph_type in graph_types:
        dir = "csv/small_train/" + graph_type + "/"
        os.makedirs(dir,exist_ok=True)
        train_loss = create_csv(path,dir,fields,graph_type,epoch_per_folds=100,steps_per_epoch=9)
    
    
    #print(train_loss)
    #print(train_loss.keys(),len(train_loss['fold0']),len(train_loss['fold1']),len(train_loss['fold2']))
    #to_csv(path)