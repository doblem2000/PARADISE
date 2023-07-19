import os
import numpy as np
import pandas as pd
import torch
import csv

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tabulate_events(dpath):
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]

    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = []

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]

        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out, steps


def to_csv(dpath):
    dirs = os.listdir(dpath)
    print("dirs: ",dirs)
    d, steps = tabulate_events(dpath)
    print("d: ",d)
    print("steps: ",steps)
    tags, values = zip(*d.items())
    np_values = np.array(values)

    for index, tag in enumerate(tags):
        df = pd.DataFrame(np_values[index], index=steps, columns=dirs)
        df.to_csv(get_file_path(dpath, tag))


def get_file_path(dpath, tag):
    file_name = tag.replace("/", "_") + '.csv'
    folder_path = os.path.join(dpath, 'csv')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


def print_data(path):
    event_acc = EventAccumulator(path)
    print(event_acc.Reload())
    print(event_acc.Tags())
    for e in event_acc.Scalars('train/loss'):
        print(e.step, e.value)

def create_dict(path):
    event_acc = EventAccumulator(path)
    print(event_acc.Reload())
    print(event_acc.Tags())
    train_loss = dict()
    fold = -1
    #train_loss.update({'fold{fold}':list()})
    for e in event_acc.Scalars('train/loss'):
        #print(e.step, e.value)
        if e.step == 0:
            fold += 1
            #print(fold)
            train_loss["fold" + str(fold)] = list()
        train_loss['fold' + str(fold)].append(e.value)
    
    return train_loss


# field names 
fields = ['SNo', 'Name', 'Subject'] 

with open('students.csv', 'w', newline='') as file: 
    writer = csv.DictWriter(file, fieldnames = fields)

    writer.writeheader()


if __name__ == '__main__':
    path = "FireNetV2_400epoch_10fold_3segment_1frampersegment_batchsize32/ignore/events.out.tfevents.1689673499.f695224c4e89.487.0"
    #print_data(path)
    train_loss = create_dict(path)

    #print(train_loss)
    print(train_loss.keys(),len(train_loss['fold0']),len(train_loss['fold1']),len(train_loss['fold2']))
    #to_csv(path)