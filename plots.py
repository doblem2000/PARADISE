import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import os
source = "csv/"
target = "plots/"

### create target directory if it doesn't exist ###
os.makedirs(target,exist_ok=True)

experiments = [d for d in os.listdir(source) if os.path.isdir(source+d)]
# for experiment in experiments:
#     print(experiment)

# Set the figure size
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

#print("Current Working Directory ", os.getcwd())

# Read a CSV file
for experiment in experiments:
    print(experiment)
    temp = os.listdir(source+experiment+"/train/acc/")
    count = len(temp)
    #print(temp, len(temp))
    
    for i in range(count):
        # Make a list of columns
        for metric in ['acc','loss']:
            columns = ['Step', 'Value']
            legends = ['train', 'val']
            #print(legends)
            df_train = pd.read_csv(source + experiment + "/train/" +metric +"/fold"+str(i+1)+".csv", usecols=columns)
            df_val = pd.read_csv(source + experiment + "/val/" +metric + "/fold"+str(i+1)+".csv", usecols=columns)
            #print(df_train==df_val)
            plt.title(experiment + " " + metric + " fold " + str(i))
            ax = df_train.plot(x='Step', y='Value', kind = 'line') #legend=True, label=legends)
            df_val.plot(ax = ax,x='Step', y='Value', kind = 'line')#,legend=True, label=legends)
            pathname = target+experiment + "/"+ metric + "_train-val/" 
            path = Path(pathname)
            path.mkdir(parents=True, exist_ok=True)
            plt.legend(legends)
            plt.savefig(pathname+"fold"+str(i+1)+".png")
            plt.clf()
            plt.close()


    