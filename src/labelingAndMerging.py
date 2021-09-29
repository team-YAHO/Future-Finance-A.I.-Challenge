import pandas as pd
import os

LabelList = ['BYJ', 'HYN', 'SJW']
Dir = 'C:\\Users\\ovo6v\\Desktop\\future finance\\DATA\\'

#Labeling and Merging
Data = pd.DataFrame({"Label":[], "times":[], "F0(pitch)":[], "F1":[], "F2":[], 'F3':[], "F4":[], "F5":[]})
for Label in LabelList:
    Directory = Dir+Label+'\\'
    FileList = os.listdir(Directory)
    for file in FileList:
        data = pd.read_csv(Directory+file)
        data.columns = ['Label', 'times', 'F0(pitch)', 'F1', 'F2', 'F3', 'F4', 'F5']
        data['Label'] = Label
        Data = Data.append(data)

Data.to_csv(Dir+"train.csv")
