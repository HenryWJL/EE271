import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier

class origin_breast_cancer_dataset(torch.utils.data.Dataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.input=pd.read_csv('/home/txke/Python_codes/project1/origin_breast_cancer_data.csv',sep='\t',usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]).values.tolist()
        self.target=pd.read_csv('/home/txke/Python_codes/project1/origin_breast_cancer_data.csv',sep='\t',usecols=['diagnosis'])['diagnosis']
        
    def __getitem__(self,idx):
        if(self.target[idx]=='M'):
            return self.input[idx],1
        else:
            return self.input[idx],0
        
    def __len__(self):
        return len(self.target)

def printAccuracy(prediction,target):
    correct=torch.sum(torch.ge(prediction,0.5).float()==target.float())
    accuracy=correct.item()/target.size(0)
    print("accuracy is:",round(accuracy,3))

# Train Process

KNN=KNeighborsClassifier(n_neighbors=20,weights='distance',algorithm='auto')
mydataset=origin_breast_cancer_dataset()
training_set=DataLoader(mydataset,shuffle=True)
Train_input=[]
Train_target=[]
for input,target in training_set:
    Train_input.append(input)
    Train_target.append(target)
Train_input=torch.FloatTensor(Train_input)
Train_target=torch.FloatTensor(Train_target)
KNN.fit(Train_input,Train_target)

# Test Process

Test_input=pd.read_csv('/home/txke/Python_codes/project1/breast_cancer_data_357B_100M.csv',sep='\t',usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]).values.tolist()
Test_input=torch.FloatTensor(Test_input)
Test_target=pd.read_csv('/home/txke/Python_codes/project1/breast_cancer_data_357B_100M.csv',sep='\t',usecols=['diagnosis']).values.tolist()
for idx in range(len(Test_target)):
    if Test_target[idx]=='M':
        Test_target[idx]=1
    else:
        Test_target[idx]=0
Test_target=torch.FloatTensor(Test_target)
Test_prediction=KNN.predict(Test_input.float())
Test_prediction=torch.FloatTensor(Test_prediction)
printAccuracy(Test_prediction,Test_target)
