import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

# If you can't find the data files, please revise the file path below.

train_data_path='breast_cancer_data_357B_100M.csv'
test_data_path='origin_breast_cancer_data.csv'

class unbalanced_breast_cancer_dataset(torch.utils.data.Dataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.input=pd.read_csv(train_data_path,sep='\t',usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]).values.tolist()
        self.target=pd.read_csv(train_data_path,sep='\t',usecols=['diagnosis'])['diagnosis']
        
    def __getitem__(self,idx):
        if(self.target[idx]=='M'):
            return self.input[idx],1
        else:
            return self.input[idx],0
        
    def __len__(self):
        return len(self.target)

def printRecallAndPrecision(prediction,target):
    TP_FP=torch.ge(prediction,0.5).sum()
    TP_FN=torch.ge(target,0.5).sum()
    TP=0
    for idx in range(target.size(0)):
        if torch.ge(prediction,0.5).float()[idx]==target.float()[idx] and torch.ge(prediction,0.5).float()[idx]==1:
            TP=TP+1
    recall=TP/TP_FN.item()
    precision=TP/TP_FP.item()
    F1_score=2*recall*precision/(recall+precision)
    print('recall:',round(recall,3),', precision:',round(precision,3),', F1 score:',round(F1_score,3))

# Train Process

KNN=KNeighborsClassifier(n_neighbors=20,weights='distance',algorithm='auto')
mydataset=unbalanced_breast_cancer_dataset()
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

Test_input=pd.read_csv(test_data_path,sep='\t',usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]).values.tolist()
Test_input=torch.FloatTensor(Test_input)
Test_target=pd.read_csv(test_data_path,sep='\t',usecols=['diagnosis'])['diagnosis']
for idx in range(len(Test_target)):
    if Test_target[idx]=='M':
        Test_target[idx]=1
    else:
        Test_target[idx]=0
Test_target=torch.FloatTensor(Test_target)
Test_prediction=KNN.predict(Test_input.float())
Test_prediction=torch.FloatTensor(Test_prediction)
printRecallAndPrecision(Test_prediction,Test_target)
print('accuracy:',round(KNN.score(Test_input,Test_target),3))
print('------------------------------')

# Improved
# Train Process

sm=SMOTE(random_state=42)
Train_input_res,Train_target_res=sm.fit_resample(Train_input,Train_target)
Train_input_res=torch.FloatTensor(Train_input_res)
Train_target_res=torch.FloatTensor(Train_target_res)
KNN.fit(Train_input_res,Train_target_res)

# Test Process

Test_prediction=KNN.predict(Test_input.float())
Test_prediction=torch.FloatTensor(Test_prediction)
printRecallAndPrecision(Test_prediction,Test_target)
print('accuracy:',round(KNN.score(Test_input,Test_target),3))
