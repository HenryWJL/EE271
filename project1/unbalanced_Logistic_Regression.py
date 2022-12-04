import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn

# When you run the program, sometimes there may appear 'ZeroDivisionError' or 'Loss-Unabated-Error'.
# Don't worry, restart the program one more times until the training process can proceed successfully.
##
# Please revise the absolute path below so that the program can run on your computer.

train_data_path='/home/txke/Python_codes/project1/breast_cancer_data_357B_100M.csv'
test_data_path='/home/txke/Python_codes/project1/origin_breast_cancer_data.csv'

class origin_breast_cancer_dataset(torch.utils.data.Dataset):
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
    
class TrainingModle(nn.Module):
    def __init__(self):
        super(TrainingModle,self).__init__()
        self.linear=nn.Linear(30,1)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        y=self.linear(x)
        z=self.sigmoid(y)
        return z
    
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

def printAccuracy(prediction,target):
    correct=torch.sum(torch.ge(prediction,0.5).float()==target.float())
    accuracy=correct.item()/prediction.size(0)
    print("accuracy:",round(accuracy,3))

# Train Process

mydataset=origin_breast_cancer_dataset()
mymodel=TrainingModle()
loss_fn=torch.nn.BCELoss()
optimizer=torch.optim.SGD(mymodel.parameters(),lr=0.00001,momentum=0.05)
training_set=DataLoader(mydataset,shuffle=True)
Train_input=[]
Train_target=[]
for input,target in training_set:
    Train_input.append(input)
    Train_target.append(target)
Train_input=torch.FloatTensor(Train_input)
Train_target=torch.unsqueeze(torch.FloatTensor(Train_target),dim=1)

for epoch in range(100000):    
    Train_prediction=mymodel(Train_input.float())
    Train_loss=loss_fn(Train_prediction.float(),Train_target.float())
    optimizer.zero_grad()
    Train_loss.backward()
    optimizer.step()
        
    if epoch%10000==0:
        print('Training_set: loss:',round(Train_loss.item(),3),end=' , ')
        printRecallAndPrecision(Train_prediction,Train_target)
        
# Test Process

Test_input=pd.read_csv(test_data_path,sep='\t',usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]).values.tolist()
Test_input=torch.FloatTensor(Test_input)
Test_target=pd.read_csv(test_data_path,sep='\t',usecols=['diagnosis'])['diagnosis']

for idx in range(len(Test_target)):
    if Test_target[idx]=='M':
        Test_target[idx]=1
    else:
        Test_target[idx]=0
Test_target=torch.unsqueeze(torch.FloatTensor(Test_target),dim=1)
Test_prediction=mymodel(Test_input.float())
printAccuracy(Test_prediction,Test_target)
