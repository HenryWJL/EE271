import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn
from imblearn.over_sampling import SMOTE

input_data_path='ecg_input_data.csv'
target_data_path='ecg_target_data.csv'

class rnn_train_dataset(torch.utils.data.Dataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.input=pd.read_csv(input_data_path,sep='\t',header=None,nrows=3000).values.tolist()
        self.target=pd.read_csv(target_data_path,sep='\t',header=None,nrows=3000).values.tolist()
        
    def __getitem__(self,idx):
        return self.input[idx],self.target[idx]
        
    def __len__(self):
        return len(self.target)
    
class rnn_validation_dataset(torch.utils.data.Dataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.input=pd.read_csv(input_data_path,sep='\t',header=None,skiprows=3000,nrows=2000).values.tolist()
        self.target=pd.read_csv(target_data_path,sep='\t',header=None,skiprows=3000,nrows=2000).values.tolist()
        
    def __getitem__(self,idx):
        return self.input[idx],self.target[idx]
        
    def __len__(self):
        return len(self.target)

    
class RNN_Model(nn.Module):
    def __init__(self):
        super(RNN_Model,self).__init__()
        self.linear_f=nn.Linear(5,4,bias=True)
        self.linear_i=nn.Linear(5,4,bias=True)
        self.linear_c=nn.Linear(5,4,bias=True)
        self.linear_o=nn.Linear(5,4,bias=True)
        self.linear_w=nn.Linear(188,1,bias=True)
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()
        
    def lstm(self,x,ht_1,ct_1):
        ft=self.sigmoid(self.linear_f(torch.cat([x,ht_1],dim=1)))
        it=self.sigmoid(self.linear_i(torch.cat([x,ht_1],dim=1)))
        ct_=self.tanh(self.linear_c(torch.cat([x,ht_1],dim=1)))
        ct=ft*ct_1+it*ct_
        ot=self.sigmoid(self.linear_o(torch.cat([x,ht_1],dim=1)))
        ht=self.tanh(ct)*ot
        return ct,ht,ot
        
    def forward(self,input):
        ct_1=torch.rand((188,4),requires_grad=True)
        ht_1=torch.rand((188,4),requires_grad=True)
        ot=torch.zeros((1,4),requires_grad=True)
        for idx in range(input.size(0)):
            x=torch.unsqueeze(input[idx],dim=1)
            ct,ht,ot0=self.lstm(x,ht_1,ct_1)
            ot=torch.cat([ot,self.linear_w(ot0.T).T],dim=0)  #此处通过w将原本188x4的输出矩阵变成了1x4
            ct_1=ct
            ht_1=ht
        ot=ot[1:][:]
        return ot
        
        
def print_f1_score(prediction,target):
    TP_FP_norm=0
    TP_FN_norm=0
    TP_norm=0
    TP_FP_af=0
    TP_FN_af=0
    TP_af=0
    TP_FP_other=0
    TP_FN_other=0
    TP_other=0
    max,indice=torch.max(prediction,dim=1) # finding the largest weights
    for idx in range(target.size(0)):
        if indice[idx]==0:
            TP_FP_norm+=1
        elif indice[idx]==1:
            TP_FP_af+=1
        elif indice[idx]==2:
            TP_FP_other+=1
            
        if target[idx]==0:
            TP_FN_norm+=1
        elif target[idx]==1:
            TP_FN_af+=1
        elif target[idx]==2:
            TP_FN_other+=1
            
        if (indice[idx]==0) and (target[idx]==0):
            TP_norm+=1
        elif (indice[idx]==1) and (target[idx]==1):
            TP_af+=1
        elif (indice[idx]==2) and (target[idx]==2):
            TP_other+=1

    # metrics of norm
    recall_norm=TP_norm/TP_FN_norm
    precision_norm=TP_norm/TP_FP_norm
    F1_score_norm=2*recall_norm*precision_norm/(recall_norm+precision_norm)
    # metrics of af
    recall_af=TP_af/TP_FN_af
    precision_af=TP_af/TP_FP_af
    F1_score_af=2*recall_af*precision_af/(recall_af+precision_af)
    # metrics of other rhythms
    recall_other=TP_other/TP_FN_other
    precision_other=TP_other/TP_FP_other
    F1_score_other=2*recall_other*precision_other/(recall_other+precision_other)
    # final accuracy
    F1_score_final=(F1_score_norm+F1_score_af+F1_score_other)/3
    
    print('F_norm:',round(F1_score_norm,3),'; F_af:',round(F1_score_af,3),
          '; F_other:',round(F1_score_other,3),'; F_T:',round(F1_score_final,3))
    
    
# initialize
mymodel=RNN_Model()
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(mymodel.parameters(),lr=0.00005,momentum=0.5)

# extract training data
train_dataset=rnn_train_dataset()
training_set=DataLoader(train_dataset,shuffle=True)
Train_input=[]
Train_target=[]
for input,target in training_set:
    Train_input.append(input)
    Train_target.append(target)
Train_input=torch.FloatTensor(Train_input)
Train_target=torch.FloatTensor(Train_target)
Train_target=Train_target-1

# resample the dataset to deal with imbalanced problems
sm=SMOTE(random_state=42)
Train_input_res,Train_target_res=sm.fit_resample(Train_input,Train_target)
Train_input_res=torch.FloatTensor(Train_input_res)
Train_target_res=torch.tensor(Train_target_res,dtype=torch.long)

# extract validation data
valid_dataset=rnn_validation_dataset()
valid_set=DataLoader(valid_dataset,shuffle=False)
Valid_input=[]
Valid_target=[]
for input,target in valid_set:
    Valid_input.append(input)
    Valid_target.append(target)
Valid_input=torch.FloatTensor(Valid_input)
Valid_target=torch.tensor(Valid_target,dtype=torch.long)
Valid_target=Valid_target-1
Valid_target=torch.squeeze(Valid_target,dim=1)

# training and validation process
for epoch in range(501):    
    Train_prediction=mymodel(Train_input_res.float())
    Valid_prediction=mymodel(Valid_input.float())
    Train_loss=loss_fn(Train_prediction.float(),Train_target_res)
    Valid_loss=loss_fn(Valid_prediction.float(),Valid_target)
    if epoch%100==0 and epoch!=0:
        print('--------------------epoch=',epoch,'------------------------')
        print('Training:\nLoss:',Train_loss.item())
        print_f1_score(Train_prediction,Train_target_res)
        print('Validation:')
        print_f1_score(Valid_prediction,Valid_target)
    optimizer.zero_grad()
    Train_loss.backward()
    optimizer.step()
