import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn
from imblearn.over_sampling import SMOTE

input_data_path='ecg_input_data.csv'
target_data_path='ecg_target_data.csv'

class key_dataset(torch.utils.data.Dataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.key=pd.read_csv(input_data_path,sep='\t',header=None,nrows=3000).values.tolist()
        self.value=pd.read_csv(target_data_path,sep='\t',header=None,nrows=3000).values.tolist()
        
    def __getitem__(self,idx):
        return self.key[idx],self.value[idx]
    def __len__(self):
        return len(self.value)
    
class query_dataset(torch.utils.data.Dataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.query=pd.read_csv(input_data_path,sep='\t',header=None,skiprows=3000,nrows=3000).values.tolist()
        self.target=pd.read_csv(target_data_path,sep='\t',header=None,skiprows=3000,nrows=3000).values.tolist()
        
    def __getitem__(self,idx):
        return self.query[idx],self.target[idx]
        
    def __len__(self):
        return len(self.target)
    
class Attention_Model(nn.Module):
    def __init__(self):
        super(Attention_Model,self).__init__()
        self.softmax=nn.Softmax(dim=1)
        self.linear1=nn.Linear(188,7136)
        self.linear2=nn.Linear(1,4)
    def forward(self,key,value,query):
        a=self.softmax(-(self.linear1(query-key))**2/2)
        fx=self.linear2(torch.mm(a,value))
        return fx

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
mymodel=Attention_Model()
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(mymodel.parameters(),lr=0.005,momentum=0.5)        
        
# extract key-value        
Keydataset=key_dataset()
keydataset=DataLoader(Keydataset,shuffle=True)
keys=[]
values=[]
for key,value in keydataset:
    keys.append(key)
    values.append(value)
keys=torch.FloatTensor(keys)
values=torch.FloatTensor(values)

# extract query-target
Querydataset=query_dataset()
querydataset=DataLoader(Querydataset,shuffle=True)
queries=[]
targets=[]
for query,target in querydataset:
    queries.append(query)
    targets.append(target)
queries=torch.FloatTensor(queries)
targets=torch.FloatTensor(targets)-1

# resample
sm=SMOTE(random_state=42)
keys_res,values_res=sm.fit_resample(keys,values)
queries_res,targets_res=sm.fit_resample(queries,targets)
keys_res=torch.FloatTensor(keys_res)
values_res=torch.unsqueeze(torch.FloatTensor(values_res),dim=1)

# modulate the size of queries and targets
queries_res=torch.FloatTensor(queries_res)[0:keys_res.size(0)][:]
targets_res=torch.tensor(targets_res,dtype=torch.long)[0:keys_res.size(0)]

# training process
for epoch in range(501):
    prediction=mymodel(keys_res.float(),values_res.float(),queries_res.float())
    loss=loss_fn(prediction,targets_res)
    if epoch%100==0 and epoch!=0:
        print('--------------------epoch=',epoch,'------------------------')
        print('Loss:',loss.item())
        print_f1_score(prediction,targets_res)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
