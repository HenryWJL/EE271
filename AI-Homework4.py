import torch;

# Q1

print('-------------Q1-------------\n')
tensorX1=torch.rand(2,3)
tensorX2=torch.rand(5,4)
tensorX3=torch.rand(1,2)

print('TensorX1 is:\n',tensorX1,'\n','and its length is:\n',len(tensorX1),'\n')
print('TensorX2 is:\n',tensorX2,'\n','and its length is:\n',len(tensorX2),'\n')
print('TensorX3 is:\n',tensorX3,'\n','and its length is:\n',len(tensorX3),'\n')

# Q2

# broadcasting semantics
print('-------------Q2-------------\n')
matrixA=torch.rand(3,4)
print('A equals to:\n',matrixA,'\n')
print('A.sum(axis=1) equals to:\n',matrixA.sum(axis=1),'\n')
print('A/A.sum(axis=1) equals to:\n',matrixA/matrixA.sum(axis=1),'\n')

# Q3

#南北走向avenue，东西走向street
# Please view the report

# Q4

print('-------------Q4-------------\n')
x1=torch.tensor(1,requires_grad=True,dtype=torch.float32)
x2=torch.tensor(0,requires_grad=True,dtype=torch.float32)
fx=3*x1*x1+5*torch.exp(x2)
fx.backward()
print('fx\'s gradient w.r.t x1 equals to:\n',x1.grad,'\n')
print('fx\'s gradient w.r.t x2 equals to:\n',x2.grad,'\n')

# Q5

# Please view the report

# Q6

print('-------------Q6-------------\n')
u=torch.tensor(1,requires_grad=True,dtype=torch.float32)
v=torch.tensor(2,requires_grad=True,dtype=torch.float32)
x=3*u+v
y=2*u*u-4*v
gx=2*x*x-y
gx.backward()
print('gx\'s gradient w.r.t u equals to:\n',u.grad,'\n')
print('gx\'s gradient w.r.t v equals to:\n',v.grad,'\n')

# Q7

#Dx=E(x*x)-(Ex)(Ex)

print('-------------Q7-------------\n')
vectorX=torch.tensor([0,1,2,3,4,5,6,7,8,9,10])
vectorP=torch.tensor([0.006,0.0403,0.1209,0.215,0.2508,0.2007,0.1115,0.0425,0.0106,0.0016,0.0001])
Ex=(vectorX*vectorP).sum()
Ex2=((vectorX**2)*vectorP).sum()
Dx=Ex2-Ex**2
print('The variance equals to:\n',Dx,'\n')

