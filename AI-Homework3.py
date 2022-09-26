import torch

wx11=torch.tensor(0.48,requires_grad=True,dtype=torch.float32)
wx12=torch.tensor(-0.43,requires_grad=True,dtype=torch.float32)
wx21=torch.tensor(-0.51,requires_grad=True,dtype=torch.float32)
wx22=torch.tensor(-0.48,requires_grad=True,dtype=torch.float32)
wa11=torch.tensor(-0.99,requires_grad=True,dtype=torch.float32)
wa12=torch.tensor(0.36,requires_grad=True,dtype=torch.float32)
wa13=torch.tensor(-0.75,requires_grad=True,dtype=torch.float32)
wa21=torch.tensor(-0.66,requires_grad=True,dtype=torch.float32)
wa22=torch.tensor(0.34,requires_grad=True,dtype=torch.float32)
wa23=torch.tensor(0.66,requires_grad=True,dtype=torch.float32)
b10=torch.tensor(0.23,requires_grad=True,dtype=torch.float32)
b20=torch.tensor(0.05,requires_grad=True,dtype=torch.float32)
b11=torch.tensor(0.32,requires_grad=True,dtype=torch.float32)
b21=torch.tensor(-0.44,requires_grad=True,dtype=torch.float32)
b31=torch.tensor(0.7,requires_grad=True,dtype=torch.float32)

# Task1 The predictions

def f(x):
    f=1/(1+torch.exp(-x))
    return f

def sigma(z):
    z1=torch.exp(z[0])
    z2=torch.exp(z[1])
    z3=torch.exp(z[2])
    sigma=[z1/(z1+z2+z3),z2/(z1+z2+z3),z3/(z1+z2+z3)]
    return sigma

def network(x):
    
    x1=x[0]
    x2=x[1]
    
    a11=f(wx11*x1+wx21*x2+b10)
    a21=f(wx12*x1+wx22*x2+b20)
    
    y1=wa11*a11+wa21*a21+b11
    y2=wa12*a11+wa22*a21+b21
    y3=wa13*a11+wa23*a21+b31
    y=[y1,y2,y3]
    
    return sigma(y)

# Task2 The loss
 
x=torch.tensor([[0.1255,0.5377],
                [0.6564,0.0365],
                [0.5837,0.7018],
                [0.3068,0.9500],
                [0.4321,0.2946],
                [0.6015,0.1762],
                [0.9945,0.3177],
                [0.9886,0.3911]]) 


Jw=torch.tensor(0,requires_grad=True,dtype=torch.float32)
for i in range(0,8):
    for j in range(0,3):
        Jw=Jw+network(x[i])[j]*torch.log(network(x[i])[j])

Jw=Jw/-8

print(Jw)


# Task3 The gradient of the loss

Jw.backward()

print(f"Jw's derivative w.r.t wx11 is {wx11.grad}")
print(f"Jw's derivative w.r.t wx12 is {wx12.grad}")
print(f"Jw's derivative w.r.t wx21 is {wx21.grad}")
print(f"Jw's derivative w.r.t wx22 is {wx22.grad}")
print(f"Jw's derivative w.r.t wa11 is {wa11.grad}")
print(f"Jw's derivative w.r.t wa12 is {wa12.grad}")
print(f"Jw's derivative w.r.t wa13 is {wa13.grad}")
print(f"Jw's derivative w.r.t wa21 is {wa21.grad}")
print(f"Jw's derivative w.r.t wa22 is {wa22.grad}")
print(f"Jw's derivative w.r.t wa23 is {wa23.grad}")
print(f"Jw's derivative w.r.t b10 is {b10.grad}")
print(f"Jw's derivative w.r.t b20 is {b20.grad}")
print(f"Jw's derivative w.r.t b11 is {b11.grad}")
print(f"Jw's derivative w.r.t b21 is {b21.grad}")
print(f"Jw's derivative w.r.t b31 is {b31.grad}")
