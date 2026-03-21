import torch
from torch import nn

w = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(-1.0, requires_grad = True)

# Function forward(x) for prediction
def forward(x):
    yhat = w * x + b
    return yhat

x = torch.tensor([[1.0]]) #single value
yhat = forward(x)
print("The prediction with single X: ", yhat)

# Practice: Make a prediction of y = 2x - 1 at x = [[1.0], [2.0], [3.0]]
x = torch.tensor([[1.0], [2.0], [3.0]])
yhat = forward(x)
print("The prediction multiple X: ", yhat)

print ("\n")

# Import Class Linear

from torch.nn import Linear
torch.manual_seed(1) #Set the random seed because the parameters are randomly initialized:

# Create Linear Regression Model, and print out the parameters
lr = Linear(in_features=1, out_features=1, bias=True)
print("Parameters w and b: ", list(lr.parameters()))

print ("\n")

#A method state_dict() Returns a Python dictionary object corresponding to the layers of each parameter tensor.
print("Python dictionary: ",lr.state_dict())
print("keys: ",lr.state_dict().keys())
print("values: ",lr.state_dict().values())
print("weight:",lr.weight)
print("bias:",lr.bias)

#  Now let us make a single prediction at x = [[1.0]].
 
print ("\n")
 
x = torch.tensor([[1.0]])
yhat = lr(x)
print(f"The prediction for x:- {x} is {yhat}")