import torch

x_data = torch.Tensor([[1.0],
                       [2.0],
                       [3.0]])
y_data = torch.Tensor([[2.0],
                       [4.0],
                       [6.0]])

class LinearModel(torch.nn.Module): # our model class should be inherit from nn.Module, which is Base class for all neural network modules

    def __init__(self): # member methods __init__() and forward() have to be implemented
        super(LinearModel, self).__init__() # super(LinearModel, self) firstly finds out LinearModel's parent(namely torch.nn.Module), and then turn the LinearModel's object into torch.nn.Module's
        self.linear = torch.nn.Linear(1, 1) # class nn.Linear contain 2 member Tensors: weight and bias

    def forward(self, x):
        y_pred = self.linear(x) # class nn.Linear is callable like a function
        return y_pred

model = LinearModel() # create a instance of class LinearModel

criterion = torch.nn.MSELoss(size_average=False) # also inherit from nn.Module
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(x_data) # forward: predict
    loss = criterion(y_pred, y_data) # forward: loss
    print(epoch, loss.item())

    optimizer.zero_grad() # to make sure the grad accumulated by .backward() is set to ZERO
    loss.backward() # backward: autograd
    optimizer.step() # update

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([4.0])
y_test = model(x_test)
print('y_pred = ', y_test.data)
