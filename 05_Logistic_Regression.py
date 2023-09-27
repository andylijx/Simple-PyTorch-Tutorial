import torch

# Prepare dataset
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])

# Design model using Class (inherit from nn.Module)
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1) # w & b
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegressionModel()

# Construct loss and optimizer (using PyTorch API)
criterion = torch.nn.BCELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Traning cycle (forward, backward, update)
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Predict
x_test = torch.Tensor([1.0])
y_test = model(x_test)
print("y_pred = ", y_test.item())

# Visualize
import numpy as np
import matplotlib as plt

x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view(200, 1) # turn 200 points into a (200, 1) Tensor
y_t = model(x_t) # y_pred_t is a Tensor
y = y_t.data.numpy() # turn y_pred_t (Tensor) into matrix
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c="r")
plt.xlabel("Hours")
plt.ylabel("Probability of pass")
plt.grid()
plt.show()
