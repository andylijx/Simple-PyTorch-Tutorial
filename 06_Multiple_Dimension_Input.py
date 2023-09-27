import torch
import numpy as np
import matplotlib.pyplot as plt

# Prepare dataset
xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])

# Design model using Class (inherit from nn.Module)
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6) # 8D --> 6D
        self.linear2 = torch.nn.Linear(6, 4) # 6D --> 4D
        self.linear3 = torch.nn.Linear(4, 1) # 4D --> 1D
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

# Construct loss and optimizer (using PyTorch API)
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epoch_list = []
loss_list = []
# Traning cycle (forward, backward, update)
for epoch in range(100):
    # Forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    epoch_list.append(epoch)
    loss_list.append(loss.item())
    print(epoch, loss.item())

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Update
    optimizer.step()

plt.plot(epoch_list, loss_list)
plt.ylabel('loss_list')
plt.xlabel('epoch_list')
plt.show()
