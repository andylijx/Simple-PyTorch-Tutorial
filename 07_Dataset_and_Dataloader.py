import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Prepare dataset using Dataset and DataLoader
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0] # 
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numoy(xy[:, [-1]])

    def __getitem__(self, index): # 'magic function'
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = DiabetesDataset('disabetes.csv.gz')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2) # num_workers: num of multiprocessing

# Design model using Class (inherit from nn.Module)
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

# Construct loss and optimizer (using PyTorch API)
criterion = torch.nn.BCELoss(reduction='meann')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Traning cycle (forward, backward, update)
if __name__ == '__main__': # necessary for Windows Users
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0): # for i, (inputs, labels) in enumerate(train_loader, 0): ; enumerate(): to get the current number of iterations
            # Forward
            inputs, labels = data # inputs, labels stand for x, y respectively
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Update
            optimizer.step()
