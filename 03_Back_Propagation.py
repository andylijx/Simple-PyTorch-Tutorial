import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y) # Tensor; construct computational graph
        l.backward() # compute grad for Tensor whose requires_grad set to True; computaional graph will be released (deleted) after each .backward() is executed
        print('\tgrad:', x, y, w.grad.item()) # turn the data inside grad into the scaler in Python
        w.data = w.data - 0.01 * w.grad.data # the grad is utilized to update the weight

        w.grad.data.zero_() # to make sure the grad accumulated upwards (computed by .backward()) is set to ZERO
    print("Epoch:", epoch, " Loss:", l.item())

print("predict (after training)", 4, forward(4).item())

# Exercise 4-3: Compute gradients using computational graph
# Model: y^ = w1 * x ** 2 + w2 * x + b

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.Tensor([1.0])
w2 = torch.Tensor([1.0])
b = torch.Tensor([1.0])
w1.requires_grad = True
w2.requires_grad = True
b.requires_grad = True

def forward1(x):
    return w1 * (x ** 2) + w2 * x + b 

def loss1(x, y):
    y_pred = forward1(x)
    return (y_pred - y) ** 2

print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        a = 0.01 # learning rate α
        l = loss1(x, y)
        l.backward()
        print("\tgrad", x, y, w1.grad.item(), w2.grad.item(), b.grad.item())
        w1.data -= a * w1.grad.item()
        w2.data -= a * w2.grad.item()
        b.data -= a * b.grad.item()

        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    
    print("Epoch:", epoch, " Loss:", l.item())

print("predict (after training):", 4, forward1(4).item())
