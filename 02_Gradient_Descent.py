import matplotlib.pyplot as plt

# Gradient Descent

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0 # y^ = x * w; To start with a random guess

def forward(x):
    return x * w

def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)

def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)

epoch_list = []
cost_list = []

print("Predict (before training)", 4, forward(4))

for epoch in range(1, 101):
    a = 0.01 # learning rate: α
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= a * grad_val
    
    print("Epoch:", epoch, " w=", w, " loss=", cost_val)
    epoch_list.append(epoch)
    cost_list.append(cost_val)

print("Predict (after training)", 4, forward(4))

plt.plot(epoch_list, cost_list)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.show()

# Stochastic Gradient Descent (SGD)

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 0.1

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

def gradient(x, y):
    return 2 * x * (x * w - y)

epoch_list = []
loss_list = []

print("Predict (before training)", 4, forward(4))

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        a = 0.01
        grad = gradient(x, y)
        w -= a * grad
        print("\tgrad:", x, y, grad)
        l = loss(x, y)
    
    print("Epoch:", epoch, " w=", w, " loss=", l)
    loss_list.append(l)
    epoch_list.append(epoch)

print("Predict (after training)", 4, forward(4))

plt.plot(epoch_list, loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
