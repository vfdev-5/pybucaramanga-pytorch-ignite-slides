+++
weight = 1
+++


<!-- Start vertical slides -->
{{% section %}}

# PyTorch in a nutshell

<table style="font-size: 20px;">
<tr>

<td>

```python
import torch
import torch.nn as nn

device = "cuda"

class MyNN(nn.Module):
    def __init__(self):
        super(MyNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = MyNN().to(device)
```

</td>

<td>

{{< add_vspace >}}

- tensor manipulations (device: CPUs, GPUs, TPUs)
- NN components, optimizers, loss functions
- Distributed computations
- Profiling
- other cool features ...
- Domain libraries: vision, text, audio
- Rich ecosystem

</td>

</tr>

</table>

https://pytorch.org/tutorials/beginner/basics/intro.html

---

# Quick-start ML with PyTorch

<div style="font-size: 20px;">

[Computer Vision example with Fashion MNIST](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)

Problem: 1 - how to classify images ?

`model(image) -> predicted label`

2 - How measure model performances ?

`predicted labels vs correct labels`


<img height="300" src="https://image.itmedia.co.jp/ait/articles/2005/28/di-01.gif" />

---

# Quick-start ML with PyTorch

<div style="font-size: 20px;">

- Setup training and testing data

```python
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

# Setup training/test data
training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, transform=ToTensor())

batch_size = 64

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Optionally, for debugging:
for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# Output:
# Shape of X [N, C, H, W]:  torch.Size([64, 1, 28, 28])
# Shape of y:  torch.Size([64]) torch.int64
```

</div>


---

# Quick-start ML with PyTorch

<div style="font-size: 20px;">

- Create a model

```python
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
```

</div>

---

# Quick-start ML with PyTorch

<div style="font-size: 20px;">

- Model training
  - Loss function: cross-entropy
  - Optimization with Stochastic Gradient Descent

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, model, loss_fn):
    # code to compute and print average loss and accuracy

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

</div>



---

## Why using PyTorch without Ignite is suboptimal ?

For NN training and evaluation:
- PyTorch gives only "low"-level building components
- Common bricks to code in any user project:
  - metrics
  - checkpointing, best model saving, early stopping, ...
  - logging to experiment tracking systems
  - code adaptation for device (e.g. GPU, XLA)

---

- Pure PyTorch code

<div style="font-size: 20px;">

```python

model = Net()
train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
criterion = torch.nn.NLLLoss()

max_epochs = 10
validate_every = 100
checkpoint_every = 100


def validate(model, val_loader):
    model = model.eval()
    num_correct = 0
    num_examples = 0
    for batch in val_loader:
        input, target = batch
        output = model(input)
        correct = torch.eq(torch.round(output).type(target.type()), target).view(-1)
        num_correct += torch.sum(correct).item()
        num_examples += correct.shape[0]
    return num_correct / num_examples


def checkpoint(model, optimizer, checkpoint_dir):
    # ...

def save_best_model(model, current_accuracy, best_accuracy):
    # ...

iteration = 0
best_accuracy = 0.0

for epoch in range(max_epochs):
    for batch in train_loader:
        model = model.train()
        optimizer.zero_grad()
        input, target = batch
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if iteration % validate_every == 0:
            binary_accuracy = validate(model, val_loader)
            print("After {} iterations, binary accuracy = {:.2f}"
                  .format(iteration, binary_accuracy))
            save_best_model(model, binary_accuracy, best_accuracy)

        if iteration % checkpoint_every == 0:
            checkpoint(model, optimizer, checkpoint_dir)
        iteration += 1

```

</div>

---

# PyTorch-Ignite: what and why? 🤔

> High-level **library** to help with training and evaluating neural networks in PyTorch flexibly and transparently.

- https://github.com/pytorch/ignite


<table style="font-size: 20px;">
<tr>

<td>

```python

def train_step(engine, batch):
  #  ... any training logic ...
  return batch_loss

trainer = Engine(train_step)

# Compose your pipeline ...

trainer.run(train_loader, max_epochs=100)


```
</td>

<td>

```python

metrics = {
  "precision": Precision(),
  "recall": Recall()
}

evaluator = create_supervised_evaluator(
  model,
  metrics=metrics
)


```
</td>

<td>

```python
@trainer.on(Events.EPOCH_COMPLETED)
def run_evaluation():
  evaluator.run(test_loader)

handler = ModelCheckpoint(
  '/tmp/models', 'checkpoint'
)
trainer.add_event_handler(
  Events.EPOCH_COMPLETED,
  handler,
  {'model': model}
)
```
</td>

</tr>

</table>

---


# Key concepts in a nutshell

#### PyTorch-Ignite is about:

1) Engine and Event System
2) Out-of-the-box metrics to easily evaluate models
3) Built-in handlers to compose training pipeline
4) Distributed Training support

---

# What makes PyTorch-Ignite unique ?

- Composable and interoperable components
- Simple and understandable code
- Open-source community involvement

---

# How PyTorch-Ignite makes user's live easier ?

With PyTorch-Ignite:

- Less code than pure PyTorch while ensuring maximum control and simplicity
- Easily get more refactored and structured code
- Extensible API for metrics, experiment managers, and other components
- Same code for non-distributed and distributed configs

<!-- End vertical slides -->
{{% /section %}}