+++
weight = 2
+++

<!-- Start vertical slides -->
{{% section %}}

# Quick-Start Example 👩‍💻👨‍💻

Let's train a MNIST classifier with PyTorch-Ignite!

---

### ⬇️ Installation ⬇️

Install PyTorch and TorchVision
```bash
$ pip install torch torchvision
```
Install PyTorch-Ignite

via `pip` 📦
```bash
$ pip install pytorch-ignite
```
or `conda` 🐍
```bash
$ conda install ignite -c pytorch
```

---

### 📦 Imports 📦

```python{1-6|8-12}
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger
```

---

### Start with a PyTorch code

<div style="font-size: 22px;">
Set up the dataflow, define a model (adapted ResNet18), a loss and an optimizer.

```python{1-7|9-20|22-23}
data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

train_dataset = MNIST(download=True, root=".", transform=data_transform, train=True)
val_dataset = MNIST(download=True, root=".", transform=data_transform, train=False)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.model = resnet18(num_classes=10)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.model(x)

device = "cuda"
model = Net().to(device)

optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()
```
</div>

---

### It's time for PyTorch-Ignite! 🔥

```python{1|3-6|8}
trainer = create_supervised_trainer(model, optimizer, criterion, device)

val_metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(criterion)
}

evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
```

- `trainer` engine to train the model
- `evaluator` engine to compute metrics on validation set + save the best models

---

#### Add handlers for logging the progress

<div style="font-size: 26px;">

```python{1-3|5-11}
@trainer.on(Events.ITERATION_COMPLETED(every=100))
def log_training_loss(engine):
    print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(f"Validation Results - Epoch[{trainer.state.epoch}] "
          f"Avg accuracy: {metrics['accuracy']:.2f} "
          f"Avg loss: {metrics['loss']:.2f}")
```

</div>

---

#### Add `ModelCheckpoint` handler with accuracy as a score function

```python
model_checkpoint = ModelCheckpoint(
    "checkpoint",
    n_saved=2,
    filename_prefix="best",
    score_function=lambda e: e.state.metrics["accuracy"],
    score_name="accuracy",
)

evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})
```

---

#### Add Tensorboard Logger

<div style="font-size: 22px;">

```python
tb_logger = TensorboardLogger(log_dir="tb-logger")

tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=100),
    tag="training",
    output_transform=lambda loss: {"batch_loss": loss},
)

tb_logger.attach_output_handler(
    evaluator,
    event_name=Events.EPOCH_COMPLETED,
    tag="validation",
    metric_names="all",
    global_step_transform=global_step_from_engine(trainer)
)
```

</div>

---

#### 🚀Liftoff!🚀

```python
trainer.run(train_loader, max_epochs=5)
```

```python
Epoch[1], Iter[100] Loss: 0.19
Epoch[1], Iter[200] Loss: 0.13
Epoch[1], Iter[300] Loss: 0.08
Epoch[1], Iter[400] Loss: 0.11
Training Results - Epoch[1] Avg accuracy: 0.97 Avg loss: 0.09
Validation Results - Epoch[1] Avg accuracy: 0.97 Avg loss: 0.08
...
Epoch[5], Iter[1900] Loss: 0.02
Epoch[5], Iter[2000] Loss: 0.11
Epoch[5], Iter[2100] Loss: 0.05
Epoch[5], Iter[2200] Loss: 0.02
Epoch[5], Iter[2300] Loss: 0.01
Training Results - Epoch[5] Avg accuracy: 0.99 Avg loss: 0.02
Validation Results - Epoch[5] Avg accuracy: 0.99 Avg loss: 0.03
```

---

### Inspect results in Tensorboard

<img height="540" src="images/tensorboard.png"/>

---

### Complete code

- https://pytorch-ignite.ai/tutorials/getting-started
- [Colab notebook](https://colab.research.google.com/github/pytorch-ignite/pytorch-ignite.ai/blob/gh-pages/tutorials/getting-started.ipynb)
---

### PyTorch-Ignite Code-Generator

<img height="300" src="https://raw.githubusercontent.com/pytorch-ignite/code-generator/main/src/assets/code-generator-demo-1080p.gif"/>

<div style="font-size: 20px;">

https://code-generator.pytorch-ignite.ai/

- **What is Code-Generator?**: web app to quickly produce quick-start python code for common training tasks in deep learning.

- **Why to use Code-Generator?**: start working on a task without rewriting everything from scratch.

</div>

---

Any questions before we go on ?


<!-- End vertical slides -->
{{% /section %}}
