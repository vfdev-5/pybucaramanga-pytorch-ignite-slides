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
    pass

iteration = 0

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


---

# Global picture

<div style="font-size: 18px;">

```python
import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss
from ignite.contrib.engines import common


def initialize():
  # Pure pytorch code ...
  return model, optimizer, criterion


def training():
    train_loader, test_loader = get_data_loaders(train_batch_size, val_batch_size)
    model, optimizer, criterion = initialize()

    trainer: Engine = create_trainer(model, optimizer, criterion)
    metrics = {
        "Accuracy": Accuracy(), "Loss": Loss(criterion),
    }
    evaluator: Engine = create_evaluator(model, metrics=metrics)

    @trainer.on(Events.EPOCH_COMPLETED(every=3) | Events.COMPLETED)
    def run_validation(engine):
        evaluator.run(test_loader)

    if rank == 0:
        evaluators = {"test": evaluator, }
        tb_logger = common.setup_tb_logging(output_path, trainer, optimizer, evaluators=evaluators)

    trainer.run(train_loader, max_epochs=100)

    if rank == 0:
        tb_logger.close()

def main():
    backend = None # "nccl", "gloo", "xla-tpu", "horovod"
    with idist.Parallel(backend=backend) as parallel:
        parallel.run(training)

```

</div>

<!-- End vertical slides -->
{{% /section %}}