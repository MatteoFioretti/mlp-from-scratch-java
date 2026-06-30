# MLP from Scratch in Java

A multilayer perceptron built entirely from scratch in pure Java to classify MNIST handwritten digits — no machine learning libraries, no autograd, no external numerical dependencies. Every matrix multiply, the backpropagation, and the binary dataset parser are implemented by hand.

Reaches **97.6% test accuracy** on MNIST.

---

## Highlights

- **Zero ML dependencies.** No TensorFlow, no DL4J, no ND4J. Just the Java standard library.
- **Backpropagation by hand.** Gradients are derived analytically and implemented directly — no automatic differentiation.
- **Custom binary data pipeline.** Reads the raw MNIST IDX files byte by byte, with magic-number validation, pixel normalization, and one-hot encoding.
- **Clean object-oriented design.** Abstract base classes, polymorphism, and dependency injection structure the codebase the way a real ML framework would.
- **97.6% test accuracy** in 10 epochs.

---

## Results

Training for 10 epochs with SGD (learning rate `0.001`, batch size `128`):

| Epoch | Loss   | Train Accuracy |
|-------|--------|----------------|
| 1     | 0.3909 | 88.72%         |
| 5     | 0.0798 | 97.70%         |
| 10    | 0.0337 | 99.17%         |

**Final test accuracy: 97.63%** (loss 0.0778)

The small gap between training (99.2%) and test (97.6%) accuracy indicates the network generalized well rather than memorizing the training set.

---

## Architecture & Design

The project is structured around abstract base classes that define contracts, with concrete implementations that fulfill them — mirroring how production ML frameworks separate interface from implementation.

### Object-oriented design

- **`Model`** *(abstract)* — defines the contract every model must fulfill: `forward`, `loss`, `backward`, and `step`. Knows nothing about any specific architecture.
- **`MLP`** *(extends `Model`)* — the concrete network: parameters, forward pass, hand-derived backpropagation, and SGD updates.
- **`DataModule`** *(abstract, implements `Iterable<Batch>`)* — defines a batched data source. Holds the batch size and the iteration contract.
- **`MNISTDataModule`** *(extends `DataModule`)* — parses the MNIST IDX binary files and serves batches via a custom iterator.
- **`Trainer`** — orchestrates training and evaluation. It depends only on the abstract `Model` and `DataModule` types, so it can train any model on any dataset without modification.
- **`MatrixUtils`** — a stateless toolkit of static linear-algebra operations (matrix multiply, transpose, ReLU, softmax, element-wise ops).
- **`Batch`** — an immutable holder pairing a batch of features with its labels.

The design deliberately demonstrates four core OOP principles:

- **Abstraction** — `Model` and `DataModule` define *what* without committing to *how*.
- **Inheritance** — `MLP` and `MNISTDataModule` extend their abstract bases.
- **Polymorphism** — `Trainer` operates on abstract types and resolves the concrete implementations at runtime.
- **Dependency injection** — the model and data module are constructed externally and passed into the `Trainer`, decoupling it from any concrete class.

### Network design

```
Input (784) → Hidden (256) → Hidden (64) → Output (10)
                  ReLU           ReLU         Softmax
```

- **He initialization** for the weights, suited to ReLU activations.
- **ReLU** hidden activations for stable gradient flow.
- **Softmax + cross-entropy** output, with the combined gradient (`ŷ − y`) used directly in the backward pass for numerical stability and simplicity.
- **Mini-batch SGD** as the optimizer.

---

## Project Structure

```
mlp-from-scratch-java/
├── data/                          # MNIST IDX files (not tracked — see below)
├── src/main/java/com/mlp/
│   ├── Main.java                  # Entry point: wires everything together
│   ├── Model.java                 # Abstract model contract
│   ├── MLP.java                   # The network implementation
│   ├── DataModule.java            # Abstract batched data source
│   ├── MNISTDataModule.java       # MNIST IDX parser + batch iterator
│   ├── Trainer.java               # Training & evaluation loop
│   ├── MatrixUtils.java           # Linear-algebra toolkit
│   └── Batch.java                 # Feature/label batch holder
├── pom.xml
└── README.md
```

---

## Running It

### 1. Get the dataset

The MNIST data is not tracked in this repository. Download the four IDX files from the [MNIST dataset on Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and place them in a `data/` folder at the project root:

```
data/
├── train-images.idx3-ubyte
├── train-labels.idx1-ubyte
├── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

### 2. Build

Requires **JDK 21+** and **Maven**.

```bash
mvn compile
```

### 3. Run

```bash
mvn exec:java
```

Or run `Main` directly from your IDE.

---

## Notes

This project was built as a learning exercise to implement a neural network end to end without relying on any machine-learning framework — from parsing the raw dataset to deriving and coding backpropagation by hand. The goal was both a working classifier and a clean, idiomatic Java codebase.

### Possible extensions

- Model persistence (saving and loading trained weights)
- Configurable layer count for arbitrary-depth networks
- Additional optimizers (Adam, momentum)
- A validation split for hyperparameter tuning
