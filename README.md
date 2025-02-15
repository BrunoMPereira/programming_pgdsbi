# Perceptron Classifier with Probabilistic Activation

This project implements a Perceptron classifier with a probabilistic activation function. It allows users to favor a specific class during classification based on a given probability.

References:
* https://sebastianraschka.com/Articles/2015_singlelayer_neurons.html
* https://medium.com/codex/single-layer-perceptron-and-activation-function-b6b74b4aae66
* https://www.simplilearn.com/tutorials/deep-learning-tutorial/perceptron
* https://www.geeksforgeeks.org/what-is-perceptron-the-simplest-artificial-neural-network/


Docstrings AND this README.md were generated by ChatGPT. Used also to optimize code and logic behind activation function and moving average error.

## Features
- Binary classification using a perceptron.
- Customizable learning rate and iterations.
- Early stopping based on classification error.
- Probabilistic activation that favors one class based on user-defined probability.
- Visualization of results for different probabilities.

## Installation
Ensure you have Python installed along with the necessary dependencies:
```sh
pip install numpy matplotlib
```

## Usage
Run the script and input the parameters when prompted:
```sh
python main.py
```

## Example: Comparing Results with Different Probabilities
The following code compares classification results when using different probability values for favoring a specific class.

```python
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron, generate_classification_dataset

# Set seed for reproducibility
np.random.seed(42)

# Generate dataset
X, y = generate_classification_dataset(200)

# Instantiate perceptrons with different probabilities
perceptron1 = Perceptron(iterations=100, learning_rate=0.01, error_iterations=5, probability=1, favor_class=1)
perceptron2 = Perceptron(iterations=100, learning_rate=0.01, error_iterations=5, probability=0.7, favor_class=1)
perceptron3 = Perceptron(iterations=100, learning_rate=0.01, error_iterations=5, probability=0.3, favor_class=1)

# Train all perceptrons
perceptron1.train(X, y)
perceptron2.train(X, y)
perceptron3.train(X, y)

# Predict new classifications
predictions1 = np.array([1 if perceptron1.predict(sample) == 1 else 0 for sample in X])
predictions2 = np.array([1 if perceptron2.predict(sample) == 1 else 0 for sample in X])
predictions3 = np.array([1 if perceptron3.predict(sample) == 1 else 0 for sample in X])

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ['blue', 'orange']

# Define probability titles
probabilities = [1, 0.7, 0.3]
predictions = [predictions1, predictions2, predictions3]

for i, (prob, preds) in enumerate(zip(probabilities, predictions)):
    axes[i].set_title(f"Probability = {prob}")
    for class_id in np.unique(preds):
        class_points = X[preds == class_id]
        axes[i].scatter(class_points[:, 0], class_points[:, 1], c=colors[int(class_id)], edgecolors='k')

plt.show()
```

## Explanation
- This script initializes three perceptrons with different probability values for class selection.
- It trains all perceptrons on the same dataset and plots the resulting classifications.
- The graphs show how different probability settings affect classification.


