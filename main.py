import numpy as np
import matplotlib.pyplot as plt


class Perceptron():
    """
    A simple perceptron classifier with probabilistic activation selection.
    
    Attributes:
        iterations (int): Number of training iterations.
        learning_rate (float): Learning rate for weight updates.
        error_iterations (int): Minimum number of classification errors to continue training.
        weights (np.array): Weights for the perceptron.
        bias (float): Bias term.
        probability (float): Probability of favoring a certain class in ambiguous cases.
        favor_class (int): The class (-1 or 1) to favor based on probability.
        errors (list): Stores the error count at each iteration.
    """
    
    def __init__(self, iterations, learning_rate, error_iterations, probability, favor_class):
        """
        Initializes the perceptron with the given parameters.

        Args:
            iterations (int): Number of training iterations.
            learning_rate (float): Learning rate for weight updates.
            error_iterations (int): Minimum number of classification errors to continue training.
            probability (float): Probability of favoring a certain class.
            favor_class (int): The class to favor (0 or 1, converted to -1 or 1 internally).
        """
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.error_iterations = error_iterations
        self.weights = None
        self.bias = None
        self.probability = probability
        self.favor_class = -1 if favor_class == 0 else 1
        self.errors = []

    def input_layer(self, input):
        """
        Computes the weighted sum of inputs plus bias.

        Args:
            input (np.array): Input feature vector.

        Returns:
            float: Computed value from the input layer.
        """
        return np.dot(self.weights,input) + self.bias
    
    def activation_function(self, computed_value):
        """
        Applies the activation function to classify the input.
        Uses probabilistic selection to favor a specific class.

        Args:
            computed_value (float): The computed value from the input layer.

        Returns:
            int: Predicted class label (-1 or 1).
        """
        deterministic = 1 if computed_value >= 0 else -1
        if deterministic == self.favor_class:
            prediction = self.favor_class if np.random.rand() < self.probability else -self.favor_class
        else:
            prediction = deterministic
        return prediction
    
    def predict(self, input):
        """
        Predicts the class label for a given input.

        Args:
            input (np.array): Input feature vector.

        Returns:
            int: Predicted class label (-1 or 1).
        """
        return self.activation_function(self.input_layer(input))
    
    def loss_function(self, y, y_pred):
        """
        Computes the loss as the number of misclassifications.

        Args:
            y (int): True class label.
            y_pred (int): Predicted class label.

        Returns:
            int: 1 if misclassified, 0 otherwise.
        """
        return int(y != y_pred)
    
    def train(self, X, y):
        """
        Trains the perceptron using the given dataset.

        Args:
            X (np.array): Feature matrix.
            y (np.array): Target labels.
        """
        y_treated = np.where(y >= 1, 1, -1)
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.iterations):
            total_error = 0
            for row_X, i_y in zip(X, y_treated):
                y_pred = self.predict(row_X)
                if i_y != y_pred:
                    self.weights += self.learning_rate * i_y * row_X
                    self.bias += self.learning_rate * i_y
                total_error += self.loss_function(i_y, y_pred)
            if len(self.errors) > 5 and np.mean(self.errors[-5:]) <= self.error_iterations:
                print('Early Stopped.')
                break

def get_inputs():
    """
    Prompts the user to enter training parameters.

    Returns:
        tuple: (iterations, error_change, probability_check, favor_class, learning_rate, amostras)
    """
    iterations = int(input('Number of iterations: '))
    error_change = int(input('Average last 5 mismatch classifications: '))
    probability_check = float(input('Probability to choose desired class: '))
    favor_class = int(input('Which class to favor based on probability (0 or 1): '))
    learning_rate = float(input('Learning Rate: '))
    samples = int(input('Number of rows of dataset: '))
    return iterations, error_change, probability_check, favor_class, learning_rate, samples

def generate_classification_dataset(n_rows):
    """
    Generates a synthetic classification dataset.

    Args:
        n_rows (int): Number of rows in the dataset.

    Returns:
        tuple: Feature matrix X and target labels y.
    """
    X_class1 = np.random.randn(n_rows // 2, 2) + np.array([-3, -3])
    X_class2 = np.random.randn(n_rows // 2, 2) + np.array([3, 3])
    X = np.vstack((X_class1, X_class2))
    y = np.hstack((np.zeros(n_rows // 2), np.ones(n_rows // 2)))
    return X, y

def plot_data(X, y, colors):
    """
    Plots the dataset with different colors for each class.

    Args:
        X (np.array): Feature matrix.
        y (np.array): Target labels.
        colors (list): List of colors for each class.
    """
    for class_id in np.unique(y).tolist():  
        class_points = X[y == class_id]
        plt.scatter(class_points[:, 0], class_points[:, 1], 
                    label=f'Class {class_id}',  
                    alpha=0.6, 
                    edgecolors='k', 
                    color=colors[int(class_id)])
    plt.legend()
    plt.show()
    return

if __name__ == "__main__":
    # Seed for reproducibility
    np.random.seed(42)

    # Get inputs from the user
    iterations, error_change, probability_check, favor_class, learning_rate, rows = get_inputs()

    # Generate dataset
    X, y = generate_classification_dataset(rows)

    # Initialize class
    pp = Perceptron(iterations, learning_rate, error_change,probability_check,favor_class)

    # Train model
    pp.train(X,y)
    
    # Class colors to distinguish
    class_colors = ['red', 'green'] 
    class_colors_predict = ['blue', 'orange']

    # Plot the first graph - original classes
    plot_data(X, y, class_colors)

    # Plot the second graph - predicted classes
    new_classes = np.array([1 if pp.predict(input) == 1 else 0 for input, _ in zip(X, y)])
    plot_data(X, new_classes, class_colors_predict)
