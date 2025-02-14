import numpy as np
import matplotlib.pyplot as plt


class Perceptron():

    def __init__(self, iterations, learning_rate, error_iterations, probability, favor_class):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.error_iterations = error_iterations
        self.weights = None
        self.bias = None
        self.probability = probability
        self.favor_class = -1 if favor_class == 0 else 1
        self.errors = []

    def input_layer(self, input):
        '''
        First layer that can calculate the computation between weights, values and bias
        '''
        return sum(self.weights*input) + self.bias
    
    def activation_function(self, computed_value):
        '''
        Activation function that defines if the output from the input is -1 or 1.
        Classification layer.
        '''
        deterministic = 1 if computed_value >= 0 else -1
        if deterministic == self.favor_class:
            prediction = self.favor_class if np.random.rand() < self.probability else -self.favor_class
        else:
            prediction = deterministic
        return prediction
    
    def predict(self, input):
        return self.activation_function(self.input_layer(input))
    
    def loss_function(self, y, y_pred):
        '''
        Calculate the error between real and prediciton value.
        In this case it will always output 0 or 1
        '''
        return int(y != y_pred)
    
    def train(self, X, y):
        y_treated = np.where(y>=1,1,-1)
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.iterations):
            total_error = 0
            for row_X, i_y in zip(X,y_treated):
                y_pred = self.predict(row_X)
                self.weights += self.learning_rate*y_pred*row_X
                self.bias += self.learning_rate*y_pred
                total_error += self.loss_function(i_y,y_pred)
            self.errors.append(total_error)
            if total_error <= self.error_iterations:
                print('Early Stopped.')
                break

def get_inputs():
    '''
    Function to get inputs from user.
    '''
    iterations = int(input('Número iterações: '))
    error_change = int(input('Número de erros na classificação mínimo para parar: '))
    probability_check = float(input('Probabilidade de escolher 1: '))
    favor_class = int(input('Which class to favor based on probability (0 or 1): '))
    learning_rate = float(input('Learning Rate: '))
    amostras = int(input('Número de linhas: '))
    return iterations, error_change, probability_check, favor_class, learning_rate, amostras

def generate_classification_dataset(n_rows):
    '''
    Funtion to generate the dataset for study.
    '''
    X_class1 = np.random.randn(n_rows // 2, 2) + np.array([-3, -3])
    X_class2 = np.random.randn(n_rows // 2, 2) + np.array([3, 3])
    # Vertical stack - one on top of the other
    X = np.vstack((X_class1, X_class2))
    # Horizontal stack - one on top of the other
    y = np.hstack((np.zeros(n_rows // 2), np.ones(n_rows // 2)))
    return X, y


def plot_data(X, y, colors):
    ''''''
    # Plot each class with a different color
    for class_id in np.unique(y).tolist():  # Iterate over unique class labels
        # Select points of the current class
        class_points = X[y == class_id]
        
        # Scatter plot for the current class
        plt.scatter(class_points[:, 0], class_points[:, 1], 
                    label=f'Class {class_id}',  # Label for the legend
                    alpha=0.6, 
                    edgecolors='k', 
                    color=colors[int(class_id)])
    
    # Add the legend
    plt.legend()

    # Display the plot
    plt.show()
    return

if __name__ == "__main__":
    np.random.seed(42)
    iterations, error_change, probability_check, favor_class, learning_rate, linhas = get_inputs()
    X, y = generate_classification_dataset(linhas)
    pp = Perceptron(iterations, learning_rate, error_change,probability_check,favor_class)
    pp.train(X,y)
    
    class_colors = ['red', 'green'] 
    class_colors_train = ['blue', 'orange']

    plot_data(X, y, class_colors)
    
    new_classes = np.array([1 if pp.predict(input) == 1 else 0 for input, _ in zip(X, y)])
    plot_data(X, new_classes, class_colors_train)
    
