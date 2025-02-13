import numpy as np
import matplotlib.pyplot as plt


def input_layer(weights,bias,values):
    '''
    First layer that can calculate the computation between weights, values and bias
    '''
    result = sum(weights*values) + bias
    return result

def activation_function(comp_values, probability=0.5):
    '''
    Activation function that defines if the output from the input is 0 or 1.
    Classification layer.
    '''
    return np.where(1 / (1 + np.exp(-comp_values)) >= probability,1,0)

def loss_function(y,y_pred):
    '''
    Calculate the error between real and prediciton value.
    In this case it will always output 0 or 1
    '''
    return (y - y_pred)**2

def weight_update(weights,bias,y,y_pred,values,learning_rate):
    '''
    Updates weights based on the learning rate given, the error and the values
    '''
    return weights + learning_rate*(y-y_pred)*values, bias + learning_rate*(y-y_pred)

def compare_errors(errors, i, error_rate):
    '''
    Function that allows to evaluate if we achieved the user-defined error.
    If the error_rate of the user is 1, it means that the user allows the difference between iterations to be 1.
    '''
    prev_error = errors[i-1]
    actual_error = errors[i]
    return (actual_error - prev_error) <= error_rate


def get_inputs():
    '''
    Function to get inputs from user.
    '''
    iterations = int(input('Número iterações: '))
    error_change = int(input('Número de erros na classificação mínimo para parar: '))
    probability_check = float(input('Probabilidade de escolher 1: '))
    learning_rate = float(input('Learning Rate: '))
    return iterations, error_change, probability_check, learning_rate

def generate_dataset():
    '''
    Funtion to generate the dataset for study.
    '''
    n_rows = int(input('Número de linhas: '))
    X_class1 = np.random.randn(n_rows // 2, 2) + np.array([-3, -3])
    X_class2 = np.random.randn(n_rows // 2, 2) + np.array([3, 3])
    # Vertical stack - one on top of the other
    X = np.vstack((X_class1, X_class2))
    # Horizontal stack - one on top of the other
    y = np.hstack((np.zeros(n_rows // 2), np.ones(n_rows // 2)))
    return X, y


if __name__ == "__main__":
    np.random.seed(42)
    iterations, error_change, probability_check, learning_rate = get_inputs()
    X, y = generate_dataset()
    weights = np.zeros(X.shape[1])
    bias = np.random.uniform(0,1)
    error = []
    for i in range(iterations):
        total_error = 0
        for input, target in zip(X,y):
            comp_values = input_layer(weights,bias,input)
            y_pred = activation_function(comp_values,probability_check)
            weights,bias = weight_update(weights,bias,target,y_pred,input,learning_rate)
            total_error += loss_function(target,y_pred)
        error.append(total_error)
        if i >= 1 and compare_errors(error, i, error_change):
            break
    # Define class colors (this is optional but can help to manually specify colors)
    class_colors = ['red', 'green']  # Red, Green

    class_colors_final = ['blue', 'orange']

    # Plot each class with a different color
    for class_id in np.unique(y).tolist():  # Iterate over unique class labels
        # Select points of the current class
        class_points = X[y == class_id]
        
        # Scatter plot for the current class
        plt.scatter(class_points[:, 0], class_points[:, 1], 
                    label=f'Class {class_id}',  # Label for the legend
                    alpha=0.6, 
                    edgecolors='k', 
                    color=class_colors[int(class_id)])
    
    # Add the legend
    plt.legend()

    # Display the plot
    plt.show()


    new_classes = np.array([activation_function(input_layer(weights,bias,input),probability_check) for input,target in zip(X,y)])

    for class_id in [0,1]:  # Iterate over unique class labels
        # Select points of the current class
        class_points = X[new_classes == class_id]
        
        # Scatter plot for the current class
        plt.scatter(class_points[:, 0], class_points[:, 1], 
                    label=f'New Class {class_id}',  # Label for the legend
                    alpha=0.6, 
                    edgecolors='k', 
                    color=class_colors_final[int(class_id)])
    plt.legend()

    # Display the plot
    plt.show()
