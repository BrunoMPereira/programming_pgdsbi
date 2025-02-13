import numpy as np
import matplotlib.pyplot as plt


def input_layer(weights,bias,values):
    result = sum(weights*values) + bias
    return result

def activation_function(comp_values, probability=0.5):
    return np.where(1 / (1 + np.exp(-comp_values)) >= probability,1,0)

def loss_function(y,y_pred):
    return (y - y_pred)**2

def weight_update(weights,bias,y,y_pred,values,learning_rate):
    return weights + learning_rate*(y-y_pred)*values, bias + learning_rate*(y-y_pred)

def compare_errors(errors, i, error_rate):
    prev_error = errors[i-1]
    actual_error = errors[i]
    return (actual_error - prev_error) <= error_rate


def get_inputs():
    iterations = int(input('Número iterações: '))
    error_change = int(input('Número de erros na classificação mínimo para parar: '))
    probability_check = float(input('Probabilidade de escolher 1: '))
    learning_rate = float(input('Learning Rate: '))
    return iterations, error_change, probability_check, learning_rate

def generate_dataset():
    n_rows = int(input('Número de linhas: '))
    X_class1 = np.random.randn(n_rows // 2, 2) + np.array([-3, -3])
    X_class2 = np.random.randn(n_rows // 2, 2) + np.array([3, 3])
    X = np.vstack((X_class1, X_class2))
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
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6, edgecolors='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Scatter Plot of 2D Random Data")
    plt.show()