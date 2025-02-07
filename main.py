import numpy as np

def computation(weights,bias,values):
    result = weights*values + bias
    return result

def activation_function(comp_values):
    return 1 / (1 + np.exp(-comp_values))



def get_inputs():
    iterations = int(input('Número iterações: '))
    error_change = float(input('Error between iterations detection: '))
    probability_check = float(input('Probabilidade to choose one over the other: '))
    learning_rate = float(input('Learning Rate: '))
    return iterations, error_change, probability_check


if __name__ == "__main__":
    np.random.seed(42)
    values = np.random.randint(10,100,10)
    weights = np.random.uniform(-1,1,len(values))
    bias = np.random.uniform(0,1)
    print(values)
    print(weights)
    computed_values = computation(weights,bias,values)
    print(computed_values)
    print(activation_function(computed_values))
    

