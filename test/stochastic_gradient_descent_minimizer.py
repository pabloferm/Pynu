import time
import numpy as np


def function(params):
    # Define your objective function here
    x, y = params
    return x**2 + y**2  # Example function: x^2 + y^2


def gradient_function(params):
    x, y = params
    print(params)
    return [2 * x, 2 * y]


def stochastic_gradient_descent(initial_guess, learning_rate, num_iterations):
    # Initialize parameters
    params = initial_guess
    nen = len(initial_guess)

    for iteration in range(num_iterations):
        # Randomly sample a data point or minibatch
        data_point = np.random.randn(nen)  # Example data point

        # Calculate the gradient of the objective function
        gradient = gradient_function(data_point)

        # Update the parameters using SGD update rule
        params -= learning_rate * gradient

    return params


# Usage example
learning_rate = 0.1
num_iterations = 1000
initial_guess = [-5, 7]
start = time.perf_counter()
optimized_params = stochastic_gradient_descent(
    initial_guess, learning_rate, num_iterations
)
print(f"It took {time.perf_counter()-start} seconds")
print("Optimized parameters:", optimized_params)
print("Objective function value:", function(optimized_params))
