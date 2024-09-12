import numpy as np
import time


def adam_minimizer(
    parameters,
    gradient,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    precission=1e-4,
    num_iterations=100,
    bounds=None,
):
    # Initialize variables
    m = np.zeros_like(parameters)
    v = np.zeros_like(parameters)
    t = 0

    gradients = gradient(parameters)
    gradient_norm = np.linalg.norm(gradients)

    while gradient_norm > precission:
        # for _ in range(num_iterations):
        t += 1
        # Compute first and second moment estimates
        m = beta1 * m + (1 - beta1) * gradients
        v = beta2 * v + (1 - beta2) * (gradients**2)

        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # Update parameters
        parameters -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        gradient_norm = np.linalg.norm(gradients)
        if t > num_iterations:
            print(
                "WARNING: Too many iterations. The precission required may be too low, the number of number of iterations too low, or the minimizer is diverging."
            )
            print(f"Gradient norm is {gradient_norm}")
            return parameters
        gradients = gradient(parameters)

    return parameters


# Example usage:
def function_to_minimize(x):
    # Define your function here
    return x[0] ** 2 + (3 * x[1] - 1) ** 2 + 8
    # print(r'Example function: $x^2$')


def gradient_function(x):
    # Calculate the derivative of the function here
    return np.array([2 * x[0], 6 * (3 * x[1] - 31)])  # Derivative of x^2 is 2x


initial_point = np.array([1.0, 10.0])  # Initial point for the minimization
bounds = ((-1, 2), (9, 11))
learning_rate = 0.1  # Learning rate determines the step size

start = time.perf_counter()
minimized_point = adam_minimizer(
    initial_point, gradient_function, learning_rate=learning_rate
)
print(f"It took {time.perf_counter()-start} seconds")
print(minimized_point)
minimized_value = function_to_minimize(minimized_point)
print(minimized_value)
