import numpy as np
import time


def linear_gradient_descent_minimizer(
    fn, gradient_fn, initial_point, precission=1e-4, num_iterations=10000, bounds=None
):
    point = initial_point
    loop_counter = 0
    in_bounds = False
    print(f"Initial point, {point}")
    gradient = gradient_fn(point)
    gradient_norm = np.linalg.norm(gradient)

    while gradient_norm > precission:
        gradient = gradient_fn(point)
        gradient_norm = np.linalg.norm(gradient)
        point = point - fn(point) / (gradient + 1e-5)
        if bounds:
            for i, (p, b) in enumerate(zip(point, bounds)):
                if p >= b[1]:
                    point[i] = b[1]
                    in_bounds = True
                elif p <= b[0]:
                    point[i] = b[0]
                    in_bounds = True
        loop_counter += 1
        print(loop_counter)

        if loop_counter > num_iterations:
            print(
                "WARNING: Too many iterations. The precission required may be too low, the number of number of iterations too low, or the minimizer is diverging."
            )
            print(f"Gradient norm is {gradient_norm}")
            return point
    print(f"Achieved minimum in {loop_counter} iterations.")
    return point


# Example usage:
def function_to_minimize(x):
    # Define your function here
    return x[0] ** 2 + 3 * x[1] ** 2 + 1
    # print(r'Example function: $x^2$')


def gradient_function(x):
    # Calculate the derivative of the function here
    return np.array([2 * x[0], 6 * x[1]])  # Derivative of x^2 is 2x


initial_point = np.array([1.0, 10.0])  # Initial point for the minimization
bounds = ((-1, 2), (-2, 11))
learning_rate = 0.1  # Learning rate determines the step size
num_iterations = 100  # Number of iterations for gradient descent

start = time.perf_counter()
minimized_point = linear_gradient_descent_minimizer(
    function_to_minimize, gradient_function, initial_point, bounds=bounds
)
print(f"It took {time.perf_counter()-start} seconds")
print(minimized_point)
minimized_value = function_to_minimize(minimized_point)
print(minimized_value)
