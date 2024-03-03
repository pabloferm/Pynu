import numpy as np


def gradient_descent_minimizer(
        f,
        initial_point,
        args=None,
        learning_rate=1e-5,
        epsilon=1e-4,
        num_iterations=100,
        bounds=None):
    point = initial_point
    loop_counter = 0
    in_bounds = False
    print(f'Initial point, {point}')
    if args:
        X2, gradient = f(point, args)
        print(X2)
        print(gradient)
    else:
        X2, gradient = f(point)
    gradient_norm = np.linalg.norm(gradient)
    print(f'Gradient norm is {gradient_norm}')
    print(epsilon)
    while gradient_norm > epsilon:
        if args:
            X2, gradient = f(point, args)
        else:
            X2, gradient = f(point)
        gradient_norm = np.linalg.norm(gradient)
        print(
            f'point will be modified by substracting {learning_rate * gradient}')
        point = point - learning_rate * gradient
        if bounds:
            for i, (p, b) in enumerate(zip(point, bounds)):
                if p >= b[1]:
                    point[i] = b[1]
                    in_bounds = True
                elif p <= b[0]:
                    point[i] = b[0]
                    in_bounds = True
                else:
                    in_bounds = False

        print(gradient)
        print(f'New point coser to the minimum, {point}')
        loop_counter += 1
        if loop_counter > num_iterations:
            print('WARNING: Too many iterations. The precission required may be too low, the number of number of iterations too low, or the minimizer is diverging.')
            print(f'Gradient norm is {gradient_norm}')
            if in_bounds:
                print('WARNING: minimum at bounds.')
            return point
    print(f'Achieved minimum in {loop_counter} iterations.')
    if in_bounds:
        print('WARNING: minimum at bounds.')
    return point
