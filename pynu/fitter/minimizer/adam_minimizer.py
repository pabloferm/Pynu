import numpy as np


def adam_minimizer(
    f,
    initial_point,
    args=None,
    learning_rate=0.01,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    precission=1e-4,
    num_iterations=100,
    bounds=None,
):
    # Initialize variables
    point = initial_point
    m = np.zeros_like(point)
    v = np.zeros_like(point)
    t = 0
    print(f"Initial point, {point}")
    if args:
        X2, gradient = f(point, args)
        print(X2)
        print(gradient)
    else:
        X2, gradient = f(point)
    gradient_norm = np.linalg.norm(gradient)
    print(f"Gradient norm is {gradient_norm}")

    while gradient_norm > precission:
        # for _ in range(num_iterations):
        t += 1

        if args:
            X2, gradient = f(point, args)
        else:
            X2, gradient = f(point)

        # Compute first and second moment estimates
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient**2)

        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # Update point
        print(learning_rate * m_hat / (np.sqrt(v_hat) + epsilon))
        point -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        print(point)
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
        print(f"New point coser to the minimum, {point}")
        gradient_norm = np.linalg.norm(gradient)
        if t > num_iterations:
            print(
                "WARNING: Too many iterations. The precission required may be too low, the number of number of iterations too low, or the minimizer is diverging."
            )
            print(f"Gradient norm is {gradient_norm}")
            return point

    if in_bounds:
        print("WARNING: minimum at bounds.")

    return point
