"""
Gradient descent is an optimization algorithm used to find the values of parameters (coefficients)
of a function (f) that minimizes a cost function (cost).
Gradient descent is best used when the parameters cannot be calculated analytically (e.g. using linear algebra)
and must be searched for by an optimization algorithm.

Gradient decent: http://machinelearningmastery.com/gradient-descent-for-machine-learning/
"""

from numpy import *

"""
The optimal values of m and b can be actually calculated with way less effort than doing a linear regression.
this is just to demonstrate gradient descent
"""

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_given_point(b, m, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y- (m * x + b)) ** 2
    return total_error / float(len(points))


def step_gradient(b_current, m_current, points, learning_rate):
    # gradient descent
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) *x *  (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
        cost = compute_error_for_given_point(b, m, points)
        print("Iteration {0} is b = {1}, m = {2}, cost = {3}".format((i+1), b, m, cost))

    return [b, m]


def run():
    points = genfromtxt("..\data\student_data.csv", delimiter=',')

    """
    hyperparameter - used to tune the machine learning model
    if LR is too low - take so long to convert
    if LR is high - never converge
    """
    learning_rate = 0.0001

    # y = mx +b
    initial_b = 0
    initial_m = 0

    # since data set is small, can train a small num of iterations
    num_iterations = 100000

    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_given_point(initial_b, initial_m, points)))
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_given_point(b, m, points)))

if __name__ == '__main__':
    run()