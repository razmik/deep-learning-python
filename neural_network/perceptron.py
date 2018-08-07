import numpy as np


class NeuralNetwork:

    def __init__(self):

        # Once seed, every time this runds, it makes sure same random number
        # is being generated
        np.random.seed(1)

        # we model a single neuron with 3 input and 1 output.
        # We assign random weights to 3 x 1 matrix with values ranging -1 to 1
        # and mean 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    # the sigmoid function, which describe as s shaped curve
    # we pass the wighted sum of the inputs through this function
    # to normlaize between 0 and 1
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # pass inputs directly through the single neural network
    def predict(self, inputs):
        return self._sigmoid(np.dot(inputs, self.synaptic_weights))

    # gradient of the sigmoid curve
    def _sigmoid_derivative(self, x):
        return x * (1-x)

    def train(self, training_set_inputs, training_set_outputs, number_of_iterations):
        for _ in np.arange(number_of_iterations):

            output = self.predict(training_set_inputs)

            error = training_set_outputs - output

            adjustment = np.dot(training_set_inputs.T, error * self._sigmoid_derivative(output))

            self.synaptic_weights += adjustment


if __name__ == "__main__":

    #iniitalize a single neural network
    neural_network = NeuralNetwork()

    print('Random starting synaptic weights:')
    print(neural_network.synaptic_weights)

    training_set_inputs = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    training_set_outputs = np.array([[0,1,1,0]]).T

    neural_network.train(training_set_inputs, training_set_outputs, 1000)

    print('New synaptic weights after training:')
    print(neural_network.synaptic_weights)

    print('Predicting')
    print(neural_network.predict(np.array([1, 0, 0])))