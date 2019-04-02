#########################################
#
# Neural Network Example with numpy array
#
#########################################
from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        random.seed(1)
        self.synaptic_weights = 2 * random.random((3,1)) - 1

    def train(self, trainning_set_inputs, trainning_set_outputs, number_of_training_interactions):
        for interation in xrange(number_of_training_interactions):
            output = self.predict(trainning_set_inputs)

            error = trainning_set_outputs - output

            adjustment = dot(trainning_set_inputs.T, error * self._sigmoid_derivate(output))

            self.synaptic_weights += adjustment

    def _sigmoid_derivate(self, x):
        return x * (1-x)

    def _sigmoid(self, x):
        return 1 / ( 1 + exp(-x) )

    def predict(self, inputs):
        return self._sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == '__main__':

    neural_network = NeuralNetwork()

    print 'Random stating synaptic weights:'
    print neural_network.synaptic_weights

    trainning_set_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,0,1]])
    trainning_set_outputs = array([[0,1,1,0]]).T

    neural_network.train(trainning_set_inputs, trainning_set_outputs, 1)

    print 'New synaptic weights after training:'
    print neural_network.synaptic_weights

    print 'Predicting'

    print neural_network.predict(array([1,0,0]))
