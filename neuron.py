# This is a simple implementation of a neuron in Python
# with basic operations and a training loop.
# CODING OUR FIRST NEURON: 3 INPUTS

def firstNeuron():
    inputs = [1, 2, 3]
    weights = [0.2, 0.8, -0.5]
    bias = 2
    outputs = (inputs[0]*weights[0] + inputs[1] *
               weights[1] + inputs[2]*weights[2] + bias)
    print(outputs)


def neuronWithFourLayers():
    # CODING OUR FIRST NEURON: 4
    # INPUTS, 3 NEURONS, 4 LAYERS
    # Step 1: Define inputs, weights, and biases
    # Inputs: 4 inputs
    # Weights: 3 neurons, each with 4 weights
    # Biases: 3 neurons, each with 1 bias
    inputs = [1, 2, 3, 2.5]
    weights = [[0.2, 0.8, -0.5, 1], [0.5, -0.91,
                                     0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
    # LIST OF WEIGHTS ASSOCIATED WITH 1ST NEURON : W11, W12, W13, W14
    weights1 = weights[0]
    # LIST OF WEIGHTS ASSOCIATED WITH 2ND NEURON : W21, W22, W23, W24
    weights2 = weights[1]
    # LIST OF WEIGHTS ASSOCIATED WITH 3RD NEURON : W31, W32, W33, W34
    weights3 = weights[2]
    biases = [2, 3, 0.5]
    bias1 = biases[0]
    bias2 = biases[1]
    bias3 = biases[2]
    outputs = [
        # Neuron 1:
        inputs[0]*weights1[0] +
        inputs[1]*weights1[1] +
        inputs[2]*weights1[2] +
        inputs[3]*weights1[3] + bias1,
        # Neuron 2:
        inputs[0]*weights2[0] +
        inputs[1]*weights2[1] +
        inputs[2]*weights2[2] +
        inputs[3]*weights2[3] + bias2,
        # Neuron 3:
        inputs[0]*weights3[0] +
        inputs[1]*weights3[1] +
        inputs[2]*weights3[2] +
        inputs[3]*weights3[3] + bias3]
    print(outputs)


def compute_layer_output(inputs, weights, biases):
    """
    Computes the output of a layer of neurons.

    Parameters:
    - inputs (list of floats): Input values to the layer.
    - weights (list of list of floats): Weights for each neuron in the layer.
    - biases (list of floats): Biases for each neuron in the layer.

    Returns:
    - list of floats: Outputs of each neuron.
    """
    layer_outputs = []

    for neuron_weights, neuron_bias in zip(weights, biases):
        neuron_output = 0
        for n_input, weight in zip(inputs, neuron_weights):
            neuron_output += n_input * weight
        neuron_output += neuron_bias
        layer_outputs.append(neuron_output)

    return layer_outputs

# Example usage
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

output = compute_layer_output(inputs, weights, biases)
print(output)
