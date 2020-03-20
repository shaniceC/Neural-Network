import numpy as np
import math
from sklearn.model_selection import train_test_split

filename = 'hw_data.txt'
write_file = 'hw_output.txt'
weights_1 = np.array([[0.1, 0.4],
                      [0.3, 0.3],
                      [0.4, 0.2]])
weights_2 = np.array([[0.1, 0.3], 
                      [0.3, 0.1]])

bias = np.array([[1.0],
                 [1.0]])

bias_weights = np.array([[0.8, 0.8],
                         [1.0, 1.0]])

learning_rate = .5

def read_data():
    try:
        with open(filename) as f_obj:
            lines = f_obj.readlines()
            return lines

    except FileNotFoundError:
        print("File Not Found")
        return None

def split_data(lines):
    """Split data into train and test sets."""
    
    num_examples = len(lines)
    input_array = np.zeros((num_examples, 3))
    output_array = np.zeros((num_examples, 2))

    index = 0
    for line in lines:
        in1, in2, in3, out = line.split(" ")

        # minimum = min(float(in1), float(in2), float(in3))
        # maximum = max(float(in1), float(in2), float(in3))

        # range1 = maximum - minimum
        # in1 = (float(in1) - minimum) / range1
        # in2 = (float(in2) - minimum) / range1
        # in3 = (float(in3) - minimum) / range1

        # range2 = 1 - (-1)
        # in1 = (in1 * range2) + (-1)
        # in2 = (in2 * range2) + (-1)
        # in3 = (in3 * range2) + (-1)

        input_array[index][0] = float(in1)
        input_array[index][1] = float(in2)
        input_array[index][2] = float(in3)
        if float(out) == 1:
            output_array[index][0] = float(out)
            output_array[index][1] = float(0)
        else:
            output_array[index][0] = float(0)
            output_array[index][1] = float(1)

        index = index + 1

    i_train, i_test, o_train, o_test = train_test_split(input_array, output_array, train_size=.80, test_size=.20)
    return i_train, i_test, o_train, o_test

def activation(node_net_input):
    """Apply the sigmoid activation function to the net input of a node."""
    # Activation function is the sigmoid function

    node_output = 1 / (1 + math.exp(-1 * node_net_input))

    return node_output

def forward_pass(inputs):
    """Perform the forward pass of the neural network"""

    # first layer
    in1 = inputs.dot(weights_1)
    net_input1 = in1
    for i in range(in1.size):
        net_input1[i] = in1[i] + (bias[0] * bias_weights[0][i])
    # net_input1 = in1 + (bias[0] * bias_weights[0])
    net_output1 = net_input1
    
    for i in range(in1.size):
        net_output1[i] = activation(net_input1[i])

    # second layer
    # in2 = weights_2.dot(net_output1)
    in2 = net_output1.dot(weights_2)
    net_input2 = in2
    for i in range(in2.size):
        net_input2[i] = in2[i] + (bias[1] * bias_weights[1][i])
    # net_input2 = in2 + (bias[1] * bias_weights[1])
    net_output2 = net_input2
    for i in range(in2.size):
        net_output2[i] = activation(net_input2[i])

    return net_output1, net_output2

def calculate_loss(outputs, targets):
    """Calculate the loss for the results of the forward pass."""
    # Loss function: L = (o1 - o1')^2 + (o2 - o2')^2

    l1 = (targets[0] - outputs[0]) * (targets[0] - outputs[0])
    l2 = (targets[1] - outputs[1]) * (targets[1] - outputs[1])

    loss = l1 + l2

    return loss

def calculate_output_layer_gradient(network_output, target_output, node_output):
    """Calculate the gradient of an output layer weight."""
    deltaz = calculate_deltaz(network_output, target_output)
    gradient = deltaz * node_output

    return gradient

def calculate_hidden_layer_gradient(network_outputs, target_outputs, weight_1, weight_2, node_output, network_input):
    part_deriv_loss_out1 = calculate_deltaz(network_outputs[0], target_outputs[0]) * weight_1
    part_deriv_loss_out2 = calculate_deltaz(network_outputs[1], target_outputs[1]) * weight_2
    part_deriv_out_net = node_output * (1 - node_output)
    part_deriv_net_weight = network_input

    gradient = (part_deriv_loss_out1 + part_deriv_loss_out2) * part_deriv_out_net * part_deriv_net_weight
    return gradient

def calculate_hidden_layer_bias_gradient(network_outputs, target_outputs, weight_1, weight_2, node_output):
    part_deriv_loss_out1 = calculate_deltaz(network_outputs[0], target_outputs[0]) * weight_1
    part_deriv_loss_out2 = calculate_deltaz(network_outputs[1], target_outputs[1]) * weight_2
    part_deriv_out_net = node_output * (1 - node_output)

    gradient = (part_deriv_loss_out1 + part_deriv_loss_out2) * part_deriv_out_net
    return gradient


def calculate_deltaz(network_output, target_output):
    part_deriv_loss = 2 * (network_output - target_output)
    part_deriv_output = network_output * (1 - network_output)
    deltaz = part_deriv_loss * part_deriv_output

    return deltaz

def calculate_gradients(network_inputs, network_outputs, targets, layer1_node_outputs):
    """Calculate the gradients of all of the weights."""
    output_layer_gradient_array = np.zeros((2, 2))
    hidden_layer_gradient_array = np.zeros((3, 2))
    output_layer_bias_gradients = np.zeros((1, 2))
    hidden_layer_bias_gradients = np.zeros((1, 2))

    # compute output layer gradients
    for i in range(np.size(output_layer_gradient_array, 1)):
        for j in range(np.size(output_layer_gradient_array, 0)):
            gradient = calculate_output_layer_gradient(network_outputs[i], targets[i], layer1_node_outputs[j])
            output_layer_gradient_array[j][i] = gradient
        output_layer_bias_gradients[0][i] = calculate_deltaz(network_outputs[i], targets[i])
    
    # compute hidden layer gradients
    for i in range(np.size(hidden_layer_gradient_array, 1)):
        for j in range(np.size(hidden_layer_gradient_array, 0)):
            gradient = calculate_hidden_layer_gradient(network_outputs, targets, weights_2[i][0], weights_2[i][1], layer1_node_outputs[i], network_inputs[j])
            hidden_layer_gradient_array[j][i] = gradient
        hidden_layer_bias_gradients[0][i] = calculate_hidden_layer_bias_gradient(network_outputs, targets, weights_2[i][0], weights_2[i][1], layer1_node_outputs[i])

    return output_layer_gradient_array, hidden_layer_gradient_array, output_layer_bias_gradients, hidden_layer_bias_gradients


def update_weights(weights, gradients):
    """Update the weights"""
    # Updating by using Stochastic Gradient Descent

    for i in range(np.size(weights, 0)):
        for j in range(np.size(weights, 1)):
            weights[i][j] = weights[i][j] - learning_rate * gradients[i][j]

    return weights

def update_bias_weights(weights, gradients):
    """Update the bias weights"""
    # Updating by using Stochastic Gradient Descent

    for i in range(weights.size):
        weights[i] = weights[i] - (learning_rate * gradients[i])

    return weights

def back_prop(network_inputs, network_outputs, targets, layer1_node_outputs):
    """Perform back propogation of the neural network."""
    global weights_1, weights_2, bias_weights
    output_layer_gradient_array = np.zeros((2, 2))
    hidden_layer_gradient_array = np.zeros((3, 2))
    output_layer_bias_gradient_array = np.zeros((1, 2))
    hidden_layer_bias_gradient_array = np.zeros((1, 2))

    # Calculate gradients
    output_layer_gradient_array, hidden_layer_gradient_array, output_layer_bias_gradient_array, hidden_layer_bias_gradient_array = calculate_gradients(network_inputs, network_outputs, targets, layer1_node_outputs)

    # Update weights
    update_weights(weights_1, hidden_layer_gradient_array)
    update_weights(weights_2, output_layer_gradient_array)
    update_bias_weights(bias_weights[0], hidden_layer_bias_gradient_array[0])
    update_bias_weights(bias_weights[1], output_layer_bias_gradient_array[0])

def train(train_set_inputs, train_set_labels):
    """Train the network with the training set."""
    for i in range(np.size(train_set_labels, 0)):
        print("\n")
        print(train_set_inputs[i])
        # Forward pass
        layer1_outputs, network_outputs = forward_pass(train_set_inputs[i])
        print("\nOutput: " + str(network_outputs))
        writeToFile("\nOutput: " + str(network_outputs))
        writeToFile("\tLabel: " + str(train_set_labels[i]))
        print("Label: " + str(train_set_labels[i]))
        loss = calculate_loss(network_outputs, train_set_labels[i])
        print("Loss: " + str(loss))
        writeToFile("\tLoss: " + str(loss))

        # Back propogation
        back_prop(train_set_inputs[i], network_outputs, train_set_labels[i], layer1_outputs)

def calculate_accuracy(truePos, falsePos, trueNeg, falseNeg):
    """Calculate the accuracy of the network."""
    accuracy = (truePos + trueNeg)/(truePos + trueNeg + falsePos + falseNeg)

    return accuracy

def test(test_set_inputs, test_set_labels):
    """Test the network with the test set."""
    tp = 0
    fp = 0
    tn = 0
    fn = 0


    for i in range(np.size(test_set_labels, 0)):
        layer1_outputs, network_outputs = forward_pass(test_set_inputs[i])
        print("\nOutput: " + str(network_outputs))
        print("Label: " + str(test_set_labels[i]))
        writeToFile("\nOutput: " + str(network_outputs))
        writeToFile("\tLabel: " + str(test_set_labels[i]))

        if test_set_labels[i][0] == 1.0:
            if network_outputs[0] > network_outputs[1]:
                tp = tp + 1
            else:
                fn = fn + 1
        elif test_set_labels[i][1] == 1.0:
            if network_outputs[1] > network_outputs[0]:
                tn = tn + 1
            else:
                fp = fp + 1

    accuracy = calculate_accuracy(tp, fp, tn, fn)
    print("\n__________________")
    print("Accuracy: " + str(accuracy))
    writeToFile("\n__________________")
    writeToFile("\nAccuracy: " + str(accuracy))

def writeToFile(text):
    with open(write_file, 'a') as f:
        f.write(text)

def main():
    lines = read_data()
    i_train, i_test, o_train, o_test = split_data(lines)

    print("\tTraining")
    writeToFile("\nTraining")
    print("__________________")
    writeToFile("\n__________________")
    train(i_train, o_train)

    print("\n\tTesting")
    writeToFile("\n\nTesting")
    print("__________________")
    writeToFile("\n__________________")
    test(i_test, o_test)


if __name__ == "__main__":
    main()