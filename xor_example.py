from neural_network import NeuralNetwork

# XOR training data
# Format: [input1, input2] -> [output]
training_data = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

# XOR expected outputs
targets = [
    [0],
    [1],
    [1],
    [0]
]

# Create a neural network with:
# - 2 input neurons (for the 2 inputs of XOR)
# - 4 hidden neurons
# - 1 output neuron
nn = NeuralNetwork([2, 4, 1])

# Train the network
print("Training the neural network...")
nn.train(training_data, targets, learning_rate=0.1, epochs=5000)

# Test the network
print("\nTesting the neural network:")
for inputs in training_data:
    prediction = nn.predict(inputs)
    print(f"Inputs: {inputs}, Predicted Output: {prediction[0]:.4f}")
