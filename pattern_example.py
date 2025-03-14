from neural_network import NeuralNetwork

# Training data for a simple pattern recognition task
# The network will learn to recognize if the sum of inputs is greater than 1.5
training_data = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.5, 0.5],
    [0.2, 0.8],
    [0.8, 0.2],
    [0.3, 0.7],
    [0.7, 0.3],
    [0.4, 0.6]
]

# Target outputs (1 if sum > 1.5, 0 otherwise)
targets = [
    [0],  # 0.0 + 0.0 = 0.0
    [0],  # 0.0 + 1.0 = 1.0
    [0],  # 1.0 + 0.0 = 1.0
    [1],  # 1.0 + 1.0 = 2.0
    [0],  # 0.5 + 0.5 = 1.0
    [0],  # 0.2 + 0.8 = 1.0
    [0],  # 0.8 + 0.2 = 1.0
    [0],  # 0.3 + 0.7 = 1.0
    [0],  # 0.7 + 0.3 = 1.0
    [0],  # 0.4 + 0.6 = 1.0
]

# Create a neural network with 2 inputs, 3 hidden neurons, and 1 output
nn = NeuralNetwork([2, 3, 1])

# Train the network
print("Training the neural network...")
nn.train(training_data, targets, learning_rate=0.1, epochs=2000)

# Test the network with some new examples
test_data = [
    [0.9, 0.9],  # Should be 1 (sum = 1.8)
    [0.6, 0.6],  # Should be 0 (sum = 1.2)
    [1.0, 0.7],  # Should be 1 (sum = 1.7)
    [0.4, 0.8],  # Should be 0 (sum = 1.2)
]

print("\nTesting with new data:")
for inputs in test_data:
    prediction = nn.predict(inputs)
    print(f"Inputs: {inputs}, Sum: {sum(inputs):.1f}, Predicted Output: {prediction[0]:.4f}")
