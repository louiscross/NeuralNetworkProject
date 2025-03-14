import math
import random
import json

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        Initialize neural network with given layer sizes.
        layer_sizes: List of integers, each number represents number of neurons in that layer
        """
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases between layers
        if not self.weights and not self.biases:  # Only initialize if empty
            for i in range(len(layer_sizes) - 1):
                # Initialize weights with small random values using Xavier initialization
                scale = math.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
                weight_matrix = [[random.uniform(-scale, scale) for _ in range(layer_sizes[i])] 
                               for _ in range(layer_sizes[i + 1])]
                self.weights.append(weight_matrix)
                
                # Initialize biases to zero
                bias_vector = [0.0 for _ in range(layer_sizes[i + 1])]
                self.biases.append(bias_vector)

    def sigmoid(self, x):
        """Sigmoid activation function with clipping to prevent overflow"""
        try:
            if x < -709:  # Approximately ln(double.MinValue)
                return 0
            if x > 709:   # Approximately ln(double.MaxValue)
                return 1
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            if x < 0:
                return 0
            return 1

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        sx = self.sigmoid(x)
        return sx * (1 - sx)

    def forward_propagate(self, inputs):
        """
        Forward propagate the input through the network
        Returns both the final output and all intermediate activations
        """
        activations = [inputs]
        weighted_sums = []

        # Process each layer
        current_activation = inputs
        for i in range(len(self.weights)):
            weighted_sum = [0] * len(self.weights[i])
            
            # Calculate weighted sum for each neuron in current layer
            for j in range(len(self.weights[i])):
                sum = self.biases[i][j]
                for k in range(len(current_activation)):
                    sum += self.weights[i][j][k] * current_activation[k]
                weighted_sum[j] = sum
            
            weighted_sums.append(weighted_sum)
            
            # Apply activation function
            current_activation = [self.sigmoid(x) for x in weighted_sum]
            activations.append(current_activation)

        return activations, weighted_sums

    def backward_propagate(self, inputs, target, learning_rate):
        """
        Backward propagate the error and update weights
        """
        activations, weighted_sums = self.forward_propagate(inputs)
        
        # Calculate output layer error (using cross-entropy derivative for binary classification)
        output_errors = []
        output_layer = activations[-1]
        for i in range(len(output_layer)):
            # Avoid division by zero
            output = max(min(output_layer[i], 0.9999), 0.0001)
            error = -(target[i] / output - (1 - target[i]) / (1 - output))
            output_errors.append(error)
        
        # Backpropagate the error
        layer_errors = [output_errors]
        for i in range(len(self.weights) - 1, 0, -1):
            prev_errors = layer_errors[0]
            current_errors = [0] * len(self.weights[i-1])
            
            # Calculate error for each neuron in current layer
            for j in range(len(self.weights[i-1])):
                error = 0
                for k in range(len(prev_errors)):
                    error += prev_errors[k] * self.weights[i][k][j] * self.sigmoid_derivative(weighted_sums[i-1][j])
                current_errors[j] = error
            
            layer_errors.insert(0, current_errors)
        
        # Update weights and biases with momentum
        momentum = 0.9
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    error = layer_errors[i][j]
                    activation = activations[i][k]
                    delta = error * activation
                    
                    # Add L2 regularization
                    reg_term = 0.0001 * self.weights[i][j][k]
                    
                    # Update with momentum and regularization
                    self.weights[i][j][k] -= learning_rate * (delta + reg_term)
                
                # Update biases
                self.biases[i][j] -= learning_rate * layer_errors[i][j]

    def train(self, training_data, targets, learning_rate=0.1, epochs=1000):
        """
        Train the neural network using the provided training data
        training_data: List of input arrays
        targets: List of target arrays
        """
        print("\nStarting training...")
        print("Progress: [", end="", flush=True)
        progress_step = max(1, epochs // 50)  # Show 50 progress marks
        
        best_error = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            total_error = 0
            
            # Shuffle training data
            combined = list(zip(training_data, targets))
            random.shuffle(combined)
            training_data, targets = zip(*combined)
            
            for inputs, target in zip(training_data, targets):
                # Forward propagate
                activations, _ = self.forward_propagate(inputs)
                output = activations[-1]
                
                # Calculate cross-entropy error
                for i in range(len(output)):
                    output_i = max(min(output[i], 0.9999), 0.0001)  # Clip to avoid log(0)
                    total_error -= target[i] * math.log(output_i) + (1 - target[i]) * math.log(1 - output_i)
                
                # Backward propagate
                self.backward_propagate(inputs, target, learning_rate)
            
            # Early stopping check
            if total_error < best_error:
                best_error = total_error
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement > 50:  # Stop if no improvement for 50 epochs
                    print("\nStopping early - no improvement in error")
                    break
            
            # Show progress bar
            if epoch % progress_step == 0:
                print("=", end="", flush=True)
            
            if epoch % 100 == 0:
                print(f"\nEpoch {epoch}, Error: {total_error:.4f}")
        
        print("] Done!")

    def predict(self, inputs):
        """Make a prediction for the given inputs"""
        activations, _ = self.forward_propagate(inputs)
        return activations[-1]

    def save_model(self, filename):
        """Save the model weights and biases to a file"""
        model_data = {
            'layer_sizes': self.layer_sizes,
            'weights': self.weights,
            'biases': self.biases
        }
        with open(filename, 'w') as f:
            json.dump(model_data, f)

    @classmethod
    def load_model(cls, filename):
        """Load a model from a file"""
        with open(filename, 'r') as f:
            model_data = json.load(f)
        
        # Create a new network with the same architecture
        network = cls(model_data['layer_sizes'])
        network.weights = model_data['weights']
        network.biases = model_data['biases']
        return network
