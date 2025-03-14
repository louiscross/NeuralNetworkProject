# Neural Network Project - Hotdog Classifier

A pure Python implementation of a neural network from scratch, featuring a hotdog image classifier. This project demonstrates the fundamentals of neural networks without using any machine learning libraries.

## Features

- Pure Python neural network implementation
- Image classification (Hotdog vs Not Hotdog)
- Interactive command-line interface
- Model saving and loading
- Training data management
- Xavier initialization
- Cross-entropy loss
- Early stopping

## Project Structure

- `neural_network.py`: Core neural network implementation
- `hotdog_classifier.py`: Hotdog classification application
- `image_processor.py`: Image processing utilities
- `xor_example.py`: XOR problem demonstration
- `pattern_example.py`: Pattern recognition example

## Requirements

- Python 3.6+
- Pillow (PIL) for image processing

## Setup

1. Clone the repository:
```bash
git clone https://github.com/louiscross/NeuralNetworkProject.git
cd NeuralNetworkProject
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create required folders:
```bash
mkdir -p training_data/hotdog training_data/not_hotdog test_images
```

## Usage

1. Add training images:
   - Place hotdog images in `training_data/hotdog/`
   - Place non-hotdog images in `training_data/not_hotdog/`
   - Place new test images in `test_images/`

2. Run the classifier:
```bash
python hotdog_classifier.py
```

3. Follow the interactive menu to:
   - List test images
   - Test an image
   - Retrain the network
   - Exit the program

## How It Works

The neural network uses:
- 24x24 RGB image input (1,728 input neurons)
- Single hidden layer with 32 neurons
- Sigmoid activation function
- Cross-entropy loss for binary classification
- Xavier initialization for better convergence
- Early stopping to prevent overfitting

## Contributing

Feel free to open issues or submit pull requests for improvements!
