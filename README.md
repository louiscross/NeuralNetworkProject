# Not Hotdog - A Neural Network from Scratch 

Welcome to my version of the world's most sophisticated hotdog detection system. Inspired by Jian-Yang's groundbreaking "Not Hotdog" app from Silicon Valley, I decided to build my own neural network from scratch to tackle this crucial problem of our time.

## What is it?

This project is a pure Python implementation of a neural network that can classify images as either "Hotdog" or "Not Hotdog". No fancy machine learning libraries - just maths and alot of backpropagation.

## Features

- **Pure Python Neural Network**: Built from the ground up, just like Jian-Yang would have wanted
- **Image Classification**: Hotdog vs Not Hotdog
- **Model Persistence**: Save your trained models for future hotdog detection needs
- **Advanced Features**:
  - Xavier initialization for better training
  - Cross-entropy loss for accurate predictions
  - Early stopping to prevent overfitting
  - Custom image processing pipeline

## 🧠 How It Works

The neural network is built with the following components:

1. **Input Layer**: Processes images into a format our network can understand
2. **Hidden Layers**: Where the magic happens (and by magic, we mean matrix multiplication)
3. **Output Layer**: The final verdict - Hotdog or Not Hotdog
4. **Backpropagation**: How our network learns from its mistakes

## Technical Requirements

- Python 3.6+
- Pillow (PIL) for image processing
- A sense of humor (optional but recommended)

## Getting Started

1. Clone this repository:
```bash
git clone https://github.com/yourusername/not-hotdog.git
cd not-hotdog
```

2. Install the requirements:
```bash
pip install -r requirements.txt
```

3. Run the classifier:
```bash
python hotdog_classifier.py
```

## Training Your Own Model

Want to train your own hotdog detection model? The process is simple:

1. Collect your training data (hotdogs and not-hotdogs)
2. Place them in the appropriate directories
3. Run the training script
4. Watch as your model learns to distinguish between hotdogs and everything else

## Contributing

Feel free to contribute! at the moment the model is not fine tuned, and lacks alot of depth, it still has alot of room for improvement

## License

This project is licensed under the MIT License - see the LICENSE file for details.
