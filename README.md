# Geometric Shape Recognition with a From-Scratch Single-Layer Perceptron

This project implements a single-layer perceptron neural network without using any machine learning libraries (e.g., no TensorFlow, PyTorch, scikit-learn). The model is trained from scratch to recognize geometric shapes (initially circles) from compressed black-and-white images. The entire implementation relies solely on NumPy for numerical operations and PIL for image processing.

## Challenge
The primary challenge was to build a functional neural network entirely from scratch, including:
- Manual implementation of gradient descent
- Hand-coded gradient computation using finite differences
- Custom weight initialization and update rules
- Image preprocessing and compression without external ML tools

## Project Files Description

### Core Implementation Files:
- **compressionImages.py** - Handles image preprocessing and compression. Converts 224x224 RGB images to 56x56 black-and-white images by averaging 4x4 blocks and applying a threshold of 0.9

- **data.py** - Contains the Data class that:
  - Loads and processes geometric shape images
  - Converts images to binary matrices (0 for white, 1 for black)
  - Flattens 56x56 matrices into 3136-dimensional input vectors
  - Provides access methods for image data and labels

- **neurone.py** - Implements the core Neurone class with:
  - Random weight initialization using Gaussian distribution
  - Forward pass computation (dot product + sigmoid activation)
  - Manual gradient calculation using finite differences
  - Weight update via custom gradient descent implementation
  - Sigmoid activation function implementation

- **main.py** - Main execution script that:
  - Manages the training process with 20 epochs
  - Handles batch processing of 600 images per epoch
  - Tests the trained model on validation images
  - Outputs classification results for circle recognition

### Data Files:
- **dataPoidsSynaptiques.txt** - Contains pre-trained synaptic weights in a serialized list format, allowing the model to be used without retraining

### Dataset:
- The project uses a custom dataset of 8 geometric shapes: circle, kite, parallelogram, rectangle, rhombus, square, trapezoid, and triangle
- Images are organized in datasetC/train/ directory with subfolders for each shape

## How It Works

1. Image Preprocessing:
   - Images are converted to black and white
   - Each image is compressed from 224x224 to 56x56 pixels by averaging 4x4 blocks with a threshold of 0.9

2. Neuron Initialization:
   - Weights are initialized randomly using random.gauss(0, 1)

3. Training:
   - Forward pass: Computes output via dot product and sigmoid activation
   - Loss: Binary error (output vs. target)
   - Gradient: Computed manually using finite differences
   - Weight update: Custom gradient descent with fixed step size

4. Testing:
   - The trained neuron classifies test images as circles or non-circles based on a 0.5 threshold

## Requirements
- Python 3.x
- Pillow (PIL)
- NumPy

## No ML Libraries Used
This project intentionally avoids:
- TensorFlow
- PyTorch
- Keras
- scikit-learn
- Any other high-level ML framework

## Results
The model was tested on sample images and successfully classified circles with reasonable accuracy.

## Future Improvements
- Extend to multiple shapes (multi-class classification)
- Implement a multi-layer network manually
- Add momentum or adaptive learning rates to gradient descent

Note: This project is educational and demonstrates neural network fundamentals without relying on high-level abstractions.
