# PyTorch GUI with PyQt for Deep Learning - Extended User Documentation

## Introduction
This project, part of Google Summer of Code (GSoC) 2023, aims to democratize the field of machine learning, particularly deep learning, by providing an intuitive and user-friendly Graphical User Interface (GUI). Developed in collaboration with Emory University, this application allows users to interact with various neural network architectures without delving into the complexities of code. The project leverages the power of PyTorch for machine learning and PyQt for creating the GUI.

### About GSoC 2023
Google Summer of Code is an annual program offering stipends to university students working on open-source projects during the summer. This particular project aims to simplify deep learning programming by providing a GUI alternative to traditional coding, lowering the entry barrier for beginners and researchers alike.

### Project at Emory University
This project has been developed in partnership with Emory University to advance research in the field of machine learning. With its user-friendly interface, the application is designed to be a tool for students and researchers to quickly prototype and test different deep learning models on the MNIST dataset.

---

## Features
- **Model Selection**: Choose from pre-defined models like Simple CNN, Deeper CNN, and Simple MLP, or add your custom models.
- **Hyperparameter Tuning**: Fine-tune learning rate and momentum without changing any code.
- **Real-time Updates**: Get instant feedback through a progress bar, loss label, and real-time loss plot.
- **Batch Operations**: Utilize the multi-instance feature to train multiple models concurrently, aiding in hyperparameter optimization and model selection.
  
---

## Dependencies
- Python 3.7+
- PyTorch
- PyQt5

### Installation
To install the necessary dependencies, run:
```bash
pip install torch torchvision PyQt5
```

---

## How to Use

### Basic Usage
1. **Model Selection**: Use the drop-down menu to choose among predefined or custom models.
2. **Set Learning Rate and Momentum**: Use the respective spin boxes.
3. **Train**: Click 'Train' to start the training. You'll see real-time loss and a progress bar.
4. **Stop**: Use 'Stop' to halt ongoing training.

### Multi-instance Training
You can open multiple instances of the application using `MultiAppDemo`. Each instance works independently, allowing you to experiment with different models or hyperparameters concurrently. This is particularly useful when you want to perform hyperparameter tuning or model comparisons in a shorter period.

---

## Input Data Shape and Custom Models

### Input Data
The code is pre-configured for MNIST, which has an input shape of 28x28 grayscale images. When using custom datasets, you'll need to reshape your data accordingly. In PyTorch, data usually has the shape \[batch_size, num_channels, height, width\].

### Adding Custom Models
To add your own models, you'll need to define a new PyTorch `nn.Module` class with the layers and forward pass logic. Once the custom model is defined, you can add an option for it in the `modelComboBox` widget.

```python
self.modelComboBox.addItem('Your Custom Model')
```

And add a corresponding condition in the `trainModel` function:

```python
elif model_name == 'Your Custom Model':
    model = YourCustomModel()
```

### Input File Configuration
If you are using custom datasets, ensure that they are in a format compatible with PyTorch's DataLoader. Your custom class should also reflect these changes.

---

## Advanced Features

### Plot Customization
The GUI includes a plotting window that shows the training loss in real-time. Advanced users can modify the plot by altering the `update_plot` method in the Python code. Custom metrics, multiple plots, and other plot customization can be done here.

### Thread Management
The project uses Python's `QThread` for handling the training process. If you wish to add more complex operations like model evaluation or multi-threading, you would need to modify the `TrainingThread` class.

### Early Stopping and Checkpointing
These features are not implemented in the current version but can be added by extending the `TrainingThread` class. 

### Exporting Trained Models
The current implementation doesn't include a feature to save trained models. However, you can easily integrate PyTorch's model-saving APIs into the code.

---

## Conclusion
This application, developed under GSoC 2023 and in partnership with Emory University, aims to make deep learning more accessible. Its features range from basic model training to advanced features like real-time loss plotting and multi-instance training. Whether you are a student, a researcher, or an enthusiast, this tool aims to simplify your journey in the world of deep learning.

For further questions or issues, feel free to open a GitHub issue or reach out to the maintainers.