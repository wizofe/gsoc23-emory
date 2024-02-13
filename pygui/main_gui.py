import sys
import random
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QPushButton, QComboBox, QDoubleSpinBox, QGraphicsView, QGraphicsScene, QProgressBar)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPen, QPolygonF
from PyQt5.QtCore import QPointF
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QHBoxLayout
import sys


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.fc1 = nn.Linear(26*26*32, 10)
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = x.view(-1, 26*26*32)
        x = self.fc1(x)
        return nn.functional.log_softmax(x, dim=1)


class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(24*24*64, 10)
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(-1, 24*24*64)
        x = self.fc1(x)
        return nn.functional.log_softmax(x, dim=1)


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)


class TrainingThread(QThread):
    update_plot = pyqtSignal(float)

    def __init__(self, model, lr, momentum):
        super().__init__()
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.is_running = True

  
    def run(self):
        transform = transforms.Compose([transforms.ToTensor()])
        train_data = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        for epoch in range(1):
            for i, (images, labels) in enumerate(train_loader):
                if not self.is_running:
                    break

                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.update_plot.emit(float(loss))
                
                # Simulate longer computations
                self.sleep(1)

    def stop(self):
        self.is_running = False

class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PyTorch MNIST Trainer')
        self.resize(1000, 600)

        layout = QVBoxLayout()

        self.label = QLabel('MNIST Data')
        layout.addWidget(self.label)

        self.modelComboBox = QComboBox()
        self.modelComboBox.addItem('Simple CNN')
        self.modelComboBox.addItem('Deeper CNN')
        self.modelComboBox.addItem('Simple MLP')
        layout.addWidget(self.modelComboBox)

        self.lrSpinBox = QDoubleSpinBox()
        self.lrSpinBox.setRange(0.001, 1)
        self.lrSpinBox.setSingleStep(0.001)
        self.lrSpinBox.setValue(0.01)
        layout.addWidget(self.lrSpinBox)

        self.momentumSpinBox = QDoubleSpinBox()
        self.momentumSpinBox.setRange(0, 1)
        self.momentumSpinBox.setSingleStep(0.1)
        self.momentumSpinBox.setValue(0.9)
        layout.addWidget(self.momentumSpinBox)

        self.trainButton = QPushButton('Train')
        self.trainButton.clicked.connect(self.trainModel)
        layout.addWidget(self.trainButton)
        
        self.progressBar = QProgressBar()
        layout.addWidget(self.progressBar)
        self.lossLabel = QLabel('Loss: N/A')
        layout.addWidget(self.lossLabel)

        self.stopButton = QPushButton('Stop')
        self.stopButton.clicked.connect(self.stopTraining)
        layout.addWidget(self.stopButton)

        self.thread = None  # Initialize thread to None

        # Visualization Window
        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        layout.addWidget(self.view)

        # Plot Timer for Real-time Plotting
        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self.update_plot)
        self.data = []
        # Initialize empty list to hold loss values for plotting
        self.loss_values = []
        
        self.setLayout(layout)
        self.show()

    def trainModel(self):
        model_name = self.modelComboBox.currentText()
        lr = self.lrSpinBox.value()
        momentum = self.momentumSpinBox.value()

        # Select Model
        if model_name == 'Simple CNN':
            model = SimpleCNN()
        elif model_name == 'Deeper CNN':
            model = DeeperCNN()
        elif model_name == 'Simple MLP':
            model = SimpleMLP()

        # Initialize Training Thread
        self.thread = TrainingThread(model, lr, momentum)
        self.thread.update_plot.connect(self.update_plot)
        self.thread.start()

    def update_plot(self, loss):
        # Update the progress bar and label
        self.progressBar.setValue(int(loss * 100))
        self.lossLabel.setText(f'Loss: {loss:.4f}')

        # Store the new loss value
        self.loss_values.append(loss)

        # Create a new QGraphicsScene
        scene = QGraphicsScene()
        pen = QPen(Qt.red)
        last_point = None

        # Draw lines between the points
        for i, loss_value in enumerate(self.loss_values):
            x = i * 10  # Multiply by 10 for better visibility
            y = loss_value * 100  # Multiply by 100 to scale the loss values
            current_point = QPointF(x, -y)  # Negative y because the QGraphicsView's y is inverted

            if last_point is not None:
                scene.addLine(last_point.x(), last_point.y(), current_point.x(), current_point.y(), pen)

            last_point = current_point

        self.view.setScene(scene)
   

    def stopTraining(self):
        if self.thread:
            self.thread.stop()

class MultiAppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Multiple PyTorch MNIST Trainers')
        self.resize(3000, 1000)  # Resize the main window to fit the three instances

        layout = QHBoxLayout()  # Main layout to hold 3 instances

        self.appDemo1 = AppDemo()
        self.appDemo2 = AppDemo()
        self.appDemo3 = AppDemo()

        layout.addWidget(self.appDemo1)
        layout.addWidget(self.appDemo2)
        layout.addWidget(self.appDemo3)

        self.setLayout(layout)
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = MultiAppDemo()
    sys.exit(app.exec_())

#app = QApplication(sys.argv)
#demo = AppDemo()
#sys.exit(app.exec_())
