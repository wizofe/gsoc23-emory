from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel, QLineEdit
from trainer import train_and_evaluate
from model import Net, ConvNet

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.model_label = QLabel('Select Model')
        self.layout.addWidget(self.model_label)

        self.model_combo = QComboBox()
        self.model_combo.addItem("Net")
        self.model_combo.addItem("ConvNet")
        self.layout.addWidget(self.model_combo)

        self.dataset_label = QLabel('Select Dataset')
        self.layout.addWidget(self.dataset_label)

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItem("MNIST")
        self.dataset_combo.addItem("TimeSeries")
        self.layout.addWidget(self.dataset_combo)

        self.learning_rate_label = QLabel('Learning Rate')
        self.layout.addWidget(self.learning_rate_label)

        self.learning_rate_edit = QLineEdit()
        self.layout.addWidget(self.learning_rate_edit)

        self.epoch_label = QLabel('Number of Epochs')
        self.layout.addWidget(self.epoch_label)

        self.epoch_edit = QLineEdit()
        self.layout.addWidget(self.epoch_edit)

        self.train_button = QPushButton('Train')
        self.train_button.clicked.connect(self.on_train_button_click)
        self.layout.addWidget(self.train_button)

    def on_train_button_click(self):
        model_name = self.model_combo.currentText()
        dataset_name = self.dataset_combo.currentText()
        learning_rate = float(self.learning_rate_edit.text())
        num_epochs = int(self.epoch_edit.text())

        if model_name == "Net":
            model = Net(784, 500, 10)
        elif model_name == "ConvNet":
            model = ConvNet(10)

        train_and_evaluate(model, dataset_name, learning_rate, num_epochs)

if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()

    app.exec_()

