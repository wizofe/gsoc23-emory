from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QComboBox

from trainer import train_and_evaluate
from model import Net, ConvNet

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.combo = QComboBox()
        self.combo.addItem("Net")
        self.combo.addItem("ConvNet")
        self.layout.addWidget(self.combo)

        self.button = QPushButton('Train')
        self.button.clicked.connect(self.on_button_click)
        self.layout.addWidget(self.button)

    def on_button_click(self):
        model_name = self.combo.currentText()

        if model_name == "Net":
            model = Net(784, 500, 10)
        elif model_name == "ConvNet":
            model = ConvNet(10)

        train_and_evaluate(model)

if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()

    app.exec_()

