# visualizer.py
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout, QWidget

class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axis = self.figure.add_subplot(111)

        self.layoutVertical = QVBoxLayout(self)        
        self.layoutVertical.addWidget(self.canvas)

    def plot(self, x_data, y_data, title=""):
        self.axis.clear()
        self.axis.plot(x_data, y_data)
        self.axis.set_title(title)
        self.canvas.draw()

