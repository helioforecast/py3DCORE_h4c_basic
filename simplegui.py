import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Create a figure and a canvas
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        # Create a horizontal slider and a label
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        self.sliderLabel = QLabel(str(self.slider.value()))

        # Create five vertical sliders and labels
        self.sliders = []
        self.sliderLabels = []
        for i in range(5):
            slider = QSlider(Qt.Vertical)
            slider.setRange(0, 100)
            slider.setValue(50)
            slider.setTickPosition(QSlider.TicksBothSides)
            slider.setTickInterval(10)
            label = QLabel(str(slider.value()))
            self.sliders.append(slider)
            self.sliderLabels.append(label)

        # Connect sliders to update functions
        self.slider.valueChanged.connect(self.updateSliderLabel)
        for slider in self.sliders:
            slider.valueChanged.connect(self.updatePlot)

        # Create a layout for the sliders and labels
        sliderLayout = QGridLayout()
        sliderLayout.addWidget(self.sliderLabel, 0, 0, 1, 5)
        for i, (slider, label) in enumerate(zip(self.sliders, self.sliderLabels)):
            sliderLayout.addWidget(label, 1, i)
            sliderLayout.addWidget(slider, 2, i)

        # Create a layout for the canvas and sliders
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.slider)
        layout.addLayout(sliderLayout)

        # Set the main layout of the window
        self.setLayout(layout)

    def updateSliderLabel(self, value):
        self.sliderLabel.setText(str(value))
        self.updatePlot()

    def updatePlot(self):
        # Update the plot based on the slider values
        x = range(100)
        y = [self.slider.value() * (i/100.0) ** 2 + sum(slider.value() for slider in self.sliders) for i in x]
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x, y)
        self.canvas.draw()

if __name__ == '__main__':
    # Create the application and window
    app = QApplication(sys.argv)
    window = MyWindow()
    window.setWindowTitle('My Application')
    window.show()

    # Run the event loop
    sys.exit(app.exec_())