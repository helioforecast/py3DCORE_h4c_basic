import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QCheckBox, QLabel, QComboBox
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # set up the main window layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # create the sidebar layout
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setAlignment(Qt.AlignTop)

        # create some checkboxes for the sidebar
        self.checkboxes = []
        for i in range(3):
            checkbox = QCheckBox("Checkbox {}".format(i+1))
            checkbox.stateChanged.connect(self.update_canvas)
            self.checkboxes.append(checkbox)
            sidebar_layout.addWidget(checkbox)

        # add a combobox for setting the font size
        font_size_label = QLabel("Label Font Size:")
        self.font_size_combobox = QComboBox()
        for i in range(6, 16):
            self.font_size_combobox.addItem(str(i))
        self.font_size_combobox.currentIndexChanged.connect(self.update_canvas)
        sidebar_layout.addWidget(font_size_label)
        sidebar_layout.addWidget(self.font_size_combobox)

        # add the sidebar to the main layout
        main_layout.addLayout(sidebar_layout)

        # create the main canvas
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        # create the initial subfigure to display
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.plot([1, 2, 3], [1, 2, 3])
        self.ax1.set_xlabel("X Label", fontsize=12)
        self.ax1.set_ylabel("Y Label", fontsize=12)

        # set the main window properties
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Matplotlib Subfigures')

        # show the main window
        self.show()

    def update_canvas(self):
        # clear the existing subfigures
        self.fig.clf()

        # get the selected font size from the combobox
        font_size = int(self.font_size_combobox.currentText())

        # create a new subfigure for each checkbox that is checked
        subplots = []
        for i, checkbox in enumerate(self.checkboxes):
            if checkbox.isChecked():
                subplots.append(self.fig.add_subplot(len(self.checkboxes), 1, i+1))
                subplots[-1].plot([1, 2, 3], [1, 2, 3])
                subplots[-1].set_xlabel("X Label", fontsize=font_size)
                subplots[-1].set_ylabel("Y Label", fontsize=font_size)

        # update the canvas size based on the number of subfigures
        num_subplots = len(subplots)
        canvas_width = max(num_subplots * 100, 500)
        canvas_height = 400
        self.canvas.setMinimumSize(canvas_width, canvas_height)

        # redraw the canvas
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())