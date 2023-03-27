import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSlider, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QGridLayout, QWidget
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        # Set window properties
        self.setWindowTitle('My Complex GUI')
        self.setGeometry(100, 100, 800, 600)
        
        # Create main widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Create left layout for sliders and buttons
        self.left_layout = QVBoxLayout()
        self.left_layout.setSpacing(10)
        self.main_layout.addLayout(self.left_layout)
        
        # Create sliders
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setRange(0, 100)
        self.slider1.setValue(50)
        self.slider1.setTickPosition(QSlider.TicksBelow)
        self.slider1.setTickInterval(10)
        self.left_layout.addWidget(self.slider1)
        
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setRange(0, 100)
        self.slider2.setValue(50)
        self.slider2.setTickPosition(QSlider.TicksBelow)
        self.slider2.setTickInterval(10)
        self.left_layout.addWidget(self.slider2)
        
        # Create button group box
        self.button_box = QGroupBox('Buttons', self)
        self.button_box_layout = QVBoxLayout(self.button_box)
        self.button_box_layout.setSpacing(10)
        self.left_layout.addWidget(self.button_box)
        
        # Create buttons
        self.button1 = QPushButton('Button 1')
        self.button1.clicked.connect(self.on_button1_clicked)
        self.button_box_layout.addWidget(self.button1)
        
        self.button2 = QPushButton('Button 2')
        self.button2.clicked.connect(self.on_button2_clicked)
        self.button_box_layout.addWidget(self.button2)
        
        self.button3 = QPushButton('Button 3')
        self.button3.clicked.connect(self.on_button3_clicked)
        self.button_box_layout.addWidget(self.button3)
        
        # Create right layout for image and label
        self.right_layout = QVBoxLayout()
        self.right_layout.setSpacing(10)
        self.main_layout.addLayout(self.right_layout)
        
        # Add image to right layout
        self.image_label = QLabel(self)
        self.image_label.setPixmap('my_image.png')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.right_layout.addWidget(self.image_label)
        
        # Add label to right layout
        self.label = QLabel('Label Text', self)
        self.label.setAlignment(Qt.AlignCenter)
        self.right_layout.addWidget(self.label)
        
        # Show the window
        self.show()

    def on_button1_clicked(self):
        self.label.setText('Button 1 was clicked')
        
    def on_button2_clicked(self):
        self.label.setText('Button 2 was clicked')
        
    def on_button3_clicked(self):
        self.label.setText('Button 3 was clicked')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())