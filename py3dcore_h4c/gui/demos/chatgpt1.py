import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        # Set window properties
        self.setWindowTitle('My Modern GUI')
        self.setGeometry(100, 100, 800, 600)
        
        # Create main widget and layout
        self.central_widget = QFrame(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Create left widget and layout
        self.left_widget = QFrame(self.central_widget)
        self.left_widget.setFixedWidth(100)
        self.left_layout = QVBoxLayout(self.left_widget)
        
        # Add label to left layout
        self.label = QLabel('Labels', self.left_widget)
        self.left_layout.addWidget(self.label)
        
        # Add sliders to left layout
        self.slider1 = QFrame(self.left_widget)
        self.slider1.setFixedHeight(150)
        self.slider1_layout = QVBoxLayout(self.slider1)
        self.slider1_layout.addWidget(QLabel('Slider 1', self.slider1))
        self.slider1_layout.addWidget(QLabel('0', self.slider1))
        self.slider1_layout.addWidget(QLabel('50', self.slider1))
        self.slider1_layout.addWidget(QLabel('100', self.slider1))
        self.left_layout.addWidget(self.slider1)
        
        self.slider2 = QFrame(self.left_widget)
        self.slider2.setFixedHeight(150)
        self.slider2_layout = QVBoxLayout(self.slider2)
        self.slider2_layout.addWidget(QLabel('Slider 2', self.slider2))
        self.slider2_layout.addWidget(QLabel('0', self.slider2))
        self.slider2_layout.addWidget(QLabel('50', self.slider2))
        self.slider2_layout.addWidget(QLabel('100', self.slider2))
        self.left_layout.addWidget(self.slider2)
        
        # Add left widget to main layout
        self.main_layout.addWidget(self.left_widget)
        
        # Add image to main layout
        self.image_label = QLabel(self.central_widget)
        self.image_label.setPixmap(QPixmap('my_image.png'))
        self.image_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.image_label)
        
        # Show the window
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())