from PyQt5.QtWidgets import *
from PyQt5 import uic

class guiGA(QMainWindow):

    def __init__(self):
        super(guiGA, self).__init__()
        uic.loadUi("gui_GA.ui", self)
        self.show()

def run_gui():
    app = QApplication([])
    window = guiGA()
    app.exec_()

run_gui()