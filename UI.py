import sys

from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QMainWindow, QGridLayout, QLabel, QTextEdit
from PyQt6.QtCore import pyqtSlot, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QTextCursor

from magnify import magnify
class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        # Setup the EmittingStream object and redirect the stdout to this stream
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)

        self.title = 'Vessel Extraction and Magnification App'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480

        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        widget = QWidget()
        layout = QGridLayout()
        widget.setLayout(layout)

        description = QLabel(
            "Welcome to the Vessel Extraction and Magnification App. \n"
            "1. Click 'Choose File' to select a video file. \n"
            "2. After file uploaded, click 'Extract Vessel' to extract the vessels from the video. \n"
            "3. Click 'Magnify Vessel' to magnify the vessels in the video."
        )
        layout.addWidget(description, 0, 0, 1, 2)

        self.button = QPushButton('Choose File', self)
        self.button.setToolTip('Click to select file')
        self.button.clicked.connect(self.openFileNameDialog)
        layout.addWidget(self.button, 1, 0, 1, 2)

        self.btn_extract = QPushButton('Extract Vessel', self)
        self.btn_extract.setToolTip('Click to extract vessels from selected video')
        self.btn_extract.clicked.connect(self.extract_vessel)
        self.btn_extract.setEnabled(False)
        layout.addWidget(self.btn_extract, 2, 0)

        self.btn_magnify = QPushButton('Magnify Vessel', self)
        self.btn_magnify.setToolTip('Click to magnify vessels in selected video')
        self.btn_magnify.clicked.connect(self.magnify_vessel)
        self.btn_magnify.setEnabled(False)
        layout.addWidget(self.btn_magnify, 2, 1)

        # Add a QTextEdit widget to show the processing output
        self.processing_output = QTextEdit()
        layout.addWidget(self.processing_output, 3, 0, 1, 2)

        self.setCentralWidget(widget)

    def openFileNameDialog(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi)")
        if fileName:
            print(f'You select file successfully: {fileName}')
            self.fileName = fileName
            self.btn_extract.setEnabled(True)
            self.btn_magnify.setEnabled(True)

    def extract_vessel(self):
        if self.fileName:
            self.do_something_with_file(self.fileName, "extract")

    def magnify_vessel(self):
        if self.fileName:
            self.do_something_with_file(self.fileName, "magnify")

    def do_something_with_file(self, fileName, action):
        if action== "extract":
            print("Extracting vessel...")
            #extract_vessel(fileName)
        elif action == "magnify":
            print("Magnifying vessel...")
            magnify(fileName)

    @pyqtSlot(str)
    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        cursor = self.processing_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.processing_output.setTextCursor(cursor)
        self.processing_output.ensureCursorVisible()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec())
