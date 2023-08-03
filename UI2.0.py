import sys
import PyQt6.uic
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QMainWindow, QGridLayout, QLabel, \
    QTextEdit, QVBoxLayout, QHBoxLayout, QListView, QStyle, QSlider, QStatusBar
from PyQt6.QtCore import pyqtSlot, QThread, pyqtSignal, QObject, QCoreApplication, Qt, QStringListModel, QSize, QUrl
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QTreeView, QVBoxLayout, QWidget
#from Extract_PPG import Find_PPG
#from Extract_Vessel import Vessel_Extract_Video
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
#from magnify import magnify
from datetime import datetime

class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass

class Worker(QThread):
    def __init__(self):
        super().__init__()
        self.fileName = None
        self.action = None
        self.ppg = None
    def run(self):
        if self.action == 'find_ppg':
            self.ppg=Find_PPG(self.fileName)

        elif self.action == 'extract':
            Vessel_Extract_Video(self.fileName, 'vessel.mp4')
        elif self.action == 'magnify':
            magnify(self.fileName)


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        self.title = 'Vessel Extraction App'
        self.left = 10
        self.top = 10
        self.width = 800
        self.height = 600
        self.selectedFiles = QStringListModel(self)
        self.selectedFiles.dataChanged.connect(self.updateFileListView)
        self.initUI()
        self.setStyleSheet("""
            QPushButton {
                background-color: #1e90ff;
                color: white;
                border-radius: 10px;
                padding: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #87cefa;
            }
            QTextEdit {
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        self.worker = Worker()


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create a main widget and layout for the three columns
        main_widget = QWidget(self)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # Column 1: App Name, Choose File, and File List View
        column1_layout = QVBoxLayout()
        main_layout.addLayout(column1_layout, 20)

        app_name = QLabel("Vessel Extraction and Magnification App", self)

        # Create a QListView to display selected files.
        self.fileListView = QListView()
        self.fileListView.setModel(self.selectedFiles)
        self.fileListView.clicked.connect(self.onFileClicked)

        # Create a 'Choose File' button
        choose_file_btn = QPushButton('Choose File', self)
        choose_file_btn.clicked.connect(self.openFileNameDialog)

        column1_layout.addWidget(app_name, 10)
        column1_layout.addWidget(self.fileListView, 80)
        column1_layout.addWidget(choose_file_btn, 10)
        #TODO add Video disaplay here
        #Column 2: PPG Chart and Video/Frame display
        column2_layout = QVBoxLayout()
        main_layout.addLayout(column2_layout)
        chart_label = QLabel("Chart Placeholder", self)  # Replace with actual chart widget

        column2_layout.addWidget(chart_label, 35)
        # Video Player
        self.mediaPlayer = QMediaPlayer()
        btnSize = QSize(16, 16)
        videoWidget = QVideoWidget()

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setFixedHeight(24)
        self.playButton.setIconSize(btnSize)
        self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Orientation.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.statusBar = QStatusBar()
        self.statusBar.setFixedHeight(14)

        column2_layout.addWidget(self.playButton)
        column2_layout.addWidget(self.positionSlider)

        column2_layout.addWidget(videoWidget,55)
        column2_layout.addWidget(self.statusBar)


        # help(self.mediaPlayer)
        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.playbackStateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.errorChanged.connect(self.handleError)
        self.statusBar.showMessage("Ready")
        # Column 3: Date Time PPG HeartRate, Extract and Magnify Buttons, and Processing message panel
        column3_layout = QVBoxLayout()
        main_layout.addLayout(column3_layout)

        # Date Time PPG HeartRate Placeholder
        row1_layout = QHBoxLayout()
        row2_layout = QHBoxLayout()
        row3_layout = QHBoxLayout()
        # Add QLabel widgets for Date, Time, PPG, and Heart Rate
        self.date_label = QLabel("Date:", self)
        self.time_label = QLabel("Time:", self)
        self.ppg_label = QLabel("PPG:", self)
        self.heart_rate_label = QLabel("Heart Rate:", self)

        # Add placeholders for the detected values
        self.detected_date = QLabel(" ", self)
        self.detected_time = QLabel(" ", self)
        self.detected_ppg_fre = QLabel(" ", self)
        self.detected_heart_rate = QLabel(" ", self)

        row1_layout.addWidget(self.date_label)
        row1_layout.addWidget(self.detected_date)

        row2_layout.addWidget(self.time_label)
        row2_layout.addWidget(self.detected_time)

        row3_layout.addWidget(self.ppg_label)
        row3_layout.addWidget(self.detected_ppg_fre)

        row3_layout.addWidget(self.heart_rate_label)
        row3_layout.addWidget(self.detected_heart_rate)

        column3_layout.addLayout(row1_layout, 10)
        column3_layout.addLayout(row2_layout, 10)
        column3_layout.addLayout(row3_layout, 10)


        extract_btn = QPushButton('Extract Vessel', self)
        magnify_btn = QPushButton('Magnify Vessel', self)
        find_ppg_btn = QPushButton('Find PPG', self)
        self.btn_extract = extract_btn
        self.btn_magnify = magnify_btn
        self.btn_find_ppg = find_ppg_btn
        extract_btn.setToolTip('Click to extract vessels in video')
        magnify_btn.setToolTip('Click to magnify vessels in video')
        find_ppg_btn.setToolTip('Click to find PPG in video')
        extract_btn.clicked.connect(self.extract_vessel)
        magnify_btn.clicked.connect(self.magnify_vessel)
        find_ppg_btn.clicked.connect(self.find_ppg)
        # column3_layout.addWidget(data_label, 60)
        column3_layout.addWidget(extract_btn, 10)
        column3_layout.addWidget(magnify_btn, 10)
        column3_layout.addWidget(find_ppg_btn, 10)



        # Add a QTextEdit widget to show the processing output
        self.processing_output = QTextEdit(self)
        column3_layout.addWidget(self.processing_output, 20)
        self.processing_output.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Set the stretch factor of each column in the main layout
        main_layout.setStretch(0, 20)
        main_layout.setStretch(1, 50)
        main_layout.setStretch(2, 30)

        self.setCentralWidget(main_widget)
    def onDirectoryClicked(self, index):
        # This method will be triggered when the user clicks on an item in the QTreeView
        file_info = self.fileModel.fileInfo(index)
        file_path = file_info.absoluteFilePath()
        if file_info.isFile():  # Check if the clicked item is a file
            self.selectedFiles.setStringList(self.selectedFiles.stringList() + [file_path])
            self.fileListView.setModel(self.selectedFiles)
            print(f'You selected file: {file_path}')

    def openFileNameDialog(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi)")
        if fileName:
            print(f'You select file successfully: {fileName}')
            self.fileName = fileName
            self.selectedFiles.setStringList(self.selectedFiles.stringList() + [fileName])
            self.btn_extract.setEnabled(True)
            self.btn_magnify.setEnabled(True)
            self.mediaPlayer.setSource(QUrl.fromLocalFile(fileName))
            self.playButton.setEnabled(True)
            self.statusBar.showMessage("Playing")
            self.play()

    @pyqtSlot()
    def updateFileListView(self):
        self.fileListView.setModel(self.selectedFiles)
    def onFileClicked(self, index):
        # This method will be triggered when the user clicks on an item in the QListView
        fileName = self.selectedFiles.data(index)
        print(f'You selected file: {fileName}')
    def extract_vessel(self):
        if self.fileName:
            self.do_something_with_file(self.fileName, "extract")
    def find_ppg(self):
        if self.fileName:
            self.do_something_with_file(self.fileName, "find_ppg")
    def magnify_vessel(self):
        if self.fileName:
            self.do_something_with_file(self.fileName, "magnify")

    def do_something_with_file(self, fileName, action):
        if self.worker.isRunning():
            print("Please wait for the current process to finish.")
        else:
            self.worker.fileName = self.fileName
            self.worker.action = action
            self.worker.start()
            current_datetime = datetime.now()
            current_date = current_datetime.date()
            current_time = current_datetime.time()
            date_str = current_date.strftime("%Y-%m-%d")
            time_str = current_time.strftime("%H:%M:%S")
            self.update_detected_values(date_str, time_str,"2","2")
    def update_detected_values(self, date, time, ppg, heart_rate):
        # Update the detected values for Date, Time, PPG, and Heart Rate
        self.detected_date.setText(date)
        self.detected_time.setText(time)
        self.detected_ppg_fre.setText(ppg)
        self.detected_heart_rate.setText(heart_rate)
    def abrir(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Media",
                ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")

        if fileName != '':
            self.mediaPlayer.setSource(QUrl.fromLocalFile(fileName))
            self.playButton.setEnabled(True)
            self.statusBar.showMessage(fileName)
            self.play()

    def play(self):
        if self.mediaPlayer.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.statusBar.showMessage("Error: " + self.mediaPlayer.errorString())
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
    ex.showMaximized()
    sys.exit(app.exec())