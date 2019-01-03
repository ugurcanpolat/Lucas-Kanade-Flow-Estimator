#!/usr/local/bin/python3

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QMessageBox, QWidget, QGroupBox, \
    QAction, QFileDialog, QGridLayout, QLabel, qApp
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np
import cv2

class App(QMainWindow):
    def __init__(self):
        super(App, self).__init__()

        self.title = 'Lucas-Kanade optical flow estimation and Face recognition with eigenfaces'

        # Booelans to track if input images are loaded
        self.opticalFlowLoaded = False
        self.faceLoaded = False

        # Fix the size so boxes cannot expand
        self.setFixedSize(self.geometry().width(), self.geometry().height())

        self.initUI()

    def addImageToGroupBox(self, image, groupBox, labelString):
        # Get the height, width information
        height, width, channel = image.shape
        bytesPerLine = channel * width # 3-channel image

        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

        pix = QPixmap(qImg)

        # Add image  to the widget
        label = QLabel(labelString)
        label.setPixmap(pix)
        label.setAlignment(Qt.AlignCenter)
        groupBox.layout().addWidget(label)

    def deleteItemsFromWidget(self, layout):
        # Deletes items in the given layout
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    deleteItemsFromWidget(item.layout())

    def openOpticalFlowImage(self):
        # This function is called when the user clicks File->Open optical flow image.
        fName = QFileDialog.getOpenFileName(self, 'Open optical flow image', './', 'Image files (*.png *.jpg)')

        # File open dialog has been dismissed or file could not be found
        if fName[0] is '':
            return

        # If there is an input image loaded, remove it
        if self.opticalFlowLoaded:
            self.deleteItemsFromWidget(self.cornerGroupBox.layout())

        self.firstOpticalFlowImage = cv2.imread(fName[0]) # Read the image
        self.opticalFlowPath = fName
        self.opticalFlowLoaded = True

        self.addImageToGroupBox(self.firstOpticalFlowImage, self.opticalFlowGroupBox, 'Optical flow image')

    def openFaceImage(self):
        # This function is called when the user clicks File->Open face image.
        fName = QFileDialog.getOpenFileName(self, 'Open face image', './', 'Image files (*.png *.jpg *.tif)')

        # File open dialog has been dismissed or file could not be found
        if fName[0] is '':
            return

        # If there is an input image loaded, remove it
        if self.faceLoaded:
            self.deleteItemsFromWidget(self.faceGroupBox.layout())

        self.faceImage = cv2.imread(fName[0]) # Read the image
        self.faceLoaded = True

        self.addImageToGroupBox(self.faceImage, self.faceGroupBox, 'Face recognition image')

    def createEmptyOpticalFlowGroupBox(self):
        self.opticalFlowGroupBox = QGroupBox('Optical Flow Estimation')
        layout = QVBoxLayout()

        self.opticalFlowGroupBox.setLayout(layout)

    def createEmptyFaceRecognitionGroupBox(self):
        self.faceGroupBox = QGroupBox('Face Recognition using Eigenfaces')
        layout = QVBoxLayout()

        self.faceGroupBox.setLayout(layout)

    def initUI(self):
        # Add menu bar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('File')
        
        # Create action buttons of the menu bar
        opticalFlowAct = QAction('Open first taxi image', self)
        opticalFlowAct.triggered.connect(self.openOpticalFlowImage)

        faceAct = QAction('Open first face image', self) 
        faceAct.triggered.connect(self.openFaceImage)

        exitAct = QAction('Exit', self)        
        exitAct.triggered.connect(qApp.quit) # Quit the app

        # Add action buttons to the menu bar
        fileMenu.addAction(opticalFlowAct)
        fileMenu.addAction(faceAct)
        fileMenu.addAction(exitAct)

        # Create calculate optical flow button for toolbar
        calculateOpticalFlowAct = QAction('Calculate Optical Flow', self) 
        calculateOpticalFlowAct.triggered.connect(self.calculateOpticalFlowButtonClicked)

        # Create face recognition button for toolbar
        recognizeFaceAct = QAction('Recognize face', self) 
        recognizeFaceAct.triggered.connect(self.recoginizeFaceButtonClicked)
        
        # Create toolbar
        toolbar = self.addToolBar('Image Operations')
        toolbar.addAction(calculateOpticalFlowAct)
        toolbar.addAction(recognizeFaceAct)

        # Create empty group boxes 
        self.createEmptyOpticalFlowGroupBox()
        self.createEmptyFaceRecognitionGroupBox()

        # Since QMainWindows layout has already been set, create central widget
        # to manipulate layout of main window
        wid = QWidget(self)
        self.setCentralWidget(wid)

        # Initialize layout with groupboxes
        windowLayout = QGridLayout()
        windowLayout.addWidget(self.opticalFlowGroupBox, 0, 0)
        windowLayout.addWidget(self.faceGroupBox, 0, 1)
        wid.setLayout(windowLayout)

        self.setWindowTitle(self.title) 
        self.showMaximized()
        self.show()

    def calculateOpticalFlowButtonClicked(self):
        if not self.opticalFlowLoaded:
            # Error: "First load optical flow image" in MessageBox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Optical flow image is missing.")
            msg.setText('First load optical flow image!')
            msg.setStandardButtons(QMessageBox.Ok)

            msg.exec()
            return

        return NotImplemented

    def recoginizeFaceButtonClicked(self):
        if not self.faceLoaded:
            # Error: "First load face image" in MessageBox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Face image is missing.")
            msg.setText('First load face image!')
            msg.setStandardButtons(QMessageBox.Ok)

            msg.exec()
            return
       	
       	return NotImplemented

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())