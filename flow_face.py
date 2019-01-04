#!/usr/local/bin/python3

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QMessageBox, QWidget, QGroupBox, \
    QAction, QFileDialog, QGridLayout, QLabel, qApp
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np
import cv2

FNAME_SEQUENCE_IMG = '/taxi_'
FNAME_FACE_IMG = '/HR_'
NUM_SEQUENCE_IMG = 41
NUM_FACE_IMG = 32

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
        # This function is called when the user clicks File->Open optical flow directory.
        opticalFlowDir = QFileDialog.getExistingDirectory(self, "Select directory of optical flow images")

        # File open dialog has been dismissed or file could not be found
        if opticalFlowDir is '':
            return

        # If there is an input image loaded, remove it
        if self.opticalFlowLoaded:
            self.deleteItemsFromWidget(self.opticalFlowGroupBox.layout())

        self.sequenceImages = []
        for t in range(NUM_SEQUENCE_IMG):
            if t < 10:
                fName = opticalFlowDir + FNAME_SEQUENCE_IMG + '000' + str(t) + '.jpg'
            else:
                fName = opticalFlowDir + FNAME_SEQUENCE_IMG + '00' + str(t) + '.jpg'

            image = cv2.imread(fName, cv2.IMREAD_GRAYSCALE)
            if image is None:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Taxi image not found.")
                msg.setText('Taxi image not found in given directory!')
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec()
                return

            self.sequenceImages.append(cv2.resize(image,None,fx=2, fy=2, interpolation=cv2.INTER_CUBIC)) # Read sequence image

        self.firstOpticalFlowImage = cv2.cvtColor(self.sequenceImages[0], cv2.COLOR_GRAY2RGB)

        self.opticalFlowLoaded = True
        self.addImageToGroupBox(self.firstOpticalFlowImage, self.opticalFlowGroupBox, 'Optical flow image')

    def openFaceImage(self):
        # This function is called when the user clicks File->Open face database directory.
        faceDatabaseDir = QFileDialog.getExistingDirectory(self, "Select directory of face images")

        # File open dialog has been dismissed or file could not be found
        if faceDatabaseDir is '':
            return

        # If there is an input image loaded, remove it
        if self.faceLoaded:
            self.deleteItemsFromWidget(self.faceGroupBox.layout())

        self.faceDatabase = []
        for t in range(1,NUM_FACE_IMG+1):
            fName = faceDatabaseDir + FNAME_FACE_IMG + str(t) + '.tif'
            image = cv2.imread(fName, cv2.IMREAD_GRAYSCALE)

            if image is None:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Face image not found.")
                msg.setText('Face image not found in given directory!')
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec()
                return

            self.faceDatabase.append(image) # Read sequence image

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText('Now, select a face image to be detected.')
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

        fName = QFileDialog.getOpenFileName(self, 'Open face to be detected', faceDatabaseDir, 'Image files (*.tif)')

        # File open dialog has been dismissed or file could not be found
        if fName[0] is '':
            return

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
        opticalFlowAct = QAction('Open optical flow directory', self)
        opticalFlowAct.triggered.connect(self.openOpticalFlowImage)

        faceAct = QAction('Open face database directory', self) 
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
        recognizeFaceAct.triggered.connect(self.recognizeFaceButtonClicked)
        
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
            # Error: "First load optical flow images" in MessageBox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Optical flow images are missing.")
            msg.setText('First load optical flow images!')
            msg.setStandardButtons(QMessageBox.Ok)

            msg.exec()
            return

        height, width = self.sequenceImages[0].shape
        
        frames = []
        for f in range(len(self.sequenceImages)-1):
            V = self.findOpticalFlowVectors(self.sequenceImages[f], self.sequenceImages[f+1], 50)
            image = cv2.cvtColor(self.sequenceImages[f].copy(), cv2.COLOR_GRAY2RGB)

            for h in range(25, height-25, 25):
                for w in range(25, width-25, 25):
                    point1 = (w,h)
                    point2 = (w+int(round(V[h,w,1])), h+int(round(V[h,w,0])))
                    cv2.arrowedLine(image, point1, point2, (0,0,255), 1, cv2.LINE_AA, 0, 0.2)

            frames.append(image) 

            self.deleteItemsFromWidget(self.opticalFlowGroupBox.layout())
            self.addImageToGroupBox(image, self.opticalFlowGroupBox, 'Optical flow image')

    def findOpticalFlowVectors(self, image1, image2, interval):
        height, width = image1.shape

        Ix = np.zeros((height, width), dtype=np.float64) # Gradient x
        Iy = np.zeros((height, width), dtype=np.float64) # Gradient y
        It = np.zeros((height, width), dtype=np.float64) # Gradient t

        image1 = image1.astype(np.float64)
        image2 = image2.astype(np.float64)

        for h in range(1,height-1):
            for w in range(1,width-1):
                Ix[h,w] = (image1[h+1,w] - image1[h-1,w]) / 2 # X gradient of image pixel
                Iy[h,w] = (image1[h,w+1] - image1[h,w-1]) / 2 # Y gradient of image pixel
                It[h,w] = image2[h,w] - image1[h,w] # Time gradient of images

        V = np.zeros((height,width,2), dtype=np.float64)

        s = int(interval/2)
        e = int(round(interval/2))
        for h in range(s, height-e):
            for w in range(s, width-e):
                dx = Ix[h-s:h+e,w-s:w+e]
                dy = Iy[h-s:h+e,w-s:w+e]
                dt = It[h-s:h+e,w-s:w+e]

                T = np.zeros((2,2), dtype=np.float64) # Structure tensor Transpose(A)*A
                T[0,0] = np.sum(dx*dx, dtype=np.float64)
                T[0,1] = np.sum(dx*dy, dtype=np.float64)
                T[1,0] = T[0,1]
                T[1,1] = np.sum(dy*dy, dtype=np.float64)

                Atb = np.zeros((2,1), dtype=np.float64) # Transpose(A)*b
                Atb[0] = -1 * np.sum(dx*dt, dtype=np.float64)
                Atb[1] = -1 * np.sum(dy*dt, dtype=np.float64)

                det = T[0,0] * T[1,1] - T[1,0]**2

                A = np.vstack((dx.flatten(), dy.flatten())).T

                if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) < 0.01:
                    continue

                if det != 0:
                    a = np.matmul(np.linalg.inv(T), Atb)
                    V[h,w,0] = a[0]
                    V[h,w,1] = a[1]

        return V * 10

    def recognizeFaceButtonClicked(self):
        if not self.faceLoaded:
            # Error: "First load face database and face image" in MessageBox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Face database and face image are missing.")
            msg.setText('First load face database and face image!')
            msg.setStandardButtons(QMessageBox.Ok)

            msg.exec()
            return
        
        return NotImplemented

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())