#!/usr/local/bin/python3

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QMessageBox, QWidget, QGroupBox, \
    QAction, QFileDialog, QGridLayout, QLabel, qApp
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QUrl
import numpy as np
import cv2
import os

FNAME_SEQUENCE_IMG = '/taxi_'
FNAME_FACE_IMG = '/HR_'
NUM_SEQUENCE_IMG = 41
NUM_FACE_IMG = 32

INTERVAL = 15

class App(QMainWindow):
    def __init__(self):
        super(App, self).__init__()

        self.title = 'Lucas-Kanade optical flow estimation and Face recognition with eigenfaces'

        # Booelans to track if input images are loaded
        self.opticalFlowLoaded = False
        self.faceLoaded = False

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

            self.sequenceImages.append(image) # Read sequence image

        firstOpticalFlowImage = cv2.cvtColor(self.sequenceImages[0], cv2.COLOR_GRAY2RGB)
        firstOpticalFlowImage = cv2.resize(firstOpticalFlowImage, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        self.opticalFlowLoaded = True
        self.addImageToGroupBox(firstOpticalFlowImage, self.opticalFlowGroupBox, 'Optical flow image')

    def openFaceImage(self):
        # This function is called when the user clicks File->Open face database directory.
        faceDatabaseDir = QFileDialog.getExistingDirectory(self, "Select directory of face images")

        # File open dialog has been dismissed or file could not be found
        if faceDatabaseDir is '':
            return

        # If there is an input image loaded, remove it
        if self.faceLoaded:
            self.deleteItemsFromWidget(self.meanImageGroupBox.layout())
            for i in range(5):
                self.deleteItemsFromWidget(self.eigenfaceGroupBoxes[i].layout())
            self.deleteItemsFromWidget(self.inputFaceGroupBox.layout())
            self.deleteItemsFromWidget(self.recognizedFaceGroupBox.layout())

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
        msg.setText('Now, select a face image to be recognized.')
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

        fName = QFileDialog.getOpenFileName(self, 'Open face to be recognized', faceDatabaseDir, 'Image files (*.tif)')

        # File open dialog has been dismissed or file could not be found
        if fName[0] is '':
            return

        self.faceImage = cv2.imread(fName[0]) # Read the image

        self.faceLoaded = True
        self.addImageToGroupBox(self.faceImage, self.inputFaceGroupBox, 'Face recognition image')

    def createEmptyOpticalFlowGroupBox(self):
        self.opticalFlowGroupBox = QGroupBox('Optical Flow Estimation')
        layout = QVBoxLayout()

        self.opticalFlowGroupBox.setLayout(layout)

    def createEmptyFaceRecognitionGroupBox(self):
        self.faceGroupBox = QGroupBox('Face Recognition using Eigenfaces')
        layout = QGridLayout()

        self.meanImageGroupBox = QGroupBox('Mean image')
        meanImageLayout = QVBoxLayout()
        self.meanImageGroupBox.setLayout(meanImageLayout)

        self.eigenfaceGroupBoxes = []

        firstEigenfaceGroupBox = QGroupBox('First Eigenface')
        firstEigenfaceLayout = QVBoxLayout()
        firstEigenfaceGroupBox.setLayout(firstEigenfaceLayout)
        self.eigenfaceGroupBoxes.append(firstEigenfaceGroupBox)

        secondEigenfaceGroupBox = QGroupBox('Second Eigenface')
        secondEigenfaceLayout = QVBoxLayout()
        secondEigenfaceGroupBox.setLayout(secondEigenfaceLayout)
        self.eigenfaceGroupBoxes.append(secondEigenfaceGroupBox)

        thirdEigenfaceGroupBox = QGroupBox('Third Eigenface')
        thirdEigenfaceLayout = QVBoxLayout()
        thirdEigenfaceGroupBox.setLayout(thirdEigenfaceLayout)
        self.eigenfaceGroupBoxes.append(thirdEigenfaceGroupBox)

        fourthEigenfaceGroupBox = QGroupBox('Fourth Eigenface')
        fourthEigenfaceLayout = QVBoxLayout()
        fourthEigenfaceGroupBox.setLayout(fourthEigenfaceLayout)
        self.eigenfaceGroupBoxes.append(fourthEigenfaceGroupBox)

        fifthEigenfaceGroupBox = QGroupBox('Fifth Eigenface')
        fifthEigenfaceLayout = QVBoxLayout()
        fifthEigenfaceGroupBox.setLayout(fifthEigenfaceLayout)
        self.eigenfaceGroupBoxes.append(fifthEigenfaceGroupBox)

        layout.addWidget(self.meanImageGroupBox, 0, 0, 1, 2)
        layout.addWidget(firstEigenfaceGroupBox, 0, 2, 1, 2)
        layout.addWidget(secondEigenfaceGroupBox, 0, 4, 1, 2)
        layout.addWidget(thirdEigenfaceGroupBox, 1, 0, 1, 2)
        layout.addWidget(fourthEigenfaceGroupBox, 1, 2, 1, 2)
        layout.addWidget(fifthEigenfaceGroupBox, 1, 4, 1, 2)

        self.inputFaceGroupBox = QGroupBox('Input image')
        inputFaceLayout = QVBoxLayout()
        self.inputFaceGroupBox.setLayout(inputFaceLayout)

        self.recognizedFaceGroupBox = QGroupBox('Recognized face')
        recognizedFaceLayout = QVBoxLayout()
        self.recognizedFaceGroupBox.setLayout(recognizedFaceLayout)

        layout.addWidget(self.inputFaceGroupBox, 2, 0, 1, 3)
        layout.addWidget(self.recognizedFaceGroupBox, 2, 3, 1, 3)

        self.faceGroupBox.setLayout(layout)

    def mediaStateChanged(self, state):
        self.mediaPlayer.play()

    def handleError(self):
        print("Error: " + self.mediaPlayer.errorString())

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

        fName = 'optical_flow.mp4'
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(fName, fourcc, 10, (width*2, height*2))

        size = 35
        s = int(size/2)
        e = int(round(size/2))
        
        for f in range(len(self.sequenceImages)-1):
            V = self.findOpticalFlowVectors(self.sequenceImages[f], self.sequenceImages[f+1], size)
            image = cv2.cvtColor(self.sequenceImages[f].copy(), cv2.COLOR_GRAY2RGB)
            image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            for h in range(s, height-e, INTERVAL):
                for w in range(s, width-e, INTERVAL):
                    point1 = (w*2,h*2)
                    point2 = ((w+int(round(V[h,w,1])))*2, (h+int(round(V[h,w,0])))*2)
                    cv2.arrowedLine(image, point1, point2, (0,255,0), 1, 8, 0, 0.25)

            video.write(image)

        video.release()

        videoWidget = QVideoWidget()
        videoWidget.setFixedSize(width*2, height*2)
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.error.connect(self.handleError)
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.dirname(os.path.abspath(__file__)) + '/' + fName)))
        self.mediaPlayer.play()

        self.deleteItemsFromWidget(self.opticalFlowGroupBox.layout())
        self.opticalFlowGroupBox.layout().addWidget(videoWidget, 0, Qt.AlignCenter)

    def findOpticalFlowVectors(self, image1, image2, size):
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

        s = int(size/2)
        e = int(round(size/2))

        for h in range(s, height-e, INTERVAL):
            for w in range(s, width-e, INTERVAL):
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
                    velocity = np.matmul(np.linalg.inv(T), Atb)
                    V[h,w,0] = velocity[0]
                    V[h,w,1] = velocity[1]

        return V * 5

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

        height, width = self.faceDatabase[0].shape
        dbSize = len(self.faceDatabase)

        X = np.zeros((dbSize,height*width))
        for i in range(dbSize):
            X[i] = self.faceDatabase[i].flatten()

        mu_X = np.sum(X, axis=0, dtype=np.float64) / dbSize
        Z = X - mu_X

        meanImage = self.flattenArrayToImage(mu_X, height, width)
        self.addImageToGroupBox(meanImage, self.meanImageGroupBox, 'Face recognition image')
        
        _, s, vh = np.linalg.svd(Z, full_matrices=False)

        eigenvalues = s**2
        B = vh.T # Eigenvectors

        grayFaceImage = cv2.cvtColor(self.faceImage, cv2.COLOR_BGR2GRAY)
        smoothedFaceImage = cv2.GaussianBlur(grayFaceImage, (5,5), 1, None, 0)
        smoothedFaceImage = smoothedFaceImage.flatten()

        d = B.shape[1]

        for j in range(5):
            eigenface = self.flattenArrayToImage(B[:,j].T, height, width)
            self.addImageToGroupBox(eigenface, self.eigenfaceGroupBoxes[j], 'Eigenface image')

        p_est = np.zeros(d)
        y_est = np.zeros((NUM_FACE_IMG, d))
        for j in range(d):
            p_est[j] = np.matmul(B[:,j].T, (smoothedFaceImage - mu_X).T)

            for i in range(NUM_FACE_IMG):
                y_est[i, j] = np.matmul(B[:,j].T, Z[i].T)

        error = np.zeros(NUM_FACE_IMG)
        for i in range(NUM_FACE_IMG):
            error[i] = np.sum((y_est[i] - p_est)**2)

        closestIndex = np.argmin(error)

        x_i = mu_X + np.matmul(B, y_est[closestIndex])
        x_i = self.flattenArrayToImage(x_i, height, width)
        self.addImageToGroupBox(x_i, self.recognizedFaceGroupBox, 'Face recognition image')

    def flattenArrayToImage(self, array, height, width):
        image = array.copy()
        image -= np.min(image)
        image *= (255 / np.max(image))
        image = image.astype(np.uint8)
        image = image.reshape((height, width))
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())