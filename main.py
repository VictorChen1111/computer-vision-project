import sys
import cv2
import os
import glob
import numpy as np

from PyQt5.QtGui import (
    QImage, 
    QPixmap,
)
from PyQt5.QtWidgets import(
    QApplication, 
    QDialog, 
    QFileDialog, 
    QGridLayout, 
    QLabel, 
    QPushButton,
    QGroupBox,
    QVBoxLayout,
    QComboBox,
    QLineEdit
)

from PyQt5.QtCore import QTimer

class MyWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('2022 OpenCvdl Hw2')
        self.initUI()

    def initUI(self):
        self.resize(800, 600)
        
        self.btnOpen1 = QPushButton('Load Folder', self)
        self.btnOpen2 = QPushButton('Load Image L', self)
        self.btnOpen3 = QPushButton('Load Image R', self)

        self.btnOpen_0_1 = QPushButton('   1.1 Draw Contour   ', self)
        self.btnOpen_0_2 = QPushButton('   1.2 Count Rings   ', self)
        self.label_1 = QLabel('label_1', self)
        self.label_2 = QLabel('label_2', self)

        self.btnOpen_1_1 = QPushButton('   2.1 Find Corners   ', self)
        self.btnOpen_1_2 = QPushButton('   2.2 Find Intrinsic   ', self)

        self.mycombobox = QComboBox(self)
        self.mycombobox.addItems(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'])
        # self.mycombobox.setCurrentIndex(0)

        self.btnOpen_1_3 = QPushButton('   2.3 Find Extrinsic   ', self)
        self.btnOpen_1_4 = QPushButton('   2.4 Find Distortion   ', self)
        self.btnOpen_1_5 = QPushButton('   2.5 Show Result   ', self)

        self.mylineedit = QLineEdit(self)
        self.btnOpen_2_1 = QPushButton('3.1 Show Words on Board', self)
        self.btnOpen_2_2 = QPushButton('3.2 Show Words Vertically', self)

        self.btnOpen_3_1 = QPushButton('4.1 Stero Disparity Map', self)

        layout = QGridLayout(self)
        self.groupbox = QGroupBox("Load Image")
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.btnOpen1)
        self.vbox.addWidget(self.btnOpen2)
        self.vbox.addWidget(self.btnOpen3)
        self.groupbox.setLayout(self.vbox)
        layout.addWidget(self.groupbox, 0, 0)

        
        self.groupbox_2 = QGroupBox("1. Find Contour")
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.mylineedit)
        self.vbox.addWidget(self.btnOpen_0_1)
        self.vbox.addWidget(self.btnOpen_0_2)
        self.vbox.addWidget(self.label_1)
        self.vbox.addWidget(self.label_2)
        self.groupbox_2.setLayout(self.vbox)
        layout.addWidget(self.groupbox_2, 0, 1)


        self.groupbox_1 = QGroupBox("2. Calibration")
        self.groupbox_1_3 = QGroupBox("2.3 Find Extrinsic")
        self.vbox_temp = QVBoxLayout()
        self.vbox_temp.addWidget(self.mycombobox)
        self.vbox_temp.addWidget(self.btnOpen_1_3)
        self.groupbox_1_3.setLayout(self.vbox_temp)

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.btnOpen_1_1)
        self.vbox.addWidget(self.btnOpen_1_2)
        self.vbox.addWidget(self.groupbox_1_3)
        self.vbox.addWidget(self.btnOpen_1_4)
        self.vbox.addWidget(self.btnOpen_1_5)
        self.groupbox_1.setLayout(self.vbox)
        layout.addWidget(self.groupbox_1, 0, 2)

        self.groupbox_2 = QGroupBox("3. Image Smoothing")
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.mylineedit)
        self.vbox.addWidget(self.btnOpen_2_1)
        self.vbox.addWidget(self.btnOpen_2_2)
        self.groupbox_2.setLayout(self.vbox)
        layout.addWidget(self.groupbox_2, 0, 3)

        self.groupbox_3 = QGroupBox("4. Stereo Disparity Map")
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.btnOpen_3_1)
        self.groupbox_3.setLayout(self.vbox)
        layout.addWidget(self.groupbox_3, 0, 4)

        ## First row
        self.btnOpen1.clicked.connect(self.load_folder)
        self.btnOpen2.clicked.connect(self.load_image_l)
        self.btnOpen3.clicked.connect(self.load_image_r)

        ## Second row
        self.btnOpen_0_1.clicked.connect(self.draw_contour)
        self.btnOpen_0_2.clicked.connect(self.count_rings)

        ## Third row
        self.btnOpen_1_1.clicked.connect(self.find_corners)
        self.btnOpen_1_2.clicked.connect(self.find_intrinsic)
        self.btnOpen_1_3.clicked.connect(self.find_extrinsic)
        self.btnOpen_1_4.clicked.connect(self.find_distortion)
        self.btnOpen_1_5.clicked.connect(self.show_result)

        ## Forth row
        self.btnOpen_2_1.clicked.connect(self.show_words_on_board)
        self.btnOpen_2_2.clicked.connect(self.show_words_vertically)

        self.btnOpen_3_1.clicked.connect(self.stereo_disparity_map)

    def load_folder(self):
        self.folder_path = QFileDialog.getExistingDirectory(self, "Open folder", "./") 
        # print(folder_path)
        self.pic_list = []
        self.text_path = ''
        for file in os.walk(self.folder_path):
            # print(len(file))
            self.text_path = file[0]
            dirPath = file[0]
            for f in file[2]:
                self.pic_list.append(os.path.join(dirPath, f))
        # print(self.pic_list)
        # print(folder_path, '/*.bmp')

    def load_image_l(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Image', 'Image', '*.png *.jpg *.bmp')
        if filename == '':
                return
        self.img_l = cv2.imread(filename, -1)
        self.img_l_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if self.img_l.size == 1:
            return
    
    def load_image_r(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Image', 'Image', '*.png *.jpg *.bmp')
        if filename == '':
                return
        self.img_r = cv2.imread(filename, -1)
        self.img_r_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        if self.img_r.size == 1:
            return

    def draw_contour(self):
        # img_original_1 = cv2.imread(r'C:\Users\ASUS\Desktop\Dataset_OpenCvDl_Hw2\Q1_Image\img1.jpg')
        # img_original_2 = cv2.imread(r'C:\Users\ASUS\Desktop\Dataset_OpenCvDl_Hw2\Q1_Image\img2.jpg')
        img_original_1 = self.img_l
        img_original_2 = self.img_r

        img_gray_1 = cv2.cvtColor(img_original_1,cv2.COLOR_BGR2GRAY)
        img_gray_2 = cv2.cvtColor(img_original_2,cv2.COLOR_BGR2GRAY)

        gray_blur_1 = cv2.GaussianBlur(img_gray_1, (15, 15), cv2.THRESH_BINARY)
        gray_blur_2 = cv2.GaussianBlur(img_gray_2, (15, 15), cv2.THRESH_BINARY)

        ret, thresh_1 = cv2.threshold(gray_blur_1, 126, 255, 0)
        ret, thresh_2 = cv2.threshold(gray_blur_2, 126, 255, 0)

        contours_1,hierarchy = cv2.findContours(thresh_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_2,hierarchy = cv2.findContours(thresh_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(img_original_1, contours_1, -1, (0, 255, 0), 3)
        cv2.drawContours(img_original_2, contours_2, -1, (0, 255, 0), 3)

        cv2.imshow('Contours_1', img_original_1)
        cv2.imshow('Contours_2', img_original_2)

        self.coin_count_1 = len(contours_1)
        self.coin_count_2 = len(contours_2)


    def count_rings(self):

        text_1 = 'There are ' + str(int(self.coin_count_1/2)) + ' coins in img1.jpg'
        text_2 = 'There are ' + str(int(self.coin_count_2/2)) + ' coins in img2.jpg'

        self.label_1.setText(text_1)
        self.label_2.setText(text_2)


    def find_corners(self):
        for pic in self.pic_list:
            cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            img = cv2.imread(pic)  
            ret, cp_img = cv2.findChessboardCorners(img, (11, 8))
            cv2.drawChessboardCorners(img, (11, 8), cp_img, ret)
            cv2.imshow('Image', img)
            key = cv2.waitKey(500)
            if key == 27:
                cv2.destroyAllWindows()
                break

    def find_intrinsic(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane

        # Select any index to grab an image from the list
        for i in range(len(self.pic_list)):
            # Read in the image
            image = cv2.imread(self.pic_list[i])
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (8, 11), None)

            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
                imgpoints.append(corners2)

        # gray.shape[::-1] = (2048, 2048)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)
        print(f"Intrinsic Matrix:\n{mtx}")

    def find_extrinsic(self):
        # cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        if(self.mycombobox.currentText()):
            image_path = self.pic_list[int(self.mycombobox.currentText())-1]
            # cv2.imshow('Image', image)
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane

        # Read in the image
        image = cv2.imread(image_path)
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

        # gray.shape[::-1] = (2048, 2048)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
        R = cv2.Rodrigues(rvecs[0])
        ext = np.hstack((R[0], tvecs[0]))
        print(f"Extrinsic Matrix:\n{ext}")

    def find_distortion(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane

        # Select any index to grab an image from the list
        for i in range(len(self.pic_list)):
            # Read in the image
            image = cv2.imread(self.pic_list[i])
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (8, 11), None)

            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
                imgpoints.append(corners2)

        # gray.shape[::-1] = (2048, 2048)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)
        # print(dist)
        print(f"Distortion Matrix:\n{dist}")

    def show_result(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane

        # Select any index to grab an image from the list
        for i in range(len(self.pic_list)):
            # Read in the image
            image = cv2.imread(self.pic_list[i])
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (8, 11), None)

            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
                imgpoints.append(corners2)

        # gray.shape[::-1] = (2048, 2048)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)
        for pic in self.pic_list:
            cv2.namedWindow('Distort', cv2.WINDOW_NORMAL)
            cv2.moveWindow('Distort', 10,50)
            cv2.namedWindow('Undistort', cv2.WINDOW_NORMAL)
            cv2.moveWindow('Undistort', 650,50)

            img = cv2.imread(pic)  
            h,  w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

            # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            # crop the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]

            height = int(img.shape[0] / 4)
            width = int(img.shape[1] / 4)
            cv2.resizeWindow("Distort", width, height)
            cv2.resizeWindow("Undistort", width, height)

            cv2.imshow('Distort', dst)
            cv2.imshow('Undistort', img)
            key = cv2.waitKey(500)
            if key == 27:
                cv2.destroyAllWindows()
                break


    def show_words_on_board(self):
        if self.mylineedit.text() != '':
            self.word = self.mylineedit.text()
            print(self.word)
        file_storage = cv2.FileStorage(f'{self.text_path}/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
        self.chessboard_ar(file_storage)
    
    def show_words_vertically(self):
        file_storage = cv2.FileStorage(f'{self.text_path}/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
        print("inside word vertically")
        self.chessboard_ar(file_storage)

    def show_word(self, img, lines, rvecs, tvecs, mtx, dist):
        # Space for showing words
        space = np.array([[7, 5, 0], [4, 5, 0], [1, 5, 0], [7, 2, 0], [4, 2, 0], [1, 2, 0]])

        # Shift the word to the Space
        shift = np.ndarray.tolist(np.zeros(len(self.word)))
        for i, line in enumerate(lines):
            shift[i] = np.float32(line + space[i])
            shift[i] = shift[i].reshape(-1, 3)
        shift = np.concatenate(tuple(shift), axis=0)

        # project 3D points to image plane
        imgpts, _ = cv2.projectPoints(shift, rvecs, tvecs, mtx, dist)

        imgpts = imgpts.reshape(int(imgpts.shape[0] / 2), 2, 2).astype(int)

        # draw lines
        for line, _ in enumerate(imgpts):
            img = cv2.line(img, tuple(imgpts[line][0]), tuple(imgpts[line][1]), (0, 0, 255), 2)
   

    def chessboard_ar(self, file_storage):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        # print("inside self")
        images = glob.glob(self.folder_path + '/*.bmp')
        for fname in images:
            img = cv2.imread(fname)
            img_height = int(img.shape[1] / 4)
            img_width = int(img.shape[0] / 4)
            img = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                        objpoints, imgpoints, gray.shape[::-1], None, None)
                # Find the rotation and translation vectors.
                _, rvecs, tvecs, _ = cv2.solvePnPRansac(objp, corners2, mtx, dist)

                # get data from library
                print(self.word)
                alphabet = np.ndarray.tolist(np.zeros(len(self.word)))
                for word, _ in enumerate(self.word):
                    alphabet[word] = file_storage.getNode(self.word[word]).mat()

                # Show word on chessboard
                self.show_word(img, alphabet, rvecs, tvecs, mtx, dist)

                cv2.imshow('AR', img)
                cv2.waitKey(1000)
            cv2.destroyWindow('AR')
    
    
    def stereo_disparity_map(self):

        self.BASELINE = 342.789  # mm
        self.FOCAL_LENGTH = 4019.284  # pixel
        self.DOFFS = 279.184  # pixel

        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        self.disparity = stereo.compute(self.img_l_gray, self.img_r_gray)
        self.disparity = cv2.normalize(self.disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        cv2.namedWindow('imgL',cv2.WINDOW_NORMAL)
        cv2.resizeWindow("imgL", int(self.disparity.shape[1]/4), int(self.disparity.shape[0]/4))
        cv2.setMouseCallback('imgL', self.draw_circle)
        cv2.imshow('imgL', self.img_l)

        cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("disparity", int(self.disparity.shape[1]/4), int(self.disparity.shape[0]/4))
        cv2.imshow('disparity', self.disparity)

        cv2.namedWindow('imgR_dot', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("imgR_dot", int(self.img_r.shape[1]/4), int(self.img_r.shape[0]/4))
        cv2.imshow('imgR_dot', self.img_r)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_circle(self, event, x, y, flags, param):

        # when click
        if event == cv2.EVENT_LBUTTONDOWN:

            img = cv2.cvtColor(np.copy(self.disparity), cv2.COLOR_GRAY2BGR)
            img_dot = cv2.cvtColor(np.copy(self.disparity), cv2.COLOR_GRAY2BGR)
            cv2.circle(img_dot, (x, y), 10, (255, 0, 0), -1)

            z = img[y][x][0]
            depth=self.BASELINE*self.FOCAL_LENGTH/(z + self.DOFFS)
            print(x,y)
            print(depth)

            if z != 0:
                cv2.circle(self.img_r,(x-z,y), 15, (0,255,0), -1)

            cv2.namedWindow('imgR_dot', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("imgR_dot", int(self.img_r.shape[1]/4), int(self.img_r.shape[0]/4))
            cv2.imshow('imgR_dot', self.img_r)
            cv2.waitKey(0)


if __name__ == '__main__':
    a = QApplication(sys.argv)
    dialog = MyWindow()
    dialog.show()
    sys.exit(a.exec_())