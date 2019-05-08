# Code for the 3rd Computer Vision Task, on 2/5/2019
# Authors @ Abdelrahman Ahmed Ramzy, Ahmed Fawzi Hosni, Moaz Khairy Hussien



import sys
import numpy as np
import pandas as pd
import os
import argparse
import time
from random import randint
# PyQt5
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSlot, QSize, QRect
from PyQt5.QtWidgets import QApplication, QDialog, QWidget, QLabel, QMessageBox, QMainWindow, QFileDialog, QComboBox, \
    QRadioButton, QLineEdit
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon, QPixmap, QMouseEvent, QPainter
from PyQt5.QtCore import Qt
# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import pyqtgraph as pg
# Image Processing
import cv2
import argparse
from skimage import color
from skimage.transform import resize
# Scipy
from scipy import signal
from scipy import misc
import scipy.fftpack as fp
# Math
import math
from math import sqrt, atan2, pi, cos, sin
from collections import defaultdict
from skimage import img_as_float
from skimage.filters import sobel, gaussian
#from skimage.draw import circle_perimeter
from scipy import signal
from scipy import misc
from skimage import color
from scipy.interpolate import RectBivariateSpline

# References
# [1] https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.convolve2d.html
# [2] https://en.wikipedia.org/wiki/Kernel_(image_processing)
# [3] https://subscription.packtpub.com/book/application_development/9781785283932/2/ch02lvl1sec22/sharpening
# [4] https://www.codingame.com/playgrounds/2524/basic-image-manipulation/filtering
# [5] https://python-reference.readthedocs.io/en/latest/docs/functions/complex.html
# [6] https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.fft.html
# [7] http://www.pyqtgraph.org/documentation/graphicsItems/imageitem.html#pyqtgraph.ImageItem.setLookupTable
# [8] https://www.afternerd.com/blog/python-lambdas/
# [9] https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
# [10] http://me.umn.edu/courses/me5286/vision/Notes/2015/ME5286-Lecture9.pdf
# [11] https://github.com/PavanGJ/Circle-Hough-Transform/blob/master/main.
# [12] Sections and Toaa Tarek, Sarah Mohamed & Mahmoud Hamza for Hough Line transform
# [13] https://www.qtcentre.org/threads/69517-How-to-get-mouse-click-position-of-QLabel-which-is-a-child-of-QMainWindow-in-PyQt5
# [14] Mean Shift : https://github.com/parag-datar/mean-shift-clustering
# [15] Snake : https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/active_contour_model.py#L7
# [16] K-mean : https://www.imageeprocessing.com/2017/12/k-means-clustering-on-rgb-image.html


def OpenedFile(fileName):
    i = len(fileName) - 1
    j = -1
    x = 1

    while x == 1:
        if fileName[i] != '/':
            j += 1
            i -= 1
        else:
            x = 0
    File_Names = np.zeros(j + 1)

    # Convert from Float to a list of strings
    File_Name = ["%.2f" % number for number in File_Names]
    for k in range(0, j + 1):
        File_Name[k] = fileName[len(fileName) - 1 + k - j]  # List of Strings
    # Convert list of strings to a string
    FileName = ''.join(File_Name)  # String
    return FileName


def image_parameters(imagergb):
    imagegray = color.rgb2gray(imagergb)  # np.dot(image[..., :3], [0.299, 0.587, 0.114])
    max = np.max(imagegray)
    min = np.min(imagegray)
    image_size = np.shape(imagegray)
    return imagegray, max, min, image_size


def gaussian_kernel(kernel, std):
    """Returns a 2D Gaussian kernel array."""
    Gaussian_Kernel_1 = signal.gaussian(kernel, std=std).reshape(kernel, 1)
    Gaussian_Kernel_2 = np.outer(Gaussian_Kernel_1, Gaussian_Kernel_1)
    return Gaussian_Kernel_2


def Histogram_Generator(Gray_image, U_max, U_min, size):
    histogram = np.zeros((2, 256), dtype=int)
    for i in range(256):
        histogram[0, i] = i
    # Linear Scaling & Histogram Calculating
    a = -1 * U_min
    b = 255 / (U_max - U_min)
    for i in range(size[0]):
        for j in range(size[1]):
            intensity = math.floor(b * (Gray_image[i, j] + a))
            histogram[1, intensity] = histogram[1, intensity] + 1
    return histogram


def Histogram_Equalization(Gray_image, U_max, U_min, size):
    Histogram = np.zeros((4, 256), dtype=int)
    new_Gray_image = np.array(Gray_image.copy(), dtype=int)
    Pixel_Count = np.size(Gray_image)
    # First Row
    for i in range(256):
        Histogram[0, i] = i

    # Linear Scaling & Histogram Calculating
    a = -1 * U_min
    b = 255 / (U_max - U_min)

    for i in range(size[0]):
        for j in range(size[1]):
            intensity = math.floor(b * (Gray_image[i, j] + a))
            new_Gray_image[i, j] = intensity
            Histogram[1, intensity] = Histogram[1, intensity] + 1

    # Histogram Equalization
    # apply CDF
    Histogram[2, 0] = Histogram[1, 0]
    for i in range(1, 256):
        Histogram[2, i] = Histogram[2, i - 1] + Histogram[1, i]
    for i in range(256):
        Histogram[2, i] = math.floor(Histogram[2, i] * 255)
    # apply Normalization
    for i in range(256):
        Histogram[3, i] = Histogram[2, i] / Pixel_Count
    return (Histogram, new_Gray_image)


# Reference [9]
def Canny_Edge_Detection(ImageGRAY):
    # Noise Reduction
    GaussianK = 5
    GaussianSTD = 4
    Gaussian_Kernel_2 = gaussian_kernel(GaussianK, GaussianSTD)
    Image_Noise_Reduced = signal.convolve2d(ImageGRAY, Gaussian_Kernel_2, boundary='symm', mode='same')

    # Gradient Calculation
    # Edge detection by Sobel then calculate the magnitude and direction
    X_Sobel_Kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Y_Sobel_Kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    X_Sobel_Image = signal.convolve2d(Image_Noise_Reduced, X_Sobel_Kernel, boundary='symm', mode='same')
    Y_Sobel_Image = signal.convolve2d(Image_Noise_Reduced, Y_Sobel_Kernel, boundary='symm', mode='same')
    # numpy.hypot: Equivalent to sqrt(x1**2 + x2**2)
    Image_Gradient = np.hypot(X_Sobel_Image, Y_Sobel_Image)
    Image_Gradient = (Image_Gradient / Image_Gradient.max()) * 255
    theta = np.arctan2(Y_Sobel_Image, X_Sobel_Image) * (180.0 / np.pi)

    # Non-Maximum Suppression
    # The final image should have thin edges.
    # Thus, we must perform non-maximum suppression to thin out the edges
    (M, N) = Image_Gradient.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = theta.copy()
    angle[angle < 0] += 180
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = Image_Gradient[i, j + 1]
                    r = Image_Gradient[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = Image_Gradient[i + 1, j + 1]
                    r = Image_Gradient[i - 1, j - 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = Image_Gradient[i + 1, j]
                    r = Image_Gradient[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = Image_Gradient[i + 1, j - 1]
                    r = Image_Gradient[i - 1, j + 1]

                if (Image_Gradient[i, j] >= q) and (Image_Gradient[i, j] >= r):
                    Z[i, j] = Image_Gradient[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass
    Image_NonMax_Supression = Z.copy()

    # Double threshold
    # The double threshold step aims at identifying 3 kinds of pixels: strong, weak, and non-relevant

    highThreshold = 120
    lowThreshold = 30

    (M, N) = Image_NonMax_Supression.shape
    Image_Double_Threshold = np.zeros((M, N), dtype=np.int32)
    weak = np.int32(50)
    strong = np.int32(255)
    strong_i, strong_j = np.where(Image_NonMax_Supression >= highThreshold)
    zeros_i, zeros_j = np.where(Image_NonMax_Supression <= lowThreshold)
    weak_i, weak_j = np.where((Image_NonMax_Supression < highThreshold) & (Image_NonMax_Supression > lowThreshold))

    Image_Double_Threshold[strong_i, strong_j] = strong
    Image_Double_Threshold[weak_i, weak_j] = weak
    Image_Double_Threshold[zeros_i, zeros_j] = 0

    # Edge Tracking by Hysteresis
    # Hysteresis consists of transforming weak pixels into strong ones,
    # iff at least one of the pixels around the one being processed is a strong one
    (M, N) = Image_Double_Threshold.shape
    Image_Hysteresis = Image_Double_Threshold.copy()
    for sure in range(1, 3):
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if (Image_Hysteresis[i, j] == weak):
                    try:
                        if ((Image_Hysteresis[i + 2, j - 2] == strong)
                                or (Image_Hysteresis[i + 2, j - 1] == strong)
                                or (Image_Hysteresis[i + 2, j] == strong)
                                or (Image_Hysteresis[i + 2, j + 1] == strong)
                                or (Image_Hysteresis[i + 2, j + 2] == strong)

                                or (Image_Hysteresis[i + 1, j - 2] == strong)
                                or (Image_Hysteresis[i + 1, j - 1] == strong)
                                or (Image_Hysteresis[i + 1, j] == strong)
                                or (Image_Hysteresis[i + 1, j + 1] == strong)
                                or (Image_Hysteresis[i + 1, j + 2] == strong)

                                or (Image_Hysteresis[i, j - 2] == strong)
                                or (Image_Hysteresis[i, j - 1] == strong)

                                or (Image_Hysteresis[i, j + 1] == strong)
                                or (Image_Hysteresis[i, j + 2] == strong)

                                or (Image_Hysteresis[i - 1, j - 2] == strong)
                                or (Image_Hysteresis[i - 1, j - 1] == strong)
                                or (Image_Hysteresis[i - 1, j] == strong)
                                or (Image_Hysteresis[i - 1, j + 1] == strong)
                                or (Image_Hysteresis[i - 1, j + 2] == strong)

                                or (Image_Hysteresis[i - 2, j - 2] == strong)
                                or (Image_Hysteresis[i - 2, j - 1] == strong)
                                or (Image_Hysteresis[i - 2, j] == strong)
                                or (Image_Hysteresis[i - 2, j + 1] == strong)
                                or (Image_Hysteresis[i - 2, j + 2] == strong)):

                            Image_Hysteresis[i, j] = strong
                        else:
                            Image_Hysteresis[i, j] = 0
                    except IndexError as e:
                        pass
                if (Image_Hysteresis[i, j] == 0):
                    try:
                        if (((Image_Hysteresis[i + 1, j - 1] == strong) and (Image_Hysteresis[i - 1, j + 1] == strong))
                                or ((Image_Hysteresis[i + 1, j] == strong) and (Image_Hysteresis[i - 1, j] == strong))
                                or ((Image_Hysteresis[i + 1, j + 1] == strong) and (
                                        Image_Hysteresis[i - 1, j - 1] == strong))
                                or ((Image_Hysteresis[i, j + 1] == strong) and (Image_Hysteresis[i, j - 1] == strong))):
                            Image_Hysteresis[i, j] = strong
                    except IndexError as e:
                        pass
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (Image_Hysteresis[i, j] == strong) or (Image_Hysteresis[i, j] == weak):
                try:
                    if ((Image_Hysteresis[i + 1, j - 1] == 0)
                            and (Image_Hysteresis[i + 1, j] == 0)
                            and (Image_Hysteresis[i + 1, j + 1] == 0)
                            and (Image_Hysteresis[i, j - 1] == 0)
                            and (Image_Hysteresis[i, j + 1] == 0)
                            and (Image_Hysteresis[i - 1, j - 1] == 0)
                            and (Image_Hysteresis[i - 1, j] == 0)
                            and (Image_Hysteresis[i - 1, j + 1] == 0)):
                        Image_Hysteresis[i, j] = 0

                except IndexError as e:
                    pass
    return Image_Hysteresis


def detectCircles(img, threshold, region, radius=None):
    (M, N) = img.shape  # Get the maximum rows and columns
    if radius == None:
        R_max = np.max((M, N))
        R_min = 3
    else:  # Determine the maximum and minimum radiuses available in the image
        [R_max, R_min] = radius

    R = R_max - R_min
    #Initializing accumulator array.
    #Accumulator array is a 3 dimensional array with the dimensions representing
    #the radius, X coordinate and Y coordinate resectively.
    #Also appending a padding of 2 times R_max to overcome the problems of overflow
    A = np.zeros((R_max, M+2*R_max, N+2*R_max))
    B = np.zeros((R_max, M+2*R_max, N+2*R_max))

    #Precomputing all angles to increase the speed of the algorithm
    theta = np.arange(0, 360)*np.pi/180
    edges = np.argwhere(img[:, :])                                               #Extracting all edge coordinates
    for val in range(R):
        r = R_min+val
        #Creating a Circle Blueprint
        bprint = np.zeros((2*(r+1), 2*(r+1)))
        (m, n) = (r+1, r+1)                                                       #Finding out the center of the blueprint
        for angle in theta:
            x = int(np.round(r*np.cos(angle)))
            y = int(np.round(r*np.sin(angle)))
            bprint[m+x, n+y] = 1
        constant = np.argwhere(bprint).shape[0]
        for x, y in edges:                                                       #For each edge coordinates
            #Centering the blueprint circle over the edges
            #and updating the accumulator array
            X=[x-m+R_max, x+m+R_max]                                           #Computing the extreme X values
            Y=[y-n+R_max, y+n+R_max]                                            #Computing the extreme Y values
            A[r, X[0]:X[1], Y[0]:Y[1]] += bprint
        A[r][A[r] < threshold*constant/r] = 0

    for r, x, y in np.argwhere(A):
        temp = A[r-region:r+region, x-region:x+region, y-region:y+region]
        try:
            p, a, b = np.unravel_index(np.argmax(temp), temp.shape)
        except:
            continue
        B[r+(p-region), x+(a-region), y+(b-region)] = 1

    return B[:, R_max:-R_max, R_max:-R_max]


def houghLine(image):
    """ Basic Hough line transform that builds the accumulator array
    Input : image : edge image (canny)
    Output : accumulator : the accumulator of hough space
             thetas : values of theta (-90 : 90)
             rs : values of radius (-max distance : max distance)
    """
    # Get image dimensions
    # y for rows and x for columns
    Ny = image.shape[0]
    Nx = image.shape[1]

    # Max distance is diagonal one
    Maxdist = int(np.round(np.sqrt(Nx**2 + Ny ** 2)))
    # Theta in range from -90 to 90 degrees
    thetas = np.deg2rad(np.arange(-90, 90))
    # Range of radius
    rs = np.linspace(-Maxdist, Maxdist, 2*Maxdist)
    accumulator = np.zeros((2 * Maxdist, len(thetas)))
    for y in range(Ny):
        for x in range(Nx):
            # Check if it is an edge pixel
            #  NB: y -> rows , x -> columns
            if image[y, x] > 0:
                # Map edge pixel to hough space
                for k in range(len(thetas)):
                    # Calculate space parameter
                    r = x*np.cos(thetas[k]) + y * np.sin(thetas[k])
                    # Update the accumulator
                    # N.B: r has value -max to max
                    # map r to its idx 0 : 2*max
                    accumulator[int(r) + Maxdist, k] += 1
    return accumulator, thetas, rs


def plotLines(image, rho, theta):
    linewidth = image.shape[0]
    lineheight = image.shape[1]
    fig = plt.figure()
    plt.imshow(image)
    x = np.linspace(0, linewidth)
    cosine = np.cos(theta)
    sine = np.sin(theta)
    cotan = cosine/sine
    ratio = rho/sine
    for i in range(len(rho)):
        # if thete is not 0
        if theta[i]:
            plt.plot(x, (-x * cotan[i]) + ratio[i])
        # if theta is 0
        else:
            # Draw a vertical line at the corresponding rho value
            plt.axvline(rho[i])
    plt.xlim(0, linewidth)
    plt.ylim(lineheight, 0)
    fig.savefig('Lines.png')
    plt.show()


def detectLines(image, accumulator, thetas, rhos, threshold):
    # Determine the number of lines in the image
    # By creating a threashold and determining it by multiplicating
    # The threshold level and the maximum value of the accumulator
    detectedLines = np.where(accumulator >= (threshold * accumulator.max()))
    rho = rhos[detectedLines[0]]  # Get the indices of the rohs corresponding to the detected lines
    theta = thetas[detectedLines[1]]  # Get the indices of the thetas corresponding to the detected lines
    plotLines(image, rho, theta)


def active_contour(image, snake, alpha=0.01, beta=0.1,
                   w_line=0, w_edge=1, gamma=0.01,
                   bc='periodic', max_px_move=1.0,
                   max_iterations=2500, convergence=0.1):
    """Active contour model.
    Active contours by fitting snakes to features of images. Supports single
    and multichannel 2D images. Snakes can be periodic (for segmentation) or
    have fixed and/or free ends.
    The output snake has the same length as the input boundary.
    As the number of points is constant, make sure that the initial snake
    has enough points to capture the details of the final contour.
    Parameters
    ----------
    image : (N, M) or (N, M, 3) ndarray  ## For either Gray images or RGB
        Input image.
    snake : (N, 2) ndarray  ## A collection of points
        Initial snake coordinates. For periodic boundary conditions, endpoints
        must not be duplicated.
    alpha : float, optional
        Snake length shape parameter. Higher values makes snake contract
        faster.
    beta : float, optional
        Snake smoothness shape parameter. Higher values makes snake smoother.
    w_line : float, optional
        Controls attraction to brightness. Use negative values to attract toward
        dark regions.
    w_edge : float, optional
        Controls attraction to edges. Use negative values to repel snake from
        edges.
    gamma : float, optional
        Explicit time stepping parameter.
    bc : {'periodic', 'free', 'fixed'}, optional
        Boundary conditions for worm. 'periodic' attaches the two ends of the
        snake, 'fixed' holds the end-points in place, and 'free' allows free
        movement of the ends. 'fixed' and 'free' can be combined by parsing
        'fixed-free', 'free-fixed'. Parsing 'fixed-fixed' or 'free-free'
        yields same behaviour as 'fixed' and 'free', respectively.
    max_px_move : float, optional
        Maximum pixel distance to move per iteration.
    max_iterations : int, optional
        Maximum iterations to optimize snake shape.
    convergence: float, optional
        Convergence criteria.
    Returns
    -------
    snake : (N, 2) ndarray
        Optimised snake, same shape as input parameter.
    References
    ----------
    .. [1]  Kass, M.; Witkin, A.; Terzopoulos, D. "Snakes: Active contour
            models". International Journal of Computer Vision 1 (4): 321
            (1988). DOI:`10.1007/BF00133570`
    """
    max_iterations = int(max_iterations)
    if max_iterations <= 0:
        raise ValueError("max_iterations should be >0.")
    ##  the speed at which the errors go to zero.
    ## Typically the order of convergence measures the asymptotic behavior of convergence, often up to constants
    convergence_order = 10
    ## The valid values to be given for the type of the contour as stated in the documentation above
    valid_bcs = ['periodic', 'free', 'fixed', 'free-fixed',
                 'fixed-free', 'fixed-fixed', 'free-free']
    if bc not in valid_bcs:
        raise ValueError("Invalid boundary condition.\n" +
                         "Should be one of: " + ", ".join(valid_bcs) + '.')
    img = img_as_float(image)
    RGB = img.ndim == 3

    # Find edges using sobel:

    if w_edge != 0:
        edge = [sobel(img)]
        for i in range(1):
            edge[i][0, :] = edge[i][1, :]
            edge[i][-1, :] = edge[i][-2, :]
            edge[i][:, 0] = edge[i][:, 1]
            edge[i][:, -1] = edge[i][:, -2]
    else:
        edge = [0]

    # Superimpose intensity and edge images:
    img = w_line * img + w_edge * edge[0]

    # Interpolate for smoothness:
    # Can be used for both smoothing and interpolating data

    intp = RectBivariateSpline(np.arange(img.shape[1]),
                               np.arange(img.shape[0]),
                               img.T, kx=2, ky=2, s=0)

    x, y = snake[:, 0].astype(np.float), snake[:, 1].astype(np.float)
    n = len(x)
    xsave = np.empty((convergence_order, n))
    ysave = np.empty((convergence_order, n))

    # Build snake shape matrix for Euler equation
    # np.roll = circular shift
    # np.eye = Return a 2-D array with ones on the diagonal and zeros elsewhere
    a = np.roll(np.eye(n), -1, axis=0) + \
        np.roll(np.eye(n), -1, axis=1) - \
        2 * np.eye(n)  # second order derivative, central difference
    b = np.roll(np.eye(n), -2, axis=0) + \
        np.roll(np.eye(n), -2, axis=1) - \
        4 * np.roll(np.eye(n), -1, axis=0) - \
        4 * np.roll(np.eye(n), -1, axis=1) + \
        6 * np.eye(n)  # fourth order derivative, central difference
    A = -alpha * a + beta * b  # Internal energy

    # Impose boundary conditions different from periodic:
    sfixed = False
    if bc.startswith('fixed'):
        A[0, :] = 0
        A[1, :] = 0
        A[1, :3] = [1, -2, 1]
        sfixed = True
    efixed = False
    if bc.endswith('fixed'):
        A[-1, :] = 0
        A[-2, :] = 0
        A[-2, -3:] = [1, -2, 1]
        efixed = True
    sfree = False
    if bc.startswith('free'):
        A[0, :] = 0
        A[0, :3] = [1, -2, 1]
        A[1, :] = 0
        A[1, :4] = [-1, 3, -3, 1]
        sfree = True
    efree = False
    if bc.endswith('free'):
        A[-1, :] = 0
        A[-1, -3:] = [1, -2, 1]
        A[-2, :] = 0
        A[-2, -4:] = [-1, 3, -3, 1]
        efree = True

    # Only one inversion is needed for implicit spline energy minimization:
    inv = np.linalg.inv(A + gamma * np.eye(n))

    # Explicit time stepping for image energy minimization:
    for i in range(max_iterations):
        fx = intp(x, y, dx=1, grid=False)
        fy = intp(x, y, dy=1, grid=False)
        if sfixed:
            fx[0] = 0
            fy[0] = 0
        if efixed:
            fx[-1] = 0
            fy[-1] = 0
        if sfree:
            fx[0] *= 2
            fy[0] *= 2
        if efree:
            fx[-1] *= 2
            fy[-1] *= 2
        xn = inv @ (gamma * x + fx)
        yn = inv @ (gamma * y + fy)

        # Movements are capped to max_px_move per iteration:
        dx = max_px_move * np.tanh(xn - x)
        dy = max_px_move * np.tanh(yn - y)
        if sfixed:
            dx[0] = 0
            dy[0] = 0
        if efixed:
            dx[-1] = 0
            dy[-1] = 0
        x += dx
        y += dy

        # Convergence criteria needs to compare to a number of previous
        # configurations since oscillations can occur.
        j = i % (convergence_order + 1)
        if j < convergence_order:
            xsave[j, :] = x
            ysave[j, :] = y
        else:
            dist = np.min(np.max(np.abs(xsave - x[None, :]) +
                                 np.abs(ysave - y[None, :]), 1))
            if dist < convergence:
                break

    return np.array([x, y]).T


def box_filter(w):
    return np.ones((w, w)) / (w*w)


def apply_region_growing(img):
    rows,columns = np.shape(img)
    #Linear Scaling
    new_Gray_image = np.array(img.copy(),dtype = int)
    U_max = np.max(img)
    U_min = np.min(img)
    a = -1 * U_min
    b = 255 / (U_max - U_min)

    for i in range(rows):
        for j in range(columns):
            intensity = math.floor( b * ( img[i,j] + a))
            new_Gray_image[i,j] = intensity

    #User selects the intial seed point
    
    X_point = int(179)
    Y_point = int(86)
    
    seed_pixel = []
    seed_pixel.append(X_point)
    seed_pixel.append(Y_point)
    
    new_Segmented_image = np.zeros((rows+1,columns+1))
    new_Segmented_image[seed_pixel[0]][seed_pixel[1]] = 255.0
    
    region_points = []
    region_points.append([X_point,Y_point])
    
    
    def find_region():
    	count = 0
    	x = [-1,  0, 1, -1, 1,-1, 0, 1]
    	y = [-1, -1, -1, 0, 0, 1, 1, 1]
    	
    	while( len(region_points)>0):
    		
    		if count == 0:
    			point = region_points.pop(0)
    			i = point[0]
    			j = point[1]
    
    		val = new_Gray_image[i][j]
    		lt = val - 5
    		ht = val + 5
    		for k in range(8):	
    			if new_Segmented_image[i+x[k]][j+y[k]] !=1:
    				try:
    					if  new_Gray_image[i+x[k]][j+y[k]] > lt and new_Gray_image[i+x[k]][j+y[k]] < ht:
    						new_Segmented_image[i+x[k]][j+y[k]] = 1
    						p = [0,0]
    						p[0] = i+x[k]
    						p[1] = j+y[k]
    						if p not in region_points: 
    							if 0< p[0] < rows and 0 < p[1] < columns:
    								''' adding points to the region '''
    								region_points.append([i+x[k],j+y[k]])
    					else:
    						new_Segmented_image[i+x[k]][j+y[k]] = 0
    				except IndexError:     
                    			continue
    		point = region_points.pop(0)
    		i = point[0]
    		j = point[1]
    		count = count +1
    		
    
    find_region()
    return new_Segmented_image



global imageGRAY, imageRGB, Max, Min, imageSize, Clicked, clicker, Circular_Image, imagegray, imagergb

global imageGRAYH, imageRGBH, MaxH, MinH, imageSizeH, ClickedH

global imageGRAYHM, imageRGBHM, MaxHM, MinHM, imageSizeHM, ClickedHM

global Initial_Positions, Adjust_Snake, Center_Snake, Radius_Snake, x1, y1, ClickedS 

global ClickedSG, Adjust_K_Mean, selectionSG, x1SG, y1SG, x2SG, y2SG, firstPoint, secondPoint
global point1_blue, point1_green, point1_red, point1_colors, point2_blue, point2_green, point2_red, point2_colors

# Clicked is used to make sure that an image is loaded before choosing a filter
Clicked = 0
ClickedH = 0
ClickedHM = 0
clicker = 0
Initial_Positions = []
Center_Snake = []
Adjust_Snake = 0
Radius_Snake = 0
x1 = 0
y1 = 0
ClickedS = 0
ClickedSG = 0
Adjust_K_Mean = 0
x1SG = 0
y1SG = 0
x2SG = 0
y2SG = 0
firstPoint = []
secondPoint = []
point1_colors = []
point2_colors = []
selectionSG = "None"
point1_blue = 0
point1_green = 0
point1_red = 0
point2_blue = 0
point2_green = 0
point2_red = 0

class CV(QMainWindow):
    def __init__(self):
        super(CV, self).__init__()
        loadUi('mainwindow.ui', self)
        self.tabWidget.setCurrentIndex(0)
        self.pushButton_filters_load.clicked.connect(self.load_image)
        self.pushButton_histograms_load.clicked.connect(self.load_histogram)
        self.pushButton_Apply_Gaussian.clicked.connect(self.gaussian_kernel)
        self.pushButton_circles_load.clicked.connect(self.load_circle)
        self.pushButton_Detect_Circles.clicked.connect(self.circle_detection)
        self.pushButton_histograms_load_target.clicked.connect(self.load_histogram_matching)
        self.pushButton_lines_load.clicked.connect(self.line_detection)
        self.pushButton_corners_load.clicked.connect(self.corners)
        self.setSnake.clicked.connect(self.snake)
        self.resetSnake.clicked.connect(self.reset) 
        self.loadSnake.clicked.connect(self.load_Snake)
        self.pushButton_segmentation_load.clicked.connect(self.load_seg)
        self.apply_mean_shift.clicked.connect(self.mean_shift)
        self.setKMeanButton.clicked.connect(self.K_Mean)
        self.resetKMeanButton.clicked.connect(self.resetKMean)
        self.comboBox.activated.connect(self.filter_selection)
        self.comboBox_Segmentation.activated.connect(self.seg_selection)
        self.radioButton.clicked.connect(self.histogram_equalization)
        self.radioButton_2.clicked.connect(self.histogram_matching)

    def browser(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        image_source = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        return image_source

    def load_image(self):
        global imageGRAY, imageRGB, Max, Min, imageSize, Clicked
        imageSource = self.browser()
        # To make sure the application doesn't crash if no image is loaded
        if imageSource:
            imageRGB = cv2.imread(imageSource)
            imageGRAY, Max, Min, imageSize = image_parameters(imageRGB)
            name = '(' + str(imageSize[0]) + 'X' + str(imageSize[1]) + ')'
            self.label_13.setText(OpenedFile(imageSource))
            self.label_14.setText(name)
            self.label_15.setText(OpenedFile(imageSource))
            self.label_16.setText(name)
            self.graphicsView_2.setImage(imageGRAY.T)
            self.graphicsView_4.setImage(imageGRAY.T)
            self.comboBox.setCurrentIndex(0)
            Clicked = 1

    def filter_selection(self):
        global imageGRAY, imageSize, Clicked
        if Clicked == 0:
            QMessageBox.about(self, "Error!", "Please choose an image")
            return
        Laplacian_Kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        selection = self.comboBox.currentText()
        if selection == "Prewitt Filter":
            X_Prewitt_Kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            Y_Prewitt_Kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            X_Prewitt_Image = signal.convolve2d(imageGRAY, X_Prewitt_Kernel, boundary='symm', mode='same')
            Y_Prewitt_Image = signal.convolve2d(imageGRAY, Y_Prewitt_Kernel, boundary='symm', mode='same')
            Prewitt_Magnitude = np.sqrt(np.square(X_Prewitt_Image) + np.square(Y_Prewitt_Image))
            Prewitt_Direction = np.arctan(np.divide(Y_Prewitt_Image, X_Prewitt_Image))
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(Prewitt_Magnitude.T)

        elif selection == "Sobel Filter":
            X_Sobel_Kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            Y_Sobel_Kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            X_Sobel_Image = signal.convolve2d(imageGRAY, X_Sobel_Kernel, boundary='symm', mode='same')
            Y_Sobel_Image = signal.convolve2d(imageGRAY, Y_Sobel_Kernel, boundary='symm', mode='same')
            Sobel_Magnitude = np.sqrt(np.square(X_Sobel_Image) + np.square(Y_Sobel_Image))
            Sobel_Direction = np.arctan(np.divide(Y_Sobel_Image, X_Sobel_Image))
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(Sobel_Magnitude.T)

        elif selection == "Laplacian Filter":
            Laplacian_Image = signal.convolve2d(imageGRAY, Laplacian_Kernel, boundary='symm', mode='same')
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(Laplacian_Image.T)

        elif selection == "Box Filter":
            Box_Kernel = np.array([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]])
            Box_Image = signal.convolve2d(imageGRAY, Box_Kernel, boundary='symm', mode='same')
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(Box_Image.T)

        elif selection == "Gaussian Filter (3x3)":
            Gaussian_Kernel_3 = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
            Gaussian_Image_3 = signal.convolve2d(imageGRAY, Gaussian_Kernel_3, boundary='symm', mode='same')
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(Gaussian_Image_3.T)

        elif selection == "Gaussian Filter (5x5)":
            Gaussian_Kernel_5 = np.array([[1/273, 4/273, 7/273, 4/273, 1/273],
                                        [4/273, 16/273, 26/273, 16/273, 4/273],
                                        [7/273, 26/273, 41/273, 26/273, 7/273],
                                        [4/273, 16/273, 26/273, 16/273, 4/273],
                                        [1/273, 4/273, 7/273, 4/273, 1/273]])
            Gaussian_Image_5 = signal.convolve2d(imageGRAY, Gaussian_Kernel_5, boundary='symm', mode='same')
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(Gaussian_Image_5.T)

        elif selection == "Sharpening Filter":
            Sharpening_Kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            Sharpening_Image = signal.convolve2d(imageGRAY, Sharpening_Kernel, boundary='symm', mode='same')
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(np.abs(Sharpening_Image).T)

        elif selection == "LoG":
            Gaussian_Kernel_3 = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
            Gaussian_Image_3 = signal.convolve2d(imageGRAY, Gaussian_Kernel_3, boundary='symm', mode='same')
            LoG_Image = signal.convolve2d(Gaussian_Image_3, Laplacian_Kernel, boundary='symm', mode='same')
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(LoG_Image.T)

        elif selection == "DoG":
            Gaussian_Kernel_3 = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
            Gaussian_Image_3 = signal.convolve2d(imageGRAY, Gaussian_Kernel_3, boundary='symm', mode='same')
            Gaussian_Kernel_5 = np.array([[1 / 273, 4 / 273, 7 / 273, 4 / 273, 1 / 273],
                                          [4 / 273, 16 / 273, 26 / 273, 16 / 273, 4 / 273],
                                          [7 / 273, 26 / 273, 41 / 273, 26 / 273, 7 / 273],
                                          [4 / 273, 16 / 273, 26 / 273, 16 / 273, 4 / 273],
                                          [1 / 273, 4 / 273, 7 / 273, 4 / 273, 1 / 273]])
            Gaussian_Image_5 = signal.convolve2d(imageGRAY, Gaussian_Kernel_5, boundary='symm', mode='same')
            DoG_Image = Gaussian_Image_3 - Gaussian_Image_5
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(DoG_Image.T)

        elif selection == "Median Filter":
            Median_Variable = [(0, 0)] * 9
            Median_Image = imageGRAY.copy()
            for i in range(1, imageSize[0] - 1):
                for j in range(1, imageSize[1] - 1):
                    Median_Variable[0] = Median_Image[i - 1, j - 1]
                    Median_Variable[1] = Median_Image[i - 1, j]
                    Median_Variable[2] = Median_Image[i - 1, j + 1]
                    Median_Variable[3] = Median_Image[i, j - 1]
                    Median_Variable[4] = Median_Image[i, j]
                    Median_Variable[5] = Median_Image[i, j + 1]
                    Median_Variable[6] = Median_Image[i + 1, j - 1]
                    Median_Variable[7] = Median_Image[i + 1, j]
                    Median_Variable[8] = Median_Image[i + 1, j + 1]
                    Median_Variable.sort()
                    Median_Image[i, j] = Median_Variable[4]
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(Median_Image.T)

        elif selection == "High-pass Filter":
            Gray_image = imageGRAY.copy()
            (h, w) = imageSize
            # Resize the image to a minimum size 255*255
            if h <= 255 or w <= 255:
                Gray_image = resize(imageGRAY, (255, 255), anti_aliasing=True)
                (h, w) = Gray_image.shape
                print(h, w)
            half_h, half_w = int(h / 2), int(w / 2)
            F1 = fp.fft2((Gray_image).astype(float))
            F2 = fp.fftshift(F1)
            n = 10
            F2_High = F2.copy()
            # select all but the first 50x50 (low) frequencies
            F2_High[half_h - n:half_h + n + 1, half_w - n:half_w + n + 1] = 0
            HighPass_Image_Frequency = 20*np.log10(F2_High).astype(int)
            HighPass_Image = fp.ifft2(fp.ifftshift(F2_High)).real
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(HighPass_Image.T)

        elif selection == "Low-pass Filter":
            Gray_image = imageGRAY.copy()
            (h, w) = imageSize
            # Resize the image to a minimum size 255*255
            if h <= 255 or w <= 255:
                Gray_image = resize(imageGRAY, (255, 255), anti_aliasing=True)
                (h, w) = Gray_image.shape
                print(h, w)
            F1 = fp.fft2((Gray_image).astype(float))
            F2 = fp.fftshift(F1)
            half_h, half_w = int(h / 2), int(w / 2)
            n = 100  # Window Size
            F2_Low = F2.copy()
            F2_Low[0:n, 0:w] = 0
            F2_Low[h - n:h, 0:w] = 0
            F2_Low[0:h, 0:n] = 0
            F2_Low[0:h, w - n:w] = 0
            LowPass_Image_Frequency = 20 * np.log10(F2_Low).astype(int)
            LowPass_Image = fp.ifft2(fp.ifftshift(F2_Low)).real
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(LowPass_Image.T)

        elif selection == "Band-pass Filter":
            Gray_image = imageGRAY.copy()
            (h, w) = imageSize
            # Resize the image to a minimum size 255*255
            if h <= 255 or w <= 255:
                Gray_image = resize(imageGRAY, (255, 255), anti_aliasing=True)
                (h, w) = Gray_image.shape
                print(h, w)
            F1 = fp.fft2((Gray_image).astype(float))
            F2 = fp.fftshift(F1)
            half_h, half_w = int(h / 2), int(w / 2)
            n = 20  # Window Size
            F2_Band = F2.copy()
            F2_Band[half_h - n:half_h + n + 1, half_w - n:half_w + n + 1] = 0
            n = 100  # Window Size
            F2_Band[0:n, 0:w] = 0
            F2_Band[h - n:h, 0:w] = 0
            F2_Band[0:h, 0:n] = 0
            F2_Band[0:h, w - n:w] = 0
            BandPass_Image_Frequency = 20*np.log10(F2_Band).astype(int)
            BandPass_Image = fp.ifft2(fp.ifftshift(F2_Band)).real
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(BandPass_Image.T)
        elif selection == "FFT Magnitude":
            imageFFT = np.fft.fft2(imageGRAY)
            phaseFFT = np.angle(imageFFT)
            magnitudeFFT = np.absolute(imageFFT)
            magnitudeFFTShifted = np.fft.fftshift(np.log10(1 + magnitudeFFT))
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(magnitudeFFTShifted.T)

        else:
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(imageGRAY.T)

    def load_histogram(self):
        global imageGRAYH, MaxH, MinH, imageSizeH, ClickedH, histogram
        imageSourceH = self.browser()
        # To make sure the application doesn't crash if no image is loaded
        if imageSourceH:
            imageRGBH = cv2.imread(imageSourceH)
            imageGRAYH, MaxH, MinH, imageSizeH = image_parameters(imageRGBH)
            name = '(' + str(imageSizeH[0]) + 'X' + str(imageSizeH[1]) + ')'
            histogram = Histogram_Generator(imageGRAYH, MaxH, MinH, imageSizeH)
            self.label_15.setText(OpenedFile(imageSourceH))
            self.label_16.setText(name)
            self.graphicsView_4.setImage(imageGRAYH.T)
            self.graphicsView_6.clear()
            self.graphicsView_6.plot(histogram[0, :], histogram[1, :])
            self.radioButton.setChecked(False)
            self.radioButton_2.setChecked(False)
            ClickedH = 1

    def histogram_equalization(self):
        global imageGRAYH, MaxH, MinH, imageSizeH, ClickedH
        if ClickedH == 0:
            QMessageBox.about(self, "Error!", "Please choose an image")
            return
        Histogram,new_Gray_image = Histogram_Equalization(imageGRAYH, MaxH, MinH, imageSizeH)
        # Create the new equalized image
        for i in range(imageSizeH[0]):
            for j in range(imageSizeH[1]):
                intensity = int(new_Gray_image[i, j])
                new_Gray_image[i, j] = Histogram[3, intensity]
        self.graphicsView_5.setImage(new_Gray_image.T)
        self.graphicsView_7.clear()
        self.graphicsView_7.plot(Histogram[0, :], Histogram[3, :])

    def load_histogram_matching(self):
        global imageGRAYHM, imageRGBHM, MaxHM, MinHM, imageSizeHM, heightHM, widthHM, imageSourceHM, ClickedHM
        imageSourceHM = self.browser()
        # To make sure the application doesn't crash if no image is loaded
        if imageSourceHM:
            imageRGBHM = cv2.imread(imageSourceHM)
            imageGRAYHM, MaxHM, MinHM, imageSizeHM = image_parameters(imageRGBHM)
            histogramHM = Histogram_Generator(imageGRAYHM, MaxHM, MinHM, imageSizeHM)
            self.graphicsView_5.setImage(imageGRAYHM.T)
            self.graphicsView_7.clear()
            self.graphicsView_7.plot(histogramHM[0, :], histogramHM[1, :])
            ClickedHM = 1
            self.radioButton.setChecked(False)
            self.radioButton_2.setChecked(False)

    def histogram_matching(self):
        global imageGRAYH, imageSizeH, imageGRAYHM, ClickedH, ClickedHM
        if ClickedH == 0 or ClickedHM == 0:
            QMessageBox.about(self, "Error!", "Please choose an image")
            return
        Histogram_Source,new_Gray_image_Source = Histogram_Equalization(imageGRAYHM, MaxHM, MinHM, imageSizeHM)
        Histogram_Input, new_Gray_image_Input = Histogram_Equalization(imageGRAYH,MaxH, MinH, imageSizeH)

        for i in range(imageSizeH[0]):
            for j in range(imageSizeH[1]):
                intensity = int(new_Gray_image_Input[i, j])
                new_Gray_image_Input[i, j] = Histogram_Source[3, intensity]
        self.graphicsView_5.setImage(new_Gray_image_Input.T)

    def gaussian_kernel(self):
        global imageGRAY, Clicked
        if Clicked == 0:
            QMessageBox.about(self, "Error!", "Please choose an image")
            return
        if self.lineEdit.text().isdigit() and self.lineEdit_2.text().isdigit():
            GaussianK = int(self.lineEdit.text())
            GaussianSTD = int(self.lineEdit_2.text())
            Gaussian_Kernel = gaussian_kernel(GaussianK, GaussianSTD)
            Gaussian_Image = signal.convolve2d(imageGRAY, Gaussian_Kernel, boundary='symm', mode='same')
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(Gaussian_Image.T)
        else:
            QMessageBox.about(self, "Error!", "Enter valid numbers")
            return

    def load_circle(self):
        global clicker, Circular_Image, imagegray, imagergb
        imagesource = self.browser()
        # To make sure the application doesn't crash if no image is loaded
        if imagesource:
            imagergb = cv2.imread(imagesource)
            imagegray, Mx, Mn, imagesize = image_parameters(imagergb)
            clicker = 1
            Circular_Image = Canny_Edge_Detection(imagegray)
            name = '(' + str(imagesize[0]) + 'X' + str(imagesize[1]) + ')'
            self.label_21.setText(OpenedFile(imagesource))
            self.label_20.setText(name)
            self.graphicsView_8.setImage(imagegray.T)

    # References [10][11]

    def circle_detection(self):
        global clicker, Circular_Image, imagegray, imagergb
        if clicker == 0:
            QMessageBox.about(self, "Error!", "Please choose an image")
            return
        if self.lineEdit_3.text().isdigit() and self.lineEdit_4.text().isdigit() and self.lineEdit_5.text().isdigit():
            Threshold = int(self.lineEdit_3.text())
            R_max = int(self.lineEdit_4.text())
            R_min = int(self.lineEdit_5.text())
            result = detectCircles(Circular_Image, Threshold, 15, radius=[R_max, R_min])
            fig = plt.figure()
            plt.imshow(imagergb)
            circleCoordinates = np.argwhere(result)  # Extracting the circle information
            circle = []
            for r, x, y in circleCoordinates:
                circle.append(plt.Circle((y, x), r, color=(1, 0, 0), fill=False))
                fig.add_subplot(111).add_artist(circle[-1])
            plt.show()
            fig.savefig('BallCircle.png')
            img = "BallCircle.png"
            self.label_22.setPixmap(QPixmap(img).scaled(self.label_22.width(), self.label_22.height()))
            self.graphicsView_9.setImage(color.rgb2gray(cv2.imread('BallCircle.png')).T)
        else:
            QMessageBox.about(self, "Error!", "Enter valid numbers")
            return

    def line_detection(self):
        imageLocation = 'images2.png'
        imageR = cv2.imread(imageLocation)
        imageG = color.rgb2gray(imageR)
        imageEDGE = Canny_Edge_Detection(imageG)
        accumulator, thetas, rhos = houghLine(imageEDGE)
        name = '(' + str(imageG.shape[0]) + 'X' + str(imageG.shape[1]) + ')'
        detectLines(imageR, accumulator, thetas, rhos, 0.5)
        self.label_24.setText(name)
        self.label_23.setText('images2.png')
        self.graphicsView_11.setImage(accumulator.T)
        self.graphicsView_12.setImage(imageG.T)
        self.graphicsView_13.setImage(color.rgb2gray(cv2.imread('Lines.png')).T)
        self.label_25.setPixmap(QPixmap('Lines.png').scaled(self.label_25.width(), self.label_25.height()))

    def load_Snake(self):
        global Image, imageSource, imageSource, Radius_Snake, Center_Snake,  x1, y1, ClickedS, \
            Adjust_Snake, Initial_Positions
        imageSource = self.browser()
        # To make sure the application doesn't crash if no image is loaded
        if imageSource:
            Image = color.rgb2gray(cv2.imread(imageSource))
            self.image.resize(int(Image.shape[0]), int(Image.shape[1]))
            self.image.setPixmap(QPixmap(imageSource))
            self.setSnake.clicked.connect(self.snake)
            self.resetSnake.clicked.connect(self.reset)
            Adjust_Snake = 0
            ClickedS = 1
            Initial_Positions = []
            Center_Snake = []
            Radius_Snake = 0
            x1 = 0
            y1 = 0

    def snake(self):
        global Image, Initial_Positions, Center_Snake, Radius_Snake, ClickedS
        if ClickedS == 0:
            QMessageBox.about(self, "Error!", "Please choose an image")
            return
        if Adjust_Snake < 2:
            QMessageBox.about(self, "Error!", "Please Choose both the center and radius of the contour")
            return
        img = Image
        N = int(self.nEdit.text())
        a = float(self.aEdit.text())
        b = float(self.bEdit.text())
        alpha = float(self.alphaEdit.text())
        beta = float(self.betaEdit.text())
        gamma = float(self.gammaEdit.text())
        w_line = int(self.wlineEdit.text())
        w_edge = int(self.wedgeEdit.text())
        s = np.linspace(0, 2 * np.pi, N)
        x = np.asarray(Center_Snake)[0][0]
        y = np.asarray(Center_Snake)[0][1]
        circleX = x + (Radius_Snake/a) * np.cos(s)
        circleY = y + (Radius_Snake/b) * np.sin(s)
        init = np.array([circleX, circleY]).T
        snake = active_contour(gaussian(img, 3), init, alpha, beta, w_line, w_edge, gamma)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img)
        ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
        ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])
        #plt.show()
        plt.savefig('snake.png')
        k = 'snake.png'
        self.image.setPixmap(QPixmap(k).scaled(Image.shape[0], Image.shape[1]))

    def mousePressEvent(self, event: QMouseEvent):
        global Image, Initial_Positions, Adjust_Snake, Center_Snake, Radius_Snake, x1, y1, Adjust_K_Mean, \
            ClickedSG, selectionSG, imageSource, imageSRGB, Adjust_K_Mean, firstPoint, secondPoint, point1_blue, \
            point1_green, point1_red, point1_colors, point2_blue, point2_green, point2_red, point2_colors, \
            x1SG, y1SG, x2SG, y2SG
        if self.tabWidget.currentIndex() == 5:
            if ClickedS == 0:
                QMessageBox.about(self, "Error!", "Please choose an image")
                return
            if event.button() == Qt.LeftButton:
                #  point = self.image.mapFromGlobal(event.pos())
                point = self.image.mapFrom(self.centralwidget, event.pos())
                x = point.x()
                y = point.y()
                print(x)
                print(y)
                if x in range(Image.shape[0]) and y in range(Image.shape[1]):
                    if Adjust_Snake == 0:
                        Center_Snake.append((x, y))
                        x1 = x
                        y1 = y
                        Adjust_Snake += 1
                        centerText = "(" + str(x) + "," + str(y) + ")"
                        self.center.setText(centerText)
                    elif Adjust_Snake == 1:
                        Initial_Positions.append((x, y))
                        Radius_Snake = np.sqrt(np.square(x-x1)+np.square(y-y1))
                        Adjust_Snake += 1
                        self.radius.setText(str(Radius_Snake))
                    else:
                        QMessageBox.about(self, "Mistake!", "To Adjust Press Reset")
                        return
                    
        if self.tabWidget.currentIndex() == 6 and selectionSG == "K-Mean":
            if ClickedSG == 0:
                QMessageBox.about(self, "Error!", "Please choose an image")
                return
            if event.button() == Qt.LeftButton:
                #  point = self.image.mapFromGlobal(event.pos())
                point = self.image_segmentaion_ip.mapFrom(self.centralwidget, event.pos())
                x = point.x()
                y = point.y()
                print("x = " + str(x))
                print("y = " + str(y))
                if x in range(imageSRGB.shape[0]) and y in range(imageSRGB.shape[1]):
                    if Adjust_K_Mean == 0:
                        firstPoint.append((x, y))
                        x1SG = x
                        y1SG = y
                        Adjust_K_Mean += 1
                        firstPoint = "(" + str(x) + "," + str(y) + ")"
                        self.label_pt1_coordinates.setText(firstPoint)
                        point1_blue = imageSRGB[y1SG, x1SG, 0]  # Extract the Blue component
                        point1_green = imageSRGB[y1SG, x1SG, 1]  # Extract the Green component
                        point1_red = imageSRGB[y1SG, x1SG, 2]  # Extract the Red component
                        point1_colors = [int(point1_blue), int(point1_green), int(point1_red)]  # The RGB components of each pixel
                        self.label_R_pt1.setText(str(point1_red))
                        self.label_G_pt1.setText(str(point1_green))
                        self.label_B_pt1.setText(str(point1_blue))

                    elif Adjust_K_Mean == 1:
                        secondPoint.append((x, y))
                        x2SG = x
                        y2SG = y
                        Adjust_K_Mean += 1
                        secondPoint = "(" + str(x) + "," + str(y) + ")"
                        self.label_pt2_coordinates.setText(secondPoint)
                        point2_blue = imageSRGB[y2SG, x2SG, 0]  # Extract the Blue component
                        point2_green = imageSRGB[y2SG, x2SG, 1]  # Extract the Green component
                        point2_red = imageSRGB[y2SG, x2SG, 2]  # Extract the Red component
                        # The RGB components of each pixel
                        point2_colors = [int(point2_blue), int(point2_green), int(point2_red)]
                        self.label_R_pt2.setText(str(point2_red))
                        self.label_G_pt2.setText(str(point2_green))
                        self.label_B_pt2.setText(str(point2_blue))
                    else:
                        QMessageBox.about(self, "Mistake!", "To Adjust Press Reset")
                        return
                      
    def reset(self):
        global Initial_Positions, Adjust_Snake, Center_Snake, Radius_Snake, x1, y1, Image, imageSource
        if ClickedS == 0:
            QMessageBox.about(self, "Error!", "Please choose an image")
            return
        Initial_Positions = []
        Center_Snake = []
        Adjust_Snake = 0
        Radius_Snake = 0
        x1 = 0
        y1 = 0
        self.nEdit.clear()
        self.aEdit.clear()
        self.bEdit.clear()
        self.radius.clear()
        self.center.clear()
        self.alphaEdit.clear()
        self.betaEdit.clear()
        self.gammaEdit.clear()
        self.wedgeEdit.clear()
        self.wlineEdit.clear()
        self.nEdit.insert("400")
        self.aEdit.insert("1")
        self.bEdit.insert("1")
        self.alphaEdit.insert("0.01")
        self.betaEdit.insert("0.1")
        self.gammaEdit.insert("0.01")
        self.wedgeEdit.insert("1")
        self.wlineEdit.insert("0")
        img = imageSource
        Image = color.rgb2gray(cv2.imread(img))
        self.image.resize(int(Image.shape[0]), int(Image.shape[1]))
        self.image.setPixmap(QPixmap(img))

    def corners(self):
        imageSource = self.browser()
        if imageSource:
            images = cv2.imread(imageSource)
            images_gr = color.rgb2gray(images)
            images_smooth = signal.convolve2d(images_gr, gaussian_kernel(7, 1.0), 'same')

            sobel_h = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
            sobel_v = sobel_h.transpose()

            images_Ix = signal.convolve2d(images_smooth, sobel_h, 'same')
            images_Iy = signal.convolve2d(images_smooth, sobel_v, 'same')
            Sobel_Magnitude = np.sqrt(np.square(images_Ix) + np.square(images_Iy))
            fig = plt.figure()
            plt.imshow(Sobel_Magnitude, zorder=1)
            plt.axis('off')
            plt.savefig('Sobel.png', bbox_inches='tight')
            Sobel_image = 'Sobel.png'

            images_Ixx = np.multiply(images_Ix, images_Ix)
            images_Iyy = np.multiply(images_Iy, images_Iy)
            images_Ixy = np.multiply(images_Ix, images_Iy)

            images_Ixx_hat = signal.convolve2d(images_Ixx, gaussian_kernel(21, 1.0), 'same')
            images_Iyy_hat = signal.convolve2d(images_Iyy, gaussian_kernel(21, 1.0), 'same')
            images_Ixy_hat = signal.convolve2d(images_Ixy, gaussian_kernel(21, 1.0), 'same')

            K = 0.05

            images_detM = np.multiply(images_Ixx_hat, images_Iyy_hat) - np.multiply(images_Ixy_hat, images_Ixy_hat)
            images_trM = images_Ixx_hat + images_Iyy_hat
            images_R = images_detM - K * images_trM
            ratio = 0.2  # Tunable value. to keep adaptivity per image.
            images_corners = np.abs(images_R) > np.quantile(np.abs(images_R), 0.999)
            fig = plt.figure()
            plt.imshow(images, zorder=1)
            corners_pos = np.argwhere(images_corners)
            plt.scatter(corners_pos[:, 1], corners_pos[:, 0], zorder=2, c='r', marker='x')
            plt.axis('off')
            plt.savefig('corners.png',  bbox_inches='tight')
            saved_image = 'corners.png'
            fig = plt.figure()
            plt.scatter(corners_pos[:, 1], corners_pos[:, 0], zorder=2, c='r', marker='x')
            plt.axis('off')
            plt.savefig('cornerscorners.png', bbox_inches='tight')
            corner_image = 'cornerscorners.png'
            name = '(' + str(images.shape[0]) + 'X' + str(images.shape[1]) + ')'
            self.label_corners_name.setText(OpenedFile(imageSource))
            self.label_corners_size.setText(name)
            self.label_corners_input.setPixmap(QPixmap(imageSource).scaled(self.label_corners_input.width(),
                                                                           self.label_corners_input.height()))
            self.label_corners_corners_output.setPixmap(QPixmap(saved_image).
                                                        scaled(self.label_corners_corners_output.width(),
                                                               self.label_corners_corners_output.height()))
            self.label_corners_edge.setPixmap(QPixmap(Sobel_image).scaled(self.label_corners_edge.width(),
                                                                          self.label_corners_edge.height()))
            self.label_corners_corners.setPixmap(QPixmap(corner_image).scaled(self.label_corners_corners.width(),
                                                                              self.label_corners_corners.height())) 

    def load_seg(self):
        global ClickedSG, imageSource, imageSRGB, imageSGRAY, Original_Segmented_Image, Adjust_K_Mean, firstPoint, secondPoint
        imageSource = self.browser()
        Original_Segmented_Image = imageSource
        # To make sure the application doesn't crash if no image is loaded
        if imageSource:
            imageSRGB = cv2.imread(imageSource)
            imageSGRAY = color.rgb2gray(cv2.imread(imageSource))
            name = '(' + str(imageSGRAY.shape[0]) + 'X' + str(imageSGRAY.shape[1]) + ')'
            self.resetKMeanButton.clicked.connect(self.resetKMean)
            self.label_Seg_name.setText(OpenedFile(imageSource))
            self.label_Seg_size.setText(name)
            self.image_segmentaion_ip.setPixmap(QPixmap(imageSource).scaled(self.image_segmentaion_ip.width(),
                                                                            self.image_segmentaion_ip.height()))
            ClickedSG = 1
            Adjust_K_Mean = 0
            firstPoint = []
            secondPoint = []
            self.comboBox_Segmentation.setCurrentIndex(0)

    def seg_selection(self):
        global ClickedSG, imageSRGB, imageSGRAY, Original_Segmented_Image, selectionSG
        if ClickedSG == 0:
            QMessageBox.about(self, "Error!", "Please choose an image")
            return
        selectionSG = self.comboBox_Segmentation.currentText()
        if selectionSG == "Region-Growing":
            new_Segmented_image = apply_region_growing(imageSGRAY)
            fig = plt.figure()
            plt.imshow(new_Segmented_image)
            plt.axis("off")
            plt.savefig("RG.png")
            img = "RG.png"
            self.image_segmentaion_op.setPixmap(QPixmap(img).scaled(self.image_segmentaion_op.width(),
                                                                    self.image_segmentaion_op.height()))
        elif selectionSG == "Mean-Shift":
            Mode = int(self.lineEdit_6.text())
            new_Segmented_image = np.zeros(np.shape(imageSRGB), np.uint8)
            new_boundary_image = np.zeros(np.shape(imageSRGB), np.uint8)
            H = float(self.lineEdit_7.text())
            Hr = float(self.lineEdit_8.text())
            Hs = float(self.lineEdit_9.text())
            acceptable_value = float(self.lineEdit_10.text())

            def createFeatureMatrix(RGB_image):
                h, w, d = np.shape(RGB_image)
                F = []
                FAppend = F.append
                for row in range(0, h):
                    for col in range(0, w):
                        r, g, b = RGB_image[row][col]
                        FAppend([r, g, b, row, col])
                F = np.array(F)
                return F

            def getNeighbors(seed, matrix, mode):
                neighbors = []
                nAppend = neighbors.append
                sqrt = math.sqrt
                for i in range(0, len(matrix)):
                    cPixel = matrix[i]

                    if (mode == 1):
                        d = sqrt(sum((cPixel - seed) ** 2))
                        if (d < H):
                            nAppend(i)
                    else:
                        r = sqrt(sum((cPixel[:3] - seed[:3]) ** 2))
                        s = sqrt(sum((cPixel[3:5] - seed[3:5]) ** 2))

                        if (s < Hs and r < Hr):
                            nAppend(i)

                return neighbors

            def markPixels(neighbors, mean, matrix, cluster):
                for i in neighbors:
                    cPixel = matrix[i]
                    x = cPixel[3]  # X_location
                    y = cPixel[4]  # Y_Location
                    new_Segmented_image[x][y] = np.array(mean[:3], np.uint8)
                    new_boundary_image[x][y] = cluster
                return np.delete(matrix, neighbors, axis=0)

            def calculateMean(neighbors, matrix):
                neighbors = matrix[neighbors]
                r = neighbors[:, :1]
                g = neighbors[:, 1:2]
                b = neighbors[:, 2:3]
                x = neighbors[:, 3:4]
                y = neighbors[:, 4:5]
                mean = np.array([np.mean(r), np.mean(g), np.mean(b), np.mean(x), np.mean(y)])
                return mean

            def performMeanShift(RGB_image):

                clusters = 0
                F = createFeatureMatrix(RGB_image)

                while (len(F) > 0):

                    randomIndex = randint(0, len(F) - 1)
                    seed = F[randomIndex]

                    initialMean = seed

                    neighbors = getNeighbors(seed, F, Mode)

                    if (len(neighbors) == 1):
                        F = markPixels([randomIndex], initialMean, F, clusters)
                        clusters += 1
                        continue

                    # If we have multiple pixels, calculate the mean of all the columns
                    mean = calculateMean(neighbors, F)
                    # Calculate mean shift based on the initial mean
                    meanShift = abs(mean - initialMean)
                    # If the mean is below an acceptable value, then we are lucky to find a cluster
                    # Else, we generate a random seed again
                    if (np.mean(meanShift) <= acceptable_value):
                        F = markPixels(neighbors, mean, F, clusters)
                        clusters += 1

                return clusters

            clusters = performMeanShift(imageSRGB)
            fig = plt.figure()
            plt.imshow(new_Segmented_image)
            plt.savefig("MS.png")
            img = "MS.png"
            self.image_segmentaion_op.setPixmap(QPixmap(img).scaled(self.image_segmentaion_op.width(),
                                                                    self.image_segmentaion_op.height()))
        elif selectionSG == "K-Mean":
            self.resetKMean()
        else:
            self.image_segmentaion_op.setPixmap(QPixmap(Original_Segmented_Image).
                                                scaled(self.image_segmentaion_op.width(),
                                                       self.image_segmentaion_op.height()))

    def resetKMean(self):
        global Adjust_K_Mean, firstPoint, secondPoint, Original_Segmented_Image, imageSRGB,  x1SG, y1SG, x2SG, y2SG, ClickedSG, \
            point1_blue, point1_green, point1_red, point1_colors, point2_blue, point2_green, point2_red, point2_colors,selectionSG
           
        if ClickedSG == 0:
            QMessageBox.about(self, "Error!", "Please choose an image")
            return
        firstPoint = []
        secondPoint = []
        Adjust_K_Mean = 0
        point1_blue = 0
        point1_green = 0
        point1_red = 0
        point2_blue = 0
        point2_green = 0
        point2_red = 0
        x1SG = 0
        y1SG = 0
        x2SG = 0
        y2SG = 0
        point1_colors = []
        point2_colors = []
        self.label_pt1_coordinates.clear()
        self.label_pt2_coordinates.clear()
        self.label_R_pt1.clear()
        self.label_G_pt1.clear()
        self.label_B_pt1.clear()
        self.label_R_pt2.clear()
        self.label_G_pt2.clear()
        self.label_B_pt2.clear()

    def mean_shift(self):
        if ClickedSG == 0:
            QMessageBox.about(self, "Error!", "Please choose an image")
            return
        self.comboBox_Segmentation.setCurrentIndex(2)
        self.seg_selection()

    def K_Mean(self):
        global ClickedSG, imageSRGB, imageSGRAY, Original_Segmented_Image, selectionSG, Adjust_K_Mean, \
            ClickedSG, selectionSG, imageSource, imageSRGB, Adjust_K_Mean, firstPoint, secondPoint, point1_blue, \
            point1_green, point1_red, point1_colors, point2_blue, point2_green, point2_red, point2_colors, \
            x1SG, y1SG, x2SG, y2SG
        if ClickedSG == 0:
            QMessageBox.about(self, "Error!", "Please choose an image")
            return
        if selectionSG != "K-Mean":
            QMessageBox.about(self, "Error!", "Please choose K-Mean from menu")
            return
        if Adjust_K_Mean < 2:
            QMessageBox.about(self, "Error!", "Please detect 2 points on input image")
            return
        if Adjust_K_Mean == 2 and selectionSG == "K-Mean":
            imageGrayNew = imageSGRAY.copy()
            size = np.shape(imageSRGB)
            # User selects the initial seed point
            C1 = point1_colors
            C2 = point2_colors
            C1_Coordinates_I = []
            C1_Coordinates_J = []
            C2_Coordinates_I = []
            C2_Coordinates_J = []
            for x in range(10):
                # Calculate the Euclidean distance
                for i in range(size[0]):
                    for j in range(size[1]):
                        D1 = np.sqrt(
                            np.square(C1[0] - imageSRGB[i, j, 0]) + np.square(C1[1] - imageSRGB[i, j, 1]) + np.square(
                                C1[2] - imageSRGB[i, j, 2]))
                        D2 = np.sqrt(
                            np.square(C2[0] - imageSRGB[i, j, 0]) + np.square(C2[1] - imageSRGB[i, j, 1]) + np.square(
                                C2[2] - imageSRGB[i, j, 2]))
                        if D1 <= D2:
                            C1_Coordinates_I.append(i)
                            C1_Coordinates_J.append(j)
                        else:
                            C2_Coordinates_I.append(i)
                            C2_Coordinates_J.append(j)

                # Calculate the mean of the first cluster
                C1_Red_count = C1_Green_count = C1_Blue_count = 0
                C1_Length = len(C1_Coordinates_I)
                for i in range(C1_Length):
                    C1_Red_count = C1_Red_count + imageSRGB[C1_Coordinates_I[i], C1_Coordinates_J[i], 0]
                    C1_Green_count = C1_Green_count + imageSRGB[C1_Coordinates_I[i], C1_Coordinates_J[i], 1]
                    C1_Blue_count = C1_Blue_count + imageSRGB[C1_Coordinates_I[i], C1_Coordinates_J[i], 2]

                C1_Red_mean = C1_Red_count / C1_Length
                C1_Green_mean = C1_Green_count / C1_Length
                C1_Blue_mean = C1_Blue_count / C1_Length

                # Calculate the mean of the second cluster
                C2_Red_count = C2_Green_count = C2_Blue_count = 0
                C2_Length = len(C2_Coordinates_I)
                for i in range(C2_Length):
                    C2_Red_count = C2_Red_count + imageSRGB[C2_Coordinates_I[i], C2_Coordinates_J[i], 0]
                    C2_Green_count = C2_Green_count + imageSRGB[C2_Coordinates_I[i], C2_Coordinates_J[i], 1]
                    C2_Blue_count = C2_Blue_count + imageSRGB[C2_Coordinates_I[i], C2_Coordinates_J[i], 2]

                C2_Red_mean = C2_Red_count / C2_Length
                C2_Green_mean = C2_Green_count / C2_Length
                C2_Blue_mean = C2_Blue_count / C2_Length

                C1[0] = np.abs(C1[0] - C1_Red_mean)
                C1[1] = np.abs(C1[1] - C1_Green_mean)
                C1[2] = np.abs(C1[2] - C1_Blue_mean)

                C2[0] = np.abs(C2[0] - C2_Red_mean)
                C2[1] = np.abs(C2[1] - C2_Green_mean)
                C2[2] = np.abs(C2[2] - C2_Blue_mean)

                if x < 9:
                    C1_Coordinates_I = []
                    C1_Coordinates_J = []
                    C2_Coordinates_I = []
                    C2_Coordinates_J = []

            for i in range(C1_Length):
                imageGrayNew[C1_Coordinates_I[i], C1_Coordinates_J[i]] = 1

            for i in range(C2_Length):
                imageGrayNew[C2_Coordinates_I[i], C2_Coordinates_J[i]] = 0

            fig = plt.figure()
            plt.imshow(imageGrayNew)
            plt.axis("off")
            plt.savefig("KMean.png")
            img = "KMean.png"
            #plt.show()
            self.image_segmentaion_op.setPixmap(QPixmap(img).scaled(self.image_segmentaion_op.width(),
                                                                    self.image_segmentaion_op.height()))


if __name__ == "__main__":
    app = 0  # This is the solution As the Kernel died every time I restarted the consol
    app = QApplication(sys.argv)
    widget = CV()
    widget.show()
    sys.exit(app.exec_())
    

