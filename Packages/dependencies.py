#Loiy Ehsan 1830887
#10/21/2022

import numpy as np
from skimage import io
import skimage.feature as sf
import matplotlib.pyplot as plt
from matplotlib.image import imread as reading
from sympy import *
import cv2 as cv
import scipy.ndimage
from tkinter import *
from tkinter import messagebox
from scipy import ndimage

def image_capture():
    messagebox.showinfo(title="Capturing A Picture", message="Hit C on the Keyboard to capture an image")
    wcam = cv.VideoCapture(0)
    while True:
        isTrue, frame=wcam.read()
        cv.putText(frame, "I love Computer Vision", (200,400), cv.FONT_HERSHEY_SIMPLEX, 0.8, (249,255,35), 4)
        cv.imshow('Video',frame)
        if cv.waitKey(20) & 0xFF==ord('c'): #simply hit c on the keyboard to stop the capture and save the image
            cv.imwrite("loiy.png",frame)
            wcam.release()
            cv.destroyAllWindows()
            break
    cv.waitKey(0)


def plot_dft(crop_gray, magnitude_spectrum):
    plt.subplot(121),plt.imshow(crop_gray, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    
def show_image():
    face=io.imread("loiy.png")
    global fig, ax
    fig, ax = plt.subplots()
    plt.imshow(face)
    plt.show()
    
def select_point(i,j):
    ax = plt.subplots()
    points, = ax.plot(i, j, 'r*')
    x, y = points.get_data()
    print(x,' ',y)
    
def display_red():
    face=io.imread("loiy.png")
    Ired = np.zeros(face.shape)
    Ired[:,:,0] = face[:,:,0]
    plt.imshow(Ired/255)
    plt.show()

def display_green():
    face=io.imread("loiy.png")
    Igreen = np.zeros(face.shape)
    Igreen[:,:,1] = face[:,:,1]
    plt.imshow(Igreen/255)
    plt.show()
    
def display_blue():
    face=io.imread("loiy.png")
    Igreen = np.zeros(face.shape)
    Igreen[:,:,2] = face[:,:,2]
    plt.imshow(Igreen/255)
    plt.show()
    
def grey_scale():
    grey_face = cv.imread(r'loiy.png')
    gray2 = cv.cvtColor(grey_face, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray image', gray2)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def blur_image():
    
    img_original = cv.imread('loiy.png')
    img=cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)    
    kernel = np.ones((3,3),np.float32)/25
    filtered_image = cv.filter2D(img, -1, kernel)
    blur_Gaussian = cv.GaussianBlur(img,(3,3),0)
    cv.imshow('Original image',img)
    cv.imshow('2D Convolution', filtered_image)
    cv.imshow('Gaussian Blurring', blur_Gaussian)
    messagebox.showinfo(title="Sum to 1", message='''The main reason that why a blurring filter's values should generally sum to 1 if it only has positive values, is of constant images (Images that have constant values everywhere). We want them to remain constant even after blurring, and because blurred images are flat valued with any constant, they are left invariant by blurring mechanisms.''')
    
    cv.waitKey(0)
    cv.destroyAllWindows()

def find_edge():
    img = cv.imread('loiy.png',0)
    edges = cv.Canny(img,100,200)
    plt.imshow(edges,cmap = 'gray')
    plt.title('Edged Image')
    plt.show()
    
def phase_image():
    
    img=cv.imread('loiy.png')
    img = cv.Canny(img,100,200)
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    phase_spectrum = np.angle(dft_shift)
    ax1 = plt.subplot(1,2,1)
    ax1.imshow(img, cmap='gray')
    ax2 = plt.subplot(1,2,2)
    ax2.imshow(phase_spectrum, cmap='gray')
    plt.show()

def plot_gray(input_image):       
    plt.imshow(input_image, cmap='gray')
    plt.show()

  
def edge_mag():
    img=cv.imread("loiy.png")
    crop_gray = cv.cvtColor(img[100:400, 100:400], cv.COLOR_BGR2GRAY)
    img=cv.Canny(img,100,200)
    dft = cv.dft(np.float32(crop_gray),flags = cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    plot_dft(img, magnitude_spectrum)
    

def histogram():
    img = cv.imread('loiy.png')
    b, g, r = cv.split(img)
    plt.hist(b.ravel(), 256, [0, 256])
    plt.hist(g.ravel(), 256, [0, 256])
    plt.hist(r.ravel(), 256, [0, 256])
    plt.show()
    
def masking_image():
    img = cv.imread('loiy.png')
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[100:300, 100:400] = 255
    masked_img = cv.bitwise_and(img,img,mask = mask)
    cv.imshow("Masked Image", masked_img)

def image_rotate():
    img = cv.imread('loiy.png')
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[100:300, 100:400] = 255
    masked_img = cv.bitwise_and(img,img,mask = mask)
    final=ndimage.rotate(masked_img, 60)
    cv.imshow("Rotated Image",final)
    
def image_h_flip():
    img = cv.imread('loiy.png')
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[100:300, 100:400] = 255
    masked_img = cv.bitwise_and(img,img,mask = mask)
    final=cv.flip(masked_img, 1)
    cv.imshow("Horizontally Flipped",final)
    
def image_v_flip():
    img = cv.imread('loiy.png')
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[100:300, 100:400] = 255
    masked_img = cv.bitwise_and(img,img,mask = mask)
    final=cv.flip(masked_img, 0)
    cv.imshow("Horizontally Flipped",final)
    
    

def chunk():
    face=io.imread("loiy.png")
    fig, ax = plt.subplots()
    points, = ax.plot(342, 125, 'r*')
    x, y = points.get_data()
    cropped_image = face[x[0]:x[0]+10, y[0]:y[0]+10]
    plt.imshow(cropped_image)
    plt.show()