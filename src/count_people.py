from codebook import Codebook
import cv2 as cv
import numpy as np

FILE_NAME = "./examples/img.png"

img = cv.imread(FILE_NAME, cv.CV_8UC1)
count = Codebook.count_people(img)
print(count)