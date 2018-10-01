import numpy as np
import cv2 as cv
import pickle
import time
import os
import random

from codeword import CodeWord

class Codebook:
    E1 = 12
    E2 = 8
    ALPHA = 0.4
    BETA = 1.1

    MAX_T = 15

    SIZE_X = 576
    SIZE_Y = 768

    DIR_TRAIN = './BG/Time_13-19/'
    DIR_TEST = './BG/Time_13-19/'

    CODEBOOKS_FILENAME = 'codebook.pickle'

    @staticmethod
    def create_codebook(codebook, pixel, t):
        """
        Updates raw codebook for a single pixel
        """
        r, g, b = pixel
        i = np.sqrt(r**2 + g**2 + b**2)
        cw, idx = Codebook.find_codeword(codebook, pixel, i)
        if cw: 
            cw.v = Codebook.calc_v(cw, pixel)
            cw.i_min = min(i, cw.i_min)
            cw.i_max = max(i, cw.i_max)
            cw.f = cw.f + 1
            cw.lamb = max(cw.lamb, (t - cw.q))
            cw.q = t
            codebook[idx] = cw
        else:
            cw = CodeWord(pixel, i, i, 1, t - 1, t, t)
            codebook.append(cw)


    @staticmethod
    def find_codeword(codebook, xt, i):
        """
        Finds matching codeword on a codebook
        """
        for m in reversed(range(len(codebook))):
            cw = codebook[m]
            vm = cw.v
            i_max = cw.i_max
            i_min = cw.i_min

            cd = Codebook.colordist(xt, vm)
            bright = Codebook.brightness(i, i_min, i_max)
            if  (cd <= Codebook.E1 and bright):
                return (cw, m)
        return (None, None)


    @staticmethod
    def update_lambda(codebook, n):
        """
        Updates lambda after creating raw codebook
        """
        for i in range(len(codebook)):
            cw = codebook[i]
            cw.lamb = max(cw.lamb, (n - cw.q + cw.p - 1))
            codebook[i] = cw

    @staticmethod
    def calc_v(cw, xt):
        """
        Calculates new R,G,B values for an existing codeword
        """
        r,g,b = xt
        r_m, g_m, b_m = cw.v
        new_r = (cw.f * r_m + r) / (cw.f + 1)
        new_g = (cw.f * g_m + g) / (cw.f + 1)
        new_b = (cw.f * b_m + b) / (cw.f + 1)
        return np.array([new_r, new_g, new_b])
        
    @staticmethod
    def colordist(xt, vm):
        """
        Calculates color distance between two pixels
        """
        dot_product = xt[0]*vm[0] + xt[1]*vm[1] + xt[2]*vm[2]
        p = np.power(dot_product, 2)/np.power(np.linalg.norm(vm), 2)
        cd = np.sqrt(np.power(np.linalg.norm(xt), 2) - p)
        return cd

    @staticmethod
    def brightness(i, i_min, i_max):
        """
        Checks if brightness distance between two pixels is 
        near enought based on ALPHA and BETA
        """
        i_low = Codebook.ALPHA * i_max
        i_high = min(Codebook.BETA * i_max, i_min/Codebook.ALPHA)
        if i_low <= i <= i_high:
            return True
        else:
            return False

    @staticmethod
    def temporal_filter(codebook, t):
        """
        Filters a codebook based on lambda 
        """
        for i in range(len(codebook)):
            cw = codebook[i]
            if cw.lamb > t:
                codebook[i] = None
        codebook = [x for x in codebook if x is not None]
        return codebook

    @staticmethod
    def save_codebooks(codebooks):
        """
        Saves trained codebooks on a pickle file
        """
        with open(Codebook.CODEBOOKS_FILENAME, 'wb') as handle:
            pickle.dump(codebooks, handle, protocol=pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def is_foreground(x, codebook):
        """
        Check if a pixel is foreground or background based on codebook
        """
        r, g, b = x
        i = np.sqrt(r**2 + g**2 + b**2)
        for t in range(len(codebook)):
            cw = codebook[t]
            cd = Codebook.colordist(x, cw.v)
            bright = Codebook.brightness(i, cw.i_min, cw.i_max)
            #print("-----------------")
            #print(x, cw.v)
            #print(cd, bright)
            if cd < Codebook.E2 and bright:
                cw.v = Codebook.calc_v(cw, x)
                cw.i_min = min(i, cw.i_min)
                cw.i_max = max(i, cw.i_max)
                cw.f = cw.f + 1
                cw.lamb = max(cw.lamb, (t - cw.q))
                cw.q = t
                codebook[t] = cw
                return False
        return True

    @staticmethod
    def load_codebooks():
        """
        Loads codebooks from pickle file
        """
        codebooks = pickle.load( open(Codebook.CODEBOOKS_FILENAME, "rb" ) )
        return codebooks

    @staticmethod
    def test_codebooks(codebooks):
        """
        Test images based on codebooks
        """
        file_list = os.listdir(Codebook.DIR_TEST)
        random.shuffle(file_list)
        i,j = codebooks.shape

        for filename in file_list:
                img = cv.imread(Codebook.DIR_TEST + filename, 1).astype(np.float32)
                img_fb = np.copy(img)
                for x in range(i):
                    for k in range(j):
                        if (Codebook.is_foreground(img[x,k], codebooks[x,k])):
                            img_fb[x,k] = np.array([0])
                        else:
                            img_fb[x,k] = np.array([255])
                cv.imshow('frame',img_fb.astype(np.uint8))
                cv.waitKey(0)
                cv.destroyAllWindows()        


    @staticmethod
    def train_codebooks():
        """
        Creates and returns filtered codebooks
        """
        codebooks = np.empty([Codebook.SIZE_X, Codebook.SIZE_Y], dtype=object)
        for i in range(Codebook.SIZE_X):
            for j in range(Codebook.SIZE_Y):
                codebooks[i,j] = []

        file_list = os.listdir(Codebook.DIR_TRAIN)
        random.shuffle(file_list)

        t = 1
        for filename in file_list:
            img = cv.imread(Codebook.DIR_TRAIN + filename, 1).astype(np.float32)
            x,y,z = img.shape

            start = time.time()
            for i in range(x):
                for j in range(y):
                    Codebook.create_codebook(codebooks[i,j], img[i,j], t)
            end = time.time()

            # Prints tempo
            print('tempo imagem {0} : {1}'.format(t, str(end - start)))

            t += 1
            if t == Codebook.MAX_T:
                break
        
        for i in range(x):
            for j in range(y):
                Codebook.update_lambda(codebooks[i,j] , Codebook.MAX_T)

        codebooks_final = np.copy(codebooks)
        for i in range(x):
            for j in range(y):
                codebooks_final[i,j] = Codebook.temporal_filter(codebooks[i,j], Codebook.MAX_T / 2)

        return codebooks_final