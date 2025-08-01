import time 
import numpy as np
import cv2

def preparationImages(containerImages, flou):
    
    start_time = time.time()
    containerImgVraie = np.copy(containerImages)
    containerImgPrep = []
    if flou == 0:
        reconstruct = containerImages
        background = np.median(containerImgPrep, axis=0).astype(np.uint8)
        exec_time = time.time() - start_time
        return reconstruct, background, exec_time
    else:
        for i in range(len(containerImages)):
            img = containerImages[i]
            #img = cv2.fastNlMeansDenoising(img, flou, flou, 7, 21)
            img = cv2.bilateralFilter(img,9,flou,flou)
            containerImgPrep.append(img)

        background = np.median(containerImgPrep, axis=0).astype(np.uint8)
        reconstruct = []

        for i in range(len(containerImgPrep)):
            diff = cv2.absdiff(containerImgPrep[i], background)
            thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 20)
            filtre = cv2.morphologyEx(thresh, cv2.MORPH_ELLIPSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            filtre = cv2.bitwise_not(filtre)
            recons = cv2.inpaint(containerImgPrep[i], filtre, 1, cv2.INPAINT_TELEA)
            reconstruct.append(recons)

        exec_time = time.time() - start_time
        print("Preparation des images reussie")
        return reconstruct, background, exec_time

def preparationImagesUnique(img, flou):
    if flou == 0:
        return img
    else:
        img = cv2.bilateralFilter(img,9,flou,flou)
        return img