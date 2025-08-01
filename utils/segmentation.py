import time
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from utils.lpe import *

def fonctionSegmentation(container, background, fond, couleurFond, couleursPiece, coordsRect, markersBonus, coordsRectFond, typeDeGrad,tailleGrad,dy,dx):
    start_time = time.time()
    listLpe = []
    lstemp = []
  
    if fond != 1: # fond sombre
        for i in range(len(container)):
            if typeDeGrad == 1: #gradMorpho
                gradient = cv2.morphologyEx(np.copy(container[i]), cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tailleGrad, tailleGrad)))
                gradient = np.floor((gradient / np.max(gradient)) * 255).astype(np.int32)
            else : 
                sobel_x = cv2.Sobel(np.copy(container[i]), cv2.CV_64F,dy,0,ksize=tailleGrad)
                sobel_y = cv2.Sobel(np.copy(container[i]), cv2.CV_64F,0,dx,ksize=tailleGrad)
                gradient = np.sqrt(sobel_x**2 + sobel_y**2).astype(np.int32)
                gradient = np.floor((gradient / np.max(gradient)) * 255).astype(np.int32)
            
            piece = np.where(container[i] >= np.floor(np.max(background) * 0.49385), 1, 0).astype(float)
            piece = np.where(binary_fill_holes(piece) == True, 1, 0).astype(float) * 2
            piece = cv2.erode(piece, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 2))) 
            piece = np.where(piece != 0, 2, 0).astype(float)
            
            markers = fond + piece
            for e in range(int(len(coordsRectFond) / 2)):
                coordsG = coordsRectFond[2 * e] # y et x
                coordsD = coordsRectFond[2 * e + 1] # y et x
                markers[coordsG[0]:coordsD[0], coordsG[1]:coordsD[1]] = 1
            for l in range(len(markersBonus)):
                markers[markersBonus[l][0], markersBonus[l][1]] = 2
            for h in range(int(len(coordsRect) / 2)):
                coordsG = coordsRect[2 * h] # y et x
                coordsD = coordsRect[2 * h + 1] # y et x
                markers[coordsG[0]:coordsD[0], coordsG[1]:coordsD[1]] = 2
            seg, _ = lpe(markers, gradient)
            seg = np.where(seg !=1, 1, 0)
            seg = cv2.morphologyEx(seg.astype(float), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 11)))
            seg = cv2.morphologyEx(seg.astype(float), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 7)))

            listLpe.append(seg)

    else:
        for i in range(len(container)):

            if typeDeGrad == 1: #gradMorpho
                gradient = cv2.morphologyEx(np.copy(container[i]), cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tailleGrad, tailleGrad)))
                gradient = np.floor((gradient / np.max(gradient)) * 255).astype(np.int32)
            else : 
                sobel_x = cv2.Sobel(np.copy(container[i]), cv2.CV_64F,dy,0,ksize=tailleGrad)
                sobel_y = cv2.Sobel(np.copy(container[i]), cv2.CV_64F,0,dx,ksize=tailleGrad)
                gradient = np.sqrt(sobel_x**2 + sobel_y**2).astype(np.int32)
                gradient = np.floor((gradient / np.max(gradient)) * 255).astype(np.int32)

            piece = np.where(container[i] <= np.floor(couleursPiece), 1, 0).astype(float)
            piece = np.where( piece != 0, 2, 0).astype(float)

            markers =piece
            for e in range(int(len(coordsRectFond) / 2)):
                coordsGF = coordsRectFond[2 * e] # y et x
                coordsDF = coordsRectFond[2 * e + 1] # y et x
                markers[coordsGF[0]:coordsDF[0], coordsGF[1]:coordsDF[1]] = 1

            for l in range(len(markersBonus)):
                markers[markersBonus[l][0], markersBonus[l][1]] = 2

            for h in range(int(len(coordsRect) / 2)):
                coordsG = coordsRect[2 * h] # y et x
                coordsD = coordsRect[2 * h + 1] # y et x
                markers[coordsG[0]:coordsD[0], coordsG[1]:coordsD[1]] = 2
            seg, _ = lpe(np.copy(markers), np.copy(gradient))
            seg = np.where(seg == 2, 1,0).astype(float)
            seg = np.where(binary_fill_holes(seg) == True,1,0).astype(float)

            listLpe.append(seg)

    exec_time = time.time() - start_time
    return listLpe, exec_time
