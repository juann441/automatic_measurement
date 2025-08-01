from utils.maths_utils import *
from tkinter import simpledialog
import os
import cv2

def load_images_from_folder(folder):
        images = []
        images2 = []
        
        # Demander à l'utilisateur la valeur du modulo
        mod = simpledialog.askinteger("Entree", "Entrez la valeur du modulo pour le nombre de photos voulues (1 pour prendre toutes les photos)")
        
        p = 0
        previous_img = None  # Initialiser previous_img comme None
        
        for filename in sorted(os.listdir(folder)):
            if p % mod == 0:
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    img_cropped = img[:285, :]
                    
                    if previous_img is not None:
                        # Calculer la MSE entre l'image actuelle et l'image precedente
                        error = mse(img_cropped.astype(float), previous_img.astype(float))
                        print(f"MSE between current and previous image: {error}")

                        # Si l'erreur est inferieure ou egale au seuil, ignorer cette image
                        if error <= 0: # On met à 0 pour ignorer
                            print(f"Image {filename} ignored due to low MSE")
                            continue
                    
                    
                    images.append(img_cropped)
                    
                    images2.append(img_cropped)
                    previous_img = img_cropped  # Mettre à jour previous_img

            p += 1

        return images, images2


import numpy as np

def coordsPointsRectangleFond(containerImgVraie, coords_rect_fond):
    """
    Permet à l'utilisateur de dessiner des rectangles sur une image via une fenêtre OpenCV.
    
    Args:
        containerImgVraie (list): Une liste d'images.
        coords_rect_fond (list): Une liste vide dans laquelle les coordonnées seront stockées.
    """
    counterRectFond = 0
    imgTestRECTFond = np.median([containerImgVraie[-1], containerImgVraie[0]], axis=0).astype(np.uint8)
    imgTestRECTFond = cv2.cvtColor(imgTestRECTFond, cv2.COLOR_GRAY2RGB)
    contDessinerRectFond = []
    rect_preview = None

    def click_event(event, x, y, flags, params):
        nonlocal counterRectFond
        nonlocal contDessinerRectFond
        nonlocal rect_preview
        
        if event == cv2.EVENT_LBUTTONDOWN:
            coords_rect_fond.append((y, x)) # On utilise la liste passée en argument
            contDessinerRectFond.append((y, x))
            counterRectFond += 1
            if counterRectFond % 2 == 0 and counterRectFond != 0:
                cv2.rectangle(imgTestRECTFond, (contDessinerRectFond[counterRectFond-2][1], contDessinerRectFond[counterRectFond-2][0]),
                              (contDessinerRectFond[counterRectFond-1][1], contDessinerRectFond[counterRectFond-1][0]), (200, 0, 255), 1)

            if counterRectFond <= 8:
                cv2.circle(imgTestRECTFond, (x, y), 5, (0, 0, 255), -1)
            rect_preview = None
        elif event == cv2.EVENT_MOUSEMOVE and counterRectFond % 2 == 1:
            rect_preview = (contDessinerRectFond[-1], (y, x))

    cv2.namedWindow('Coordonnees Points Rectangle')
    cv2.setMouseCallback('Coordonnees Points Rectangle', click_event)

    while True:
        display_img = imgTestRECTFond.copy()
        if rect_preview:
            pt1, pt2 = rect_preview
            cv2.rectangle(display_img, (pt1[1], pt1[0]), (pt2[1], pt2[0]), (200, 0, 255), 1)

        cv2.imshow('Coordonnees Points Rectangle', display_img)
        
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or counterRectFond == 9:
            break

    cv2.destroyAllWindows()
    print(coords_rect_fond)
    return coords_rect_fond

def coordsPointsRectangle(containerImgVraie):
    """
    Permet à l'utilisateur de dessiner des rectangles sur une image via une fenêtre OpenCV
    et retourne les coordonnées des points.
    
    Args:
        containerImgVraie (list): Une liste d'images.
        
    Returns:
        list: Une liste de tuples contenant les coordonnées (y, x) des points sélectionnés.
    """
    coords_rect = []
    counterRect = 0
    imgTestRECT = np.median([containerImgVraie[-1], containerImgVraie[0]], axis=0).astype(np.uint8)
    imgTestRECT = cv2.cvtColor(imgTestRECT, cv2.COLOR_GRAY2RGB)
    contDessinerRect = []
    rect_preview = None

    def click_event(event, x, y, flags, params):
        nonlocal counterRect, contDessinerRect, rect_preview, coords_rect
        if event == cv2.EVENT_LBUTTONDOWN:
            coords_rect.append((y, x))
            contDessinerRect.append((y, x))
            counterRect += 1
            if counterRect % 2 == 0 and counterRect != 0:
                cv2.rectangle(imgTestRECT, (contDessinerRect[counterRect-2][1], contDessinerRect[counterRect-2][0]),
                              (contDessinerRect[counterRect-1][1], contDessinerRect[counterRect-1][0]), (0, 255, 255), 1)

            if counterRect <= 8:
                cv2.circle(imgTestRECT, (x, y), 5, (0, 0, 255), -1)
            rect_preview = None
        elif event == cv2.EVENT_MOUSEMOVE and counterRect % 2 == 1:
            rect_preview = (contDessinerRect[-1], (y, x))

    cv2.namedWindow('Coordonnees Points Rectangle')
    cv2.setMouseCallback('Coordonnees Points Rectangle', click_event)

    while True:
        display_img = imgTestRECT.copy()
        if rect_preview:
            pt1, pt2 = rect_preview
            cv2.rectangle(display_img, (pt1[1], pt1[0]), (pt2[1], pt2[0]), (0, 255, 255), 1)

        cv2.imshow('Coordonnees Points Rectangle', display_img)
        
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or counterRect == 9:
            break

    cv2.destroyAllWindows()
    print(coords_rect)
    return coords_rect

def sectionPlusDeforme(derniereImgDeformationLPE, imgReelle):
    """
    Permet à l'utilisateur de sélectionner une zone sur une image, 
    puis calcule et affiche la section la plus déformée.

    Args:
        derniereImgDeformationLPE (list): La liste des images de déformation.
        imgReelle (list): La liste des images réelles.

    Returns:
        int: La colonne de la section la plus déformée.
    """
    zone_sectionmin = [] # La liste est créée ici, à l'intérieur de la fonction
    counterZ = 0
    derniereImgDeformationLPE = derniereImgDeformationLPE[-1]
    imgReelle = imgReelle[-1]
    imgTest = np.copy(derniereImgDeformationLPE)

    def click_event(event, x, y, flags, params, imgTest=imgTest):
        nonlocal counterZ, zone_sectionmin
        if event == cv2.EVENT_LBUTTONDOWN:
            zone_sectionmin.append((y, x))
            counterZ += 1
            if counterZ <= 2:
                cv2.circle(imgTest, (x, y), 5, (0, 0, 255), -1)

    cv2.namedWindow('Selectionner zone section min')
    cv2.setMouseCallback('Selectionner zone section min', click_event)

    while True:
        cv2.imshow('Selectionner zone section min', imgTest)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or counterZ == 2:
            break
    
    cv2.destroyAllWindows()
    
    gauche = zone_sectionmin[0][1]
    droite = zone_sectionmin[1][1]
    
    temp = np.sum(derniereImgDeformationLPE, axis=0)
    temp = np.where(temp[gauche:droite] == np.min(temp[gauche:droite]))[0][0].astype(int) + gauche
    
    tempImg = imgReelle
    tempImg[:, int(temp)] = 255
    
    while True:
        cv2.imshow('Resultat (Appuyer sur Echap)', tempImg)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    print(int(temp))
    return int(temp)

def suiviPoint(container, memoire, taille_pattern, tauxMaj, pointSectionMin, zoneminValide):
    """
    Permet à l'utilisateur de sélectionner un point et suit ses déplacements sur une série d'images.
    
    Args:
        container (list): La liste des images à traiter.
        memoire (int): Un drapeau pour le mode de mise à jour du pattern.
        taille_pattern (int): La taille du motif à suivre.
        tauxMaj (int): Le taux de mise à jour du pattern.
        pointSectionMin (int): La coordonnée de la section minimum.
        zoneminValide (int): Un drapeau pour valider la zone.
        
    Returns:
        list: Une liste de tuples contenant les coordonnées du point suivi.
    """
    counter1 = 0
    img = container[0]
    img2Color = np.copy(img)
    img2Color[:, pointSectionMin] = 255
    pointInit = []
    img2Color = cv2.cvtColor(img2Color, cv2.COLOR_GRAY2RGB)
    
    def click_event(event, x, y, flags, params):
        nonlocal counter1, pointInit
        if event == cv2.EVENT_LBUTTONDOWN:
            pointInit.append((y, x))
            counter1 += 1
            if counter1 == 1:
                cv2.circle(img2Color, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Coordonnees suivi point', img2Color)
            
    cv2.namedWindow('Coordonnees suivi point')
    cv2.setMouseCallback('Coordonnees suivi point', click_event)
    
    while True:
        cv2.imshow('Coordonnees suivi point', img2Color)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or counter1 == 1:
            break
            
    cv2.destroyAllWindows()
    if zoneminValide == 1:
        coordsG = [pointInit[0][0], pointSectionMin]
    else:
        coordsG = [pointInit[0][0], pointInit[0][1]]
    
    # ... (le reste de la fonction reste le même) ...
    # Le code de suivi des points suit ici
    carreeG = np.copy(img)[coordsG[0]-int(taille_pattern/2) : coordsG[0]+int(taille_pattern/2), 
                           coordsG[1]-int(taille_pattern/2) : coordsG[1]+int(taille_pattern/2)]
    carreeG = np.array(carreeG, dtype=np.uint8)

    corr1 = [coordsG[0], coordsG[1]]
    tempDeltaCorr1 = corr1
    nbMaxItr = 1000
    listCoordonneesPoint = []

    for i in range(len(container)):
        imgVisee = np.copy(container[i])

        maskCorr1 = np.zeros(container[i].shape)
        maskCorr1[corr1[0] - int(taille_pattern/2): corr1[0] + int(taille_pattern/2), 
                corr1[1] - int(taille_pattern/2): corr1[1] + int(taille_pattern/2)] = 1
        maskCorr1 = np.copy(container[i]) * maskCorr1
        maskCorr1 = np.array(maskCorr1, dtype=np.uint8)

        corr1_res = cv2.matchTemplate(np.copy(maskCorr1), carreeG, method=cv2.TM_CCOEFF_NORMED)
        corr1_res[:,0:corr1[1] - 45] = 0
        corr1_res[:,corr1[1] + 55:corr1_res.shape[1]] = 0
        corr1_res[0:corr1[0] - 65,:] = 0
        corr1_res[corr1[0] + 45:corr1_res.shape[0],:] = 0

        corr1Presque = np.where(corr1_res == np.max(corr1_res))
        corr1Presque = [corr1Presque[0] + int(taille_pattern/2), corr1Presque[1] + int(taille_pattern/2)]
        
        counterItr = 0
        while ((corr1Presque[1] - tempDeltaCorr1[1])**2 + (corr1Presque[0] - tempDeltaCorr1[0])**2)**0.5 > 15 and counterItr <= nbMaxItr :
            corr1_res[corr1Presque[0] + int(taille_pattern/2), corr1Presque[1] + int(taille_pattern/2)] = 0
            corr1Presque = np.where(corr1_res == np.max(corr1_res))
            corr1Presque = [corr1Presque[0][0] + int(taille_pattern/2), corr1Presque[1][0] + int(taille_pattern/2)]
            counterItr +=1
        if counterItr > nbMaxItr:
            corr1Presque = tempDeltaCorr1
        corr1 = corr1Presque
        corr1 = [int(np.mean(corr1[0])) , int(np.mean(corr1[1]))]
        tempDeltaCorr1 = corr1

        listCoordonneesPoint.append(corr1)

        if memoire == 1 and i % tauxMaj == 0:
            carreeG = imgVisee[corr1[0]-int(taille_pattern/2):corr1[0]+int(taille_pattern/2), 
                               corr1[1]-int(taille_pattern/2):corr1[1]+int(taille_pattern/2)]
            
    return list(reversed(listCoordonneesPoint))

def sommeSectionFinale(listLpe, coordonnesPointMin, taillemoy, containerImgVraie, taille_pattern):
    """
    Calcule la somme des sections finales, les déformations d'ingénieur et les déformations vraies.
    
    Args:
        listLpe (list): Liste d'images de déformation.
        coordonnesPointMin (list): Liste des coordonnées des points minimum.
        taillemoy (int): La taille de la zone pour la moyenne.
        containerImgVraie (list): Liste des images originales.
        taille_pattern (int): La taille du motif.

    Returns:
        tuple: Un tuple contenant les listes de déformations d'ingénieur, 
               des déformations vraies, et des images traitées.
    """
    containerSommeSection = []
    container_images_trait_suivi = []
    
    for i in range(len(listLpe)):
        temp2 = np.sum(listLpe[i], axis=0)
        temp = containerImgVraie[i] * listLpe[i]
        moy = 0
        for k in range(taillemoy):
            moy += temp2[coordonnesPointMin[i][1] - int(np.floor(taillemoy/2)) + k]
            temp[:, coordonnesPointMin[i][1] - int(np.floor(taillemoy/2)) + k] *= 0.5
        
        containerSommeSection.append(moy / taillemoy)
        
        temp[:, coordonnesPointMin[i][1]] = 255
        temp[coordonnesPointMin[i][0], :] = 255
        
        center_y, center_x = coordonnesPointMin[i]
        size = taille_pattern
        
        top_left = (max(0, center_y - size // 2), max(0, center_x - size // 2))
        bottom_right = (min(temp.shape[0] - 1, center_y + size // 2), min(temp.shape[1] - 1, center_x + size // 2))
        
        temp[top_left[0]:bottom_right[0]+1, top_left[1]] = 255
        temp[top_left[0]:bottom_right[0]+1, bottom_right[1]] = 255
        temp[top_left[0], top_left[1]:bottom_right[1]+1] = 255
        temp[bottom_right[0], top_left[1]:bottom_right[1]+1] = 255
        
        container_images_trait_suivi.append(temp)
    
    L0 = containerSommeSection[0]
    containerSommeSectionIngenieur = (L0 / np.array(containerSommeSection))**2 - 1
    containerDeformationVraie = np.log(containerSommeSectionIngenieur + 1)
    
    return containerSommeSectionIngenieur.tolist(), containerDeformationVraie.tolist(), container_images_trait_suivi

import matplotlib.pyplot as plt

def apply_morphological_gradient(image, tailleGrad):
    """
    Applique un gradient morphologique à une image.
    Retourne l'image traitée.
    """
    first_grad = cv2.morphologyEx(
        np.copy(image),
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tailleGrad, tailleGrad))
    )
    return np.floor((first_grad / np.max(first_grad)) * 255).astype(np.int32)

def apply_sobel_gradient(image, tailleGrad, dX, dY):
    """
    Applique un filtre de Sobel à une image et retourne le gradient.
    """
    temp_sobel_x = cv2.Sobel(np.copy(image), cv2.CV_64F, dY, 0, ksize=tailleGrad)
    temp_sobel_y = cv2.Sobel(np.copy(image), cv2.CV_64F, 0, dX, ksize=tailleGrad)

    # Note: Votre code avait une logique complexe pour la soustraction qui
    # pourrait être simplifiée. Je la garde pour conserver la logique originale.
    tempx1 = cv2.Sobel(np.copy(image), cv2.CV_64F, 9, 0, ksize=11)
    tempx2 = cv2.Sobel(np.copy(image), cv2.CV_64F, 0, 9, ksize=11)
    
    temp_sobel_x = temp_sobel_x - np.where(tempx1 / np.max(tempx1) > 0.5, 1, 0) * temp_sobel_x
    temp_sobel_y = temp_sobel_y - np.where(tempx2 / np.max(tempx2) > 0.5, 1, 0) * temp_sobel_y
    
    gradient = np.sqrt(temp_sobel_x**2 + temp_sobel_y**2).astype(np.int32)
    return np.floor((gradient / np.max(gradient)) * 255).astype(np.int32)

def display_gradient_images(first_grad, last_grad):
    """
    Affiche la première et la dernière image traitée avec matplotlib.
    """
    plt.figure()
    plt.subplot(211)
    plt.imshow(first_grad, cmap='gray')
    plt.title("Première image")
    plt.subplot(212)
    plt.imshow(last_grad, cmap='gray')
    plt.title("Dernière image")
    plt.show()