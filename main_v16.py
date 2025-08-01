import sys
print(sys.executable)
import numpy
print(numpy.__file__)
print(numpy.__version__)


import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import binary_fill_holes
import time
import csv

def correlationTranches(containerImageTrait, memoire2, tauxMaj, flagreverse, recupCoords,imageVerif, coordsGTranches,coordsDTranches,taillePattern,manuel):
    """
    containerImageTrait : Liste contenant les images de base. En fin d'iteration, un trait vertical est dessine sur l'image aux points du suivi

    memoire2 : Si egal à 1, le rechargement de la tranche est active, c'est à dire, la tranche du suivi est reprise sur l'ième image

    tauxMaj : Frequence de rechargement de la tranche

    flagreverse : Sert à indiquer si le suivi se fait de la premiere image à la dernière (egal à 1) ou de la dernière à la premiere ( si different de 1)

    recupCoords : Recuperation des points selectionnes dans le suivi par imagettes, (seulement visible en cas de reglage manuel)

    imageVerif : Images contenant les traits dessines sur la zone de striction

    coordsGTranches,coordsDTranches : coordonnees des tranches selectionnees dans la fenetre opencv

    taillePattern : taille de pattern/imagette , la taille sera de taillePattern * taillePattern

    manuel : Active (egal à 1) le placement manuel des tranches initiales, sinon, les tranches auront le meme point de depart que le suivi des patterns 

    """
    imgBase = containerImageTrait[0]
    if manuel == 1: # Selection des points de depart des tranches
        counter2 = 0 # Compteur de clics
        
        img3Color = cv2.cvtColor(imgBase, cv2.COLOR_BGR2RGB)

        
        # Dessiner les points de coordonnees initiales
        cv2.circle(img3Color, (recupCoords[0][1], recupCoords[0][0]), 2, (255, 0, 255), -1)
        cv2.circle(img3Color, (recupCoords[1][1], recupCoords[1][0]), 2, (255, 0, 255), -1)

        coords = [] # Initialisation de la liste contenant les coordonnees
        temp_img = img3Color.copy()

        def click_event(event, x, y, flags, params):
            nonlocal counter2, img3Color, temp_img

            if event == cv2.EVENT_LBUTTONDOWN:
                coords.append((y, x))
                counter2 += 1

                # Determine la couleur basee sur le compteur
                if counter2 % 2 == 0:
                    couleur = (0, 0, 255)  # Rouge
                else:
                    couleur = (0, 255, 0)  # Vert

                # Dessine le point et la ligne verticale
                cv2.circle(img3Color, (x, y), 2, couleur, -1)
                cv2.line(img3Color, (x, 0), (x, img3Color.shape[0]), couleur, 1)

            elif event == cv2.EVENT_MOUSEMOVE:
                # Cree une copie temporaire de l'image pour dessiner la ligne en deplacement de la souris
                img3Color = temp_img.copy()

                # Determine la couleur en fonction du compteur actuel
                if counter2 % 2 == 0:
                    couleur = (0, 0, 255)  # Rouge
                else:
                    couleur = (0, 255, 0)  # Vert

                # Dessine la ligne verticale temporaire à l'endroit où la souris se deplace
                cv2.line(img3Color, (x, 0), (x, img3Color.shape[0]), couleur, 1)

        cv2.namedWindow('Coordonnees Tranches')
        cv2.setMouseCallback('Coordonnees Tranches', click_event)

        while True:
            cv2.imshow('Coordonnees Tranches', img3Color)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or counter2 == 5:
                break

        cv2.destroyAllWindows()
        fenetreGauche = np.array(imgBase[:, coords[0][1]:coords[1][1]], dtype=np.float32) / np.max(imgBase) # On rogne l'image initiale aux points selectionnees, puis on normalise les valeurs des pixels
        fenetreDroite = np.array(imgBase[:, coords[2][1]:coords[3][1]], dtype=np.float32) / np.max(imgBase)
    else:
        fenetreGauche = np.array(imgBase[:, coordsGTranches[1] - int(taillePattern/2):coordsGTranches[1] + int(taillePattern/2)], dtype=np.float32) / np.max(imgBase)
        fenetreDroite = np.array(imgBase[:, coordsDTranches[1] - int(taillePattern/2):coordsDTranches[1] + int(taillePattern/2)], dtype=np.float32) / np.max(imgBase)



    containerDistance = []

    for k in range(len(containerImageTrait)): # On parcours le container des images de base
        img = containerImageTrait[k] # Image ciblee
        img = np.array(img, dtype=np.float32) / np.max(img)
        tempBis = imageVerif[k] # Images dans laquelle on dessine le trait vetical (sur ces images il y a le pointeur de la zone de striction)

        listvideGauche = [] # Liste vide contanant la valeur de la MSE dans chaque tranche analyse
        for i in range((img.shape[1] - fenetreGauche.shape[1])): # On bouge le scan selon l'axe X
            temp = img[:, 0 + i: fenetreGauche.shape[1] + i]
            listvideGauche.append(mse(temp, fenetreGauche))
        pointGauche = np.where(listvideGauche == np.min(listvideGauche))[0] # Recuperation du point où la mse vaut le min , ce qui donne le point suivi

        listvideDroite = []
        for i in range((img.shape[1] - fenetreDroite.shape[1])):
            temp = img[:, 0 + i: fenetreDroite.shape[1] + i]
            listvideDroite.append(mse(temp, fenetreDroite))
        pointDroite = np.where(listvideDroite == np.min(listvideDroite))[0]

        tempBis[:,pointGauche+ fenetreGauche.shape[1]] = 255 # Dessiner les droites veticales
        tempBis[:,pointDroite] = 255
        tempBis[:,pointGauche] = 255
        tempBis[:,pointDroite+fenetreDroite.shape[1]] = 255
        containerDistance.append((pointDroite - pointGauche)[0]) # Calcul du Delta L 
        if memoire2 == 1 and k % tauxMaj == 0: # Si le rechagrment de la tranches est active
            fenetreDroite = img[:, int(np.where(listvideDroite == np.min(listvideDroite))[0]): int(np.where(listvideDroite == np.min(listvideDroite))[0]) + fenetreDroite.shape[1]]
            fenetreGauche = img[:, int(np.where(listvideGauche == np.min(listvideGauche))[0]): int(np.where(listvideGauche == np.min(listvideGauche))[0]) + fenetreGauche.shape[1]]

    print(containerDistance, type(containerDistance))
    if flagreverse != 1: # Si les images sont inversees
        containerDistance = list(reversed(containerDistance))
        imageVerif = list(reversed(imageVerif))
    L0 = containerDistance[0]
    print(L0, type(L0))
    containerDistance = (containerDistance - L0) / L0

    return containerDistance,imageVerif


def correlation(memoire,taillePattern , tauxMaj, containerImageTrait, flagreverse,imageVerif):

    """
    Mêmes parametres que correlationTranches

    """
    counter1 = 0
    img = containerImageTrait[0]
    img2Color = cv2.cvtColor(np.copy(containerImageTrait[0]),cv2.COLOR_BGR2RGB)
    def click_event(event,x,y,flags,params):
        nonlocal counter1
        if event == cv2.EVENT_LBUTTONDOWN:
            coords.append((y,x))
            counter1 +=1
            if counter1 <=2:    
                cv2.circle(img2Color,(x,y),5,(0,0,255),-1)
            
    coords = []
    cv2.namedWindow('Coordonnees 2 points')
    cv2.setMouseCallback('Coordonnees 2 points',click_event)

    while True :
        cv2.imshow('Coordonnees 2 points',img2Color) # Fenetre OpenCV qui sert à selectionner les imagettes
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or counter1 ==3:
            break
    cv2.destroyAllWindows()

    recupCoords = np.copy(coords)

    coordsG = coords[0]# Coordonnees de l'imagette de gauche (y,x)
    coordsD = coords[1] # Coordonnees de l'imagette de droite (y,x)

    coordsGTranches = np.copy(coordsG) # Recuperation des points, c'est ce qui sert à mettre les tranches au même endroit que les imagettes
    coordsDTranches = np.copy(coordsD)

    ### Patterns Initiaux
    carreeG = np.copy(img)[coordsG[0]-int(taillePattern/2) : coordsG[0]+int(taillePattern/2) , coordsG[1]-int(taillePattern/2) : coordsG[1]+int(taillePattern/2)] # Imagettes de taillePattern ^2
    carreeD = np.copy(img)[coordsD[0]-int(taillePattern/2) : coordsD[0]+int(taillePattern/2), coordsD[1]-int(taillePattern/2): coordsD[1]+int(taillePattern/2)]
    carreeG = np.array(carreeG, dtype=np.uint8)
    carreeD = np.array(carreeD, dtype=np.uint8)
    ### Initialisation corr1 et corr2, les variables contenant les coordonnees en (y,x) des points de correlation, à t=0
    corr1 = [coordsG[0] , coordsG[1]]
    corr2 = [coordsD[0] , coordsD[1]]
    tempDeltaCorr1 = corr1 # Variables temporelles qui servent à comparer la position i avec la position en i-1
    tempDeltaCorr2 = corr2 

    nbMaxItr = 1000 # Nombre max d'iterations pour la recherche du point de correlation
    listDistance = []

    for i in range(len(containerImageTrait)):
        temp = imageVerif[i] # Image dans laquelle on dessine tous les trait des suivis (image qui apparait dans le menu principal)
        imgVisee = np.copy(containerImageTrait[i]) # I-ème image

        maskCorr1 = np.zeros(containerImageTrait[i].shape) # Initialisation 
        maskCorr1 [corr1[0] - int(taillePattern/2) : corr1[0] + int(taillePattern/2), corr1[1] - int(taillePattern/2): corr1[1] + int(taillePattern/2)] = 1
        maskCorr1 = np.copy(containerImageTrait[i]) * maskCorr1
        maskCorr1 = np.array(maskCorr1, dtype=np.uint8)

        # Carte de correlation 
        corr1_res = cv2.matchTemplate(np.copy(imgVisee), carreeG, method=cv2.TM_CCOEFF_NORMED)
        # Garder les points compris dans un carre de (45+55)*(65+45)
        corr1_res[:,0:corr1[1] - 45] = 0
        corr1_res[:,corr1[1] + 55:corr1_res.shape[1]] = 0
        corr1_res[0:corr1[0] - 65,:] = 0
        corr1_res[corr1[0] + 45:corr1_res.shape[0],:] = 0
        #Recherche du point avec la correlation max
        corr1Presque = np.where(corr1_res == np.max(corr1_res))
        corr1Presque = [corr1Presque[0] + int(taillePattern/2), corr1Presque[1] + int(taillePattern/2)]

        counterItr = 0
        # Verification de la distance entre le point precedent et le i-ème point, s'il est trop eloigne c'est peut etre une erreur, donc on le met à 0 et on recherche encore
        while ((corr1Presque[1] - tempDeltaCorr1[1])**2 + (corr1Presque[0] - tempDeltaCorr1[0])**2)**0.5 > 20 and counterItr <= nbMaxItr :
            corr1_res[corr1Presque[0] + int(taillePattern/2), corr1Presque[1] + int(taillePattern/2)] = 0
            corr1Presque = np.where(corr1_res == np.max(corr1_res))
            corr1Presque = [corr1Presque[0][0] + int(taillePattern/2), corr1Presque[1][0] + int(taillePattern/2)]
            counterItr +=1
        if counterItr > nbMaxItr: # Si on depasse le nombre max d'iterations
            corr1Presque = tempDeltaCorr1
        corr1 = corr1Presque
        corr1 = [int(np.mean(corr1[0])) , int(np.mean(corr1[1]))]
        tempDeltaCorr1 = corr1 # Mise à jour du point de correlation precedent

        maskCorr2 = np.zeros(containerImageTrait[i].shape)
        maskCorr2[corr2[0] - int(taillePattern/2): corr2[0] + int(taillePattern/2), corr2[1] - int(taillePattern/2): corr2[1] + int(taillePattern/2)] = 1
        maskCorr2 = np.copy(containerImageTrait[i]) * maskCorr2
        maskCorr2 = np.array(maskCorr2, dtype=np.uint8)

        corr2_res = cv2.matchTemplate(np.copy(imgVisee), carreeD, method=cv2.TM_CCOEFF_NORMED)
        corr2_res[:,0:corr2[1] - 45] = 0
        corr2_res[:,corr2[1] + 55:corr2_res.shape[1]] = 0
        corr2_res[0:corr2[0] - 65,:] = 0
        corr2_res[corr2[0] + 45:corr2_res.shape[0],:] = 0

        corr2Presque = np.where(corr2_res == np.max(corr2_res))
        corr2Presque = [corr2Presque[0] + int(taillePattern/2), corr2Presque[1] + int(taillePattern/2)]
        counterItr = 0
        while ((corr2Presque[1] - tempDeltaCorr2[1])**2 )**0.5 > 15 and counterItr <= nbMaxItr :
            corr2_res[corr2Presque[0] + int(taillePattern/2), corr2Presque[1] + int(taillePattern/2)] = 0
            corr2Presque = np.where(corr2_res == np.max(corr2_res))
            corr2Presque = [corr2Presque[0][0] + int(taillePattern/2), corr2Presque[1][0] + int(taillePattern/2)]
            counterItr +=1
        if counterItr > nbMaxItr:
            corr2Presque = tempDeltaCorr2
        corr2 = corr2Presque
        corr2= [int(np.mean(corr2[0])) , int(np.mean(corr2[1]))]
        tempDeltaCorr2 = corr2

        # Dessiner un carre centre sur les coordonnees
        center_y, center_x = corr1
        size = taillePattern  # Taille du côte du carre
        
        top_left = (center_y - size // 2, center_x - size // 2)
        bottom_right = (center_y + size // 2, center_x + size // 2)
        
        # Assurez-vous que les coordonnees sont dans les limites de l'image
        top_left = (max(0, top_left[0]), max(0, top_left[1]))
        bottom_right = (min(temp.shape[0] - 1, bottom_right[0]), min(temp.shape[1] - 1, bottom_right[1]))
        
        # Dessiner le carre en mettant les pixels à 255
        temp[top_left[0]:bottom_right[0]+1, top_left[1]] = 255
        temp[top_left[0]:bottom_right[0]+1, bottom_right[1]] = 255
        temp[top_left[0], top_left[1]:bottom_right[1]+1] = 255
        temp[bottom_right[0], top_left[1]:bottom_right[1]+1] = 255

        # Dessiner un carre centre sur les coordonnees
        center_y, center_x = corr2
        size = taillePattern  # Taille du côte du carre
        
        top_left = (center_y - size // 2, center_x - size // 2)
        bottom_right = (center_y + size // 2, center_x + size // 2)
        
        # Assurez-vous que les coordonnees sont dans les limites de l'image
        top_left = (max(0, top_left[0]), max(0, top_left[1]))
        bottom_right = (min(temp.shape[0] - 1, bottom_right[0]), min(temp.shape[1] - 1, bottom_right[1]))
        
        # Dessiner le carre en mettant les pixels à 255
        temp[top_left[0]:bottom_right[0]+1, top_left[1]] = 255
        temp[top_left[0]:bottom_right[0]+1, bottom_right[1]] = 255
        temp[top_left[0], top_left[1]:bottom_right[1]+1] = 255
        temp[bottom_right[0], top_left[1]:bottom_right[1]+1] = 255
        
        listDistance.append(corr2[1] - corr1[1])
        
        #### MISE A JOUR PATTERNS
        if memoire == 1 and i%tauxMaj==0:
            carreeG = imgVisee[corr1[0]-int(taillePattern/2):corr1[0]+int(taillePattern/2),corr1[1]-int(taillePattern/2):corr1[1]+int(taillePattern/2)]
            carreeD =  imgVisee[corr2[0]-int(taillePattern/2):corr2[0]+int(taillePattern/2),corr2[1]-int(taillePattern/2):corr2[1]+int(taillePattern/2)]
 
    ###########################################################
    
    listDistance = np.array(listDistance, dtype=np.int64)
    
    if flagreverse !=1:
        listDistance = list(reversed(listDistance)) 
        imageVerif = list(reversed(imageVerif))
    L0 = np.int64(listDistance[0])
    print(L0,type(L0))
    listDistance = (listDistance - L0)/L0
    
    return  (listDistance),recupCoords,imageVerif,coordsGTranches,coordsDTranches

def D(x):
    grad = np.concatenate( (x[1:] - x[:-1] , [0]))/2
    return grad
def Dt(x):
    div = -np.concatenate(( [x[0]], x[1:-1] - x[:-2] , [-x[-2]]))/2
    return div


def mse(I, ref):
    return np.sum((I-ref)**2)/I.size


def regression(listDistance,lam):
    dt = 0.01
    N = 1000000
    x = np.random.randn(len(listDistance))

    eps = 1 
    conv = 10e-8 #critere de convergence
    i=0

    while eps > conv and i < N:
        temp = x
        x = x - dt *2*(x - listDistance + lam*Dt(D(x)))
        eps = np.linalg.norm(temp - x)
  
        i +=1
    return x 



def lpe(markers,distImg):

    nbMax = np.max(distImg) + 1
    file_x = [[] for _ in range(256)]
    file_y = [[] for _ in range(256)]

    ligne, colonne = distImg.shape

    for i in range(ligne):
        for j in range(colonne):
            if markers[i,j] !=0:
                niveau_priorite = distImg[i,j]
                file_x[niveau_priorite].append(i)
                file_y[niveau_priorite].append(j)


    while any(file_x):
        for niveau in range(256):
            if file_x[niveau]:
                l = file_x[niveau].pop(0)
                c = file_y[niveau].pop(0)
                break
        voisins = [(l,max(c-1,0)),(l,min(c+1,colonne-1)),(l,min(c+1,colonne-1)),(max(l-1,0),c),(min(l+1,ligne-1),c)]

        for l_voisin, c_voisin in voisins:
            if markers[l_voisin, c_voisin] == 0:
                markers[l_voisin, c_voisin] = markers[l,c]
                niveau_priorite = distImg[l_voisin, c_voisin]
                file_x[niveau_priorite].append(l_voisin)
                file_y[niveau_priorite].append(c_voisin)
    

    nombre_zones = len(np.unique(markers)) - 1
    return markers, nombre_zones


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

def selectColors(containerImgVraie):
    counterColor = 0
    backgroundCouleur = cv2.cvtColor(np.copy(containerImgVraie[-1]), cv2.COLOR_BGR2RGB)

    def click_event(event, x, y, flags, params):
        nonlocal counterColor, backgroundCouleur
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Clic gauche : ajouter un point
        if event == cv2.EVENT_LBUTTONDOWN:
            if counterColor < 30:
                couleurs.append((backgroundCouleur[y, x])[2])
                counterColor += 1

                # Dessiner les points et mettre à jour le texte pour indiquer les clics restants
                if counterColor <= 5:
                    clics_restants = 5 - counterColor
                    texte_fond = f"Fond: {clics_restants} clics restants"
                    cv2.putText(backgroundCouleur, texte_fond, (10, 30), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.circle(backgroundCouleur, (x, y), 5, (0, 0, 255), -1)

                if counterColor == 5:
                    cv2.putText(backgroundCouleur, "Selectionnez la piece", (10, 30), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                if counterColor > 5 and counterColor <= 29:
                    markersBonus.append((y, x))
                    clics_restants_piece = 29 - counterColor
                    texte_piece = f"Piece: {clics_restants_piece} clics restants"
                    cv2.putText(backgroundCouleur, texte_piece, (10, 60), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.circle(backgroundCouleur, (x, y), 2, (0, 255, 0), -1)

        # Clic droit : annuler le dernier point
        elif event == cv2.EVENT_RBUTTONDOWN:
            if counterColor > 0:
                if counterColor <= 5:
                    couleurs.pop()
                else:
                    markersBonus.pop()

                counterColor -= 1

                # Redessiner l'image sans le dernier point annule
                backgroundCouleur = cv2.cvtColor(np.copy(containerImgVraie[-1]), cv2.COLOR_BGR2RGB)
                for i, color_value in enumerate(couleurs):
                    if i < 5:
                        # Recuperer les coordonnees pour redessiner le point supprime
                        y_coord, x_coord = np.where((backgroundCouleur[:, :, 2] == color_value))
                        cv2.circle(backgroundCouleur, (x_coord[0], y_coord[0]), 5, (0, 0, 255), -1)
                for point in markersBonus:
                    cv2.circle(backgroundCouleur, point[::-1], 2, (0, 255, 0), -1)

                # Mettre à jour les textes
                if counterColor <= 5:
                    clics_restants = 5 - counterColor
                    texte_fond = f"Fond: {clics_restants} clics restants"
                    cv2.putText(backgroundCouleur, texte_fond, (10, 30), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    clics_restants_piece = 29 - counterColor
                    texte_piece = f"Piece: {clics_restants_piece} clics restants"
                    cv2.putText(backgroundCouleur, texte_piece, (10, 60), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    couleurs = []
    markersBonus = []

    cv2.namedWindow('Selectionner couleurs')
    cv2.setMouseCallback('Selectionner couleurs', click_event)

    while True:
        cv2.imshow('Selectionner couleurs', backgroundCouleur)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or counterColor == 30:
            break

    cv2.destroyAllWindows()

    couleursFond = np.min(couleurs[:4])
    couleursPiece = np.mean(couleurs[5:8])
    return couleursFond, couleursPiece, markersBonus





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



class ImageViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mesure des deformations moyenne et locales version PRO ")
        self.label = tk.Label(self, text="Selectionnez le dossier contenant les images :")
        self.label.pack()
        self.select_button = tk.Button(self, text="Selectionner le dossier", command=self.select_folder)
        self.select_button.pack()
        self.image_frame = tk.Frame(self)
        self.image_frame.pack()
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()
        self.nav_frame = tk.Frame(self)
        self.nav_frame.pack()
        self.image_number_label = tk.Label(self, text="Numero")
        self.image_number_label.pack(side=tk.TOP)
        self.prev_button = tk.Button(self.nav_frame, text="Precedente", command=self.show_prev_image)
        self.play_button = tk.Button(self.nav_frame, text="Play", command=self.play_images)
        self.next_button = tk.Button(self.nav_frame, text="Suivante", command=self.show_next_image)
        self.next_ten_button = tk.Button(self.nav_frame, text="Suivante x10", command=self.show_next_ten_image)
        self.prev_button.pack(side=tk.LEFT)
        self.play_button.pack(side=tk.RIGHT)
        self.next_ten_button.pack(side=tk.RIGHT)
        self.next_button.pack(side=tk.RIGHT)
        self.plot_frame = tk.Frame(self)
        self.plot_frame.pack()
        self.process_button = None
        self.rect_button = None
        self.rect_fond_button = None
        self.color_button = None
        self.segment_button = None
        self.result_button = None
        self.tailleGrad = 0
        self.result_suivi_point_button = None
        self.original_images = []
        self.original_imagesCopie = []
        self.original_images_reversed = []
        self.reconstructed_images = []
        self.current_image_index = 0
        self.flou = 0
        self.zoneminValide = 0
        self.background = None
        self.coords_rect = []
        self.coords_rect_fond = []
        self.couleurs_fond = None
        self.couleurs_piece = None
        self.markers_bonus = []
        self.segmented_images = []
        self.zone_sectionmin = []
        self.coords_suivi = []
        self.resultsomme_tranches = []
        self.resultsomme_tranchesVraie = []
        self.result_distancedeltaL = []
        self.images_trait_suivi = []
        self.memoire = 0
        self.taillePattern = 0
        self.taillemoy = 0
        self.tauxMaj = 0
        self.typeDeGrad = 0
        self.resultsomme_tranchesCopie ,self.resultsomme_tranchesVraieCopie ,self.result_distancedeltaLCopie , self.result_distanceTranchesCopie = [],[],[],[]
        self.flagReverseTranches = 0
        self.flagReverseDeltaL = 0
        self.show_suivi = False   # Flag to show suivi images
        self.playing = False
        self.switch_to_segmented = False
        self.delay = 100
        self.coordsGTranches,self.coordsDTranches = 0,0
        self.manuel = 0
        self.dX = 0
        self.dY = 0
        self.first_grad = 0
        self.last_grad = 0
        self.tailleCorrect = 0
        self.flagSegmentationCorrection = 0
        self.flagCorrectSeg = 0

    def select_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:

            self.original_images,self.original_imagesCopie = self.load_images_from_folder(folder_selected)
            self.original_images_reversed = list(reversed(np.copy(self.original_imagesCopie)))
            self.show_image()
            self.process_button = tk.Button(self, text="Appliquer le pre-traitement les images", command=self.prepare_images)
            self.process_button.pack()
            self.status_label = tk.Label(self, text="Chargement des images termine")
            self.status_label.pack()

    def load_images_from_folder(self, folder):
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

    def show_image(self):
        if self.original_images:
            if self.switch_to_segmented and self.segmented_images:
                img = self.segmented_images[self.current_image_index] * self.original_imagesCopie[self.current_image_index]
                img = img.astype(np.uint8)
            else:
                img = self.original_images[self.current_image_index]
            img = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
            self.image_number_label.config(text=f"Image numero : {int(self.current_image_index) :n} ")
            self.update()

    def play_images(self):
        self.playing = not self.playing
        if self.playing:
            self.play_button.config(text="Pause")
            self.play_next_image()
        else:
            self.play_button.config(text="Play")

    def play_next_image(self):
        if self.playing:
            self.current_image_index = (self.current_image_index + 1) % len(self.original_images)
            self.show_image()
            self.after(self.delay, self.play_next_image)



    def show_next_image(self):
        self.playing = False  # Stop playing if the user manually changes the image
        self.play_button.config(text="Play")
        if self.original_images:
            self.current_image_index = (self.current_image_index + 1) % len(self.original_images)
            self.show_image()

    def show_prev_image(self):
        self.playing = False  # Stop playing if the user manually changes the image
        self.play_button.config(text="Play")
        if self.original_images:
            self.current_image_index = (self.current_image_index - 1) % len(self.original_images)
            self.show_image()

    def show_next_ten_image(self):
        self.playing = False  # Stop playing if the user manually changes the image
        self.play_button.config(text="Play")
        if self.original_images:
            self.current_image_index = (self.current_image_index + 10) % len(self.original_images)
            self.show_image()

    def prepare_images(self):
        self.status_label.config(text="Chargement en cours....")
        
        flagFlou = 0
        while flagFlou==0:
            self.flou = simpledialog.askinteger("Entree", "Entrez la valeur de la puissance du flou")
            imgFlouTemp = preparationImagesUnique(np.copy(self.original_images[0]), self.flou)
            imgFlouTempLast = preparationImagesUnique(np.copy(self.original_images[-1]), self.flou)
            plt.figure()
            plt.subplot(211)
            plt.imshow(imgFlouTemp, cmap='gray')
            plt.title("Première image")
            plt.subplot(212)
            plt.imshow(imgFlouTempLast, cmap='gray')
            plt.title("Dernière image")
            plt.show()
            flagFlou= simpledialog.askinteger("Entree", "Valider ce Flou (1 pour oui)")

        self.status_label.config(text="Application du flou en cours....")
        self.update()
        self.reconstructed_images, self.background, exec_time = preparationImages(self.original_images, self.flou)
        self.rect_button = tk.Button(self, text="Selectionner les Rectangles EPROUVETTE", command=self.select_rectangle_points)
        self.rect_button.pack()
        self.rect_fond_button = tk.Button(self, text="Selectionner les Rectangles FOND", command=self.select_rectangle_fond_points)
        self.rect_fond_button.pack()
        self.color_button = tk.Button(self, text="Selectionner les Couleurs", command=self.select_colors)
        self.color_button.pack()
        self.status_label.config(text=f"Temps de preparation des images : {exec_time:.2f} secondes")

    def select_rectangle_points(self):
        self.coords_rect = self.coordsPointsRectangle(self.original_images)

    def select_rectangle_fond_points(self):
        self.coords_rect_fond = self.coordsPointsRectangleFond(self.original_images)

    def select_colors(self):
        self.couleurs_fond, self.couleurs_piece, self.markers_bonus = selectColors(self.original_images)
        self.segment_button = tk.Button(self, text="Lancer la Segmentation", command=self.run_segmentation)
        self.segment_button.pack()


    def coordsPointsRectangleFond(self, containerImgVraie):
        counterRectFond = 0
        imgTestRECTFond = np.median([containerImgVraie[-1], containerImgVraie[0]], axis=0).astype(np.uint8)
        imgTestRECTFond = cv2.cvtColor(imgTestRECTFond, cv2.COLOR_GRAY2RGB)
        contDessinerRectFond = []
        rect_preview = None ####
        def click_event(event, x, y, flags, params):
            nonlocal counterRectFond
            nonlocal contDessinerRectFond
            nonlocal rect_preview ####
            if event == cv2.EVENT_LBUTTONDOWN:
                self.coords_rect_fond.append((y, x))
                contDessinerRectFond.append((y, x))
                counterRectFond += 1
                if counterRectFond%2 == 0 and counterRectFond !=0:
                    cv2.rectangle(imgTestRECTFond,(contDessinerRectFond[counterRectFond-2][1],contDessinerRectFond[counterRectFond-2][0] ) , (contDessinerRectFond[counterRectFond-1][1],contDessinerRectFond[counterRectFond-1][0] ),(200,0,255),1)

                if counterRectFond <= 8:
                    cv2.circle(imgTestRECTFond, (x, y), 5, (0, 0, 255), -1)
                # Reset the preview rectangle
                rect_preview = None
            elif event == cv2.EVENT_MOUSEMOVE and counterRectFond % 2 == 1:
                rect_preview = (contDessinerRectFond[-1], (y, x))

        cv2.namedWindow('Coordonnees Points Rectangle')
        cv2.setMouseCallback('Coordonnees Points Rectangle', click_event)

        while True:
            display_img = imgTestRECTFond.copy()

            # Draw the preview rectangle
            if rect_preview:
                pt1, pt2 = rect_preview
                cv2.rectangle(display_img, (pt1[1], pt1[0]), (pt2[1], pt2[0]), (200, 0, 255), 1)

            cv2.imshow('Coordonnees Points Rectangle', display_img)
            
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or counterRectFond == 9:
                break

        cv2.destroyAllWindows()
        print(self.coords_rect_fond)
        return self.coords_rect_fond

    def coordsPointsRectangle(self, containerImgVraie):
        counterRect = 0
        imgTestRECT = np.median([containerImgVraie[-1], containerImgVraie[0]], axis=0).astype(np.uint8)
        imgTestRECT = cv2.cvtColor(imgTestRECT, cv2.COLOR_GRAY2RGB)
        contDessinerRect = []
        rect_preview = None ####
        def click_event(event, x, y, flags, params):
            nonlocal counterRect
            nonlocal contDessinerRect
            nonlocal rect_preview ####
            if event == cv2.EVENT_LBUTTONDOWN:
                self.coords_rect.append((y, x))
                contDessinerRect.append((y, x))
                counterRect += 1
                if counterRect%2 == 0 and counterRect !=0:
                    cv2.rectangle(imgTestRECT,(contDessinerRect[counterRect-2][1],contDessinerRect[counterRect-2][0] ) , (contDessinerRect[counterRect-1][1],contDessinerRect[counterRect-1][0] ),(0,255,255),1)

                if counterRect <= 8:
                    cv2.circle(imgTestRECT, (x, y), 5, (0, 0, 255), -1)
                # Reset the preview rectangle
                rect_preview = None
            elif event == cv2.EVENT_MOUSEMOVE and counterRect % 2 == 1:
                rect_preview = (contDessinerRect[-1], (y, x))

        cv2.namedWindow('Coordonnees Points Rectangle')
        cv2.setMouseCallback('Coordonnees Points Rectangle', click_event)

        while True:
            display_img = imgTestRECT.copy()

            # Draw the preview rectangle
            if rect_preview:
                pt1, pt2 = rect_preview
                cv2.rectangle(display_img, (pt1[1], pt1[0]), (pt2[1], pt2[0]), (0, 255, 255), 1)

            cv2.imshow('Coordonnees Points Rectangle', display_img)
            
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or counterRect == 9:
                break

        cv2.destroyAllWindows()
        print(self.coords_rect)
        return self.coords_rect
    
    def sectionPlusDeforme(self, derniereImgDeformationLPE, imgReelle):
        counterZ = 0
        derniereImgDeformationLPE = derniereImgDeformationLPE[-1]
        imgReelle = imgReelle[-1]
        imgTest = np.copy(derniereImgDeformationLPE)

        def click_event(event, x, y, flags, params, imgTest=imgTest):
            nonlocal counterZ
            if event == cv2.EVENT_LBUTTONDOWN:
                self.zone_sectionmin.append((y,x))
                counterZ +=1
                if counterZ <=2:    
                    cv2.circle(imgTest,(x,y),5,(0,0,255),-1)
        cv2.namedWindow('Selectionner zone section min')
        cv2.setMouseCallback('Selectionner zone section min', click_event)

        while True:
            cv2.imshow('Selectionner zone section min', imgTest)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or counterZ == 3:
                break
        gauche = self.zone_sectionmin[0][1]
        droite = self.zone_sectionmin[1][1]
        cv2.destroyAllWindows()
        temp = np.sum(derniereImgDeformationLPE,axis=0)
        temp = np.where(temp[gauche:droite] == np.min(temp[gauche:droite]))[0][0].astype(int) + gauche
        #temp = np.where( np.sum(derniereImgDeformationLPE[:,gauche:droite],axis=0) == np.min(np.sum(derniereImgDeformationLPE[:,gauche:droite],axis=0)))[0][0] + gauche
        tempImg = imgReelle
        tempImg[:,int(temp)] = 255
        while True:
            cv2.imshow('Resultat (Appuyer sur Echap)', tempImg)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()
        print(int(temp))
        return int(temp)

    def suiviPoint(self, container, memoire, taillePattern, tauxMaj, pointSectionMin,zoneminValide):
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
        carreeG = np.copy(img)[coordsG[0]-int(taillePattern/2) : coordsG[0]+int(taillePattern/2), 
                                coordsG[1]-int(taillePattern/2) : coordsG[1]+int(taillePattern/2)]
        carreeG = np.array(carreeG, dtype=np.uint8)

        corr1 = [coordsG[0], coordsG[1]]
        tempDeltaCorr1 = corr1
        nbMaxItr = 1000
        listCoordonneesPoint = []
    
        for i in range(len(container)):
            imgVisee = np.copy(container[i])

            maskCorr1 = np.zeros(container[i].shape)
            maskCorr1[corr1[0] - int(taillePattern/2): corr1[0] + int(taillePattern/2), 
                    corr1[1] - int(taillePattern/2): corr1[1] + int(taillePattern/2)] = 1
            maskCorr1 = np.copy(container[i]) * maskCorr1
            maskCorr1 = np.array(maskCorr1, dtype=np.uint8)

            corr1_res = cv2.matchTemplate(np.copy(maskCorr1), carreeG, method=cv2.TM_CCOEFF_NORMED)
            corr1_res[:,0:corr1[1] - 45] = 0
            corr1_res[:,corr1[1] + 55:corr1_res.shape[1]] = 0
            corr1_res[0:corr1[0] - 65,:] = 0
            corr1_res[corr1[0] + 45:corr1_res.shape[0],:] = 0

            corr1Presque = np.where(corr1_res == np.max(corr1_res))
            corr1Presque = [corr1Presque[0] + int(taillePattern/2), corr1Presque[1] + int(taillePattern/2)]
            
            counterItr = 0
            while ((corr1Presque[1] - tempDeltaCorr1[1])**2 + (corr1Presque[0] - tempDeltaCorr1[0])**2)**0.5 > 15 and counterItr <= nbMaxItr :
                corr1_res[corr1Presque[0] + int(taillePattern/2), corr1Presque[1] + int(taillePattern/2)] = 0
                corr1Presque = np.where(corr1_res == np.max(corr1_res))
                corr1Presque = [corr1Presque[0][0] + int(taillePattern/2), corr1Presque[1][0] + int(taillePattern/2)]
                counterItr +=1
            if counterItr > nbMaxItr:
                corr1Presque = tempDeltaCorr1
            corr1 = corr1Presque
            corr1 = [int(np.mean(corr1[0])) , int(np.mean(corr1[1]))]
            tempDeltaCorr1 = corr1

            listCoordonneesPoint.append(corr1)

            if memoire == 1 and i % tauxMaj == 0:
                carreeG = imgVisee[corr1[0]-int(taillePattern/2):corr1[0]+int(taillePattern/2), 
                                corr1[1]-int(taillePattern/2):corr1[1]+int(taillePattern/2)]
        
        return list(reversed(listCoordonneesPoint))

    def sommeSectionFinale(self,listLpe,coordonnesPointMin,taillemoy,containerImgVraie ):
        containerSommeSection = []
        container_images_trait_suivi = []
        containerSommeSection = list(containerSommeSection)
        for i in range(len(listLpe)):
            temp2 = np.sum(listLpe[i],axis=0)
            temp = containerImgVraie[i]*listLpe[i]
            moy = 0
            for k in range(taillemoy):
                print(coordonnesPointMin[i][1])
                moy += temp2[coordonnesPointMin[i][1] -int(np.floor(taillemoy/2)) + k]
                temp[:,coordonnesPointMin[i][1] -int(np.floor(taillemoy/2)) + k] = temp[:,coordonnesPointMin[i][1] -int(np.floor(taillemoy/2)) + k]*0.5
            
            containerSommeSection.append(moy / taillemoy)

            temp[:,coordonnesPointMin[i][1]] = 255
            temp[coordonnesPointMin[i][0],:] = 255
                    # Dessiner un carre centre sur les coordonnees
            center_y, center_x = coordonnesPointMin[i]
            size = self.taillePattern  # Taille du côte du carre
            
            top_left = (center_y - size // 2, center_x - size // 2)
            bottom_right = (center_y + size // 2, center_x + size // 2)
            
            # Assurez-vous que les coordonnees sont dans les limites de l'image
            top_left = (max(0, top_left[0]), max(0, top_left[1]))
            bottom_right = (min(temp.shape[0] - 1, bottom_right[0]), min(temp.shape[1] - 1, bottom_right[1]))
            
            # Dessiner le carre en mettant les pixels à 255
            temp[top_left[0]:bottom_right[0]+1, top_left[1]] = 255
            temp[top_left[0]:bottom_right[0]+1, bottom_right[1]] = 255
            temp[top_left[0], top_left[1]:bottom_right[1]+1] = 255
            temp[bottom_right[0], top_left[1]:bottom_right[1]+1] = 255
            container_images_trait_suivi.append(temp)

        L0 = containerSommeSection[0]
        #containerSommeSection = -(containerSommeSection - L0)/L0
        ##containerSommeSectionIngenieur = (L0 - containerSommeSection)/L0
        containerSommeSectionIngenieur = (L0 / containerSommeSection)**2 -1
        containerDeformationVraie = np.log(containerSommeSectionIngenieur+1)

        return containerSommeSectionIngenieur,containerDeformationVraie, container_images_trait_suivi

    def run_segmentation(self):
        fond = simpledialog.askinteger("Entree", "Entrez la valeur du fond (1 pour clair, autre pour sombre):")
        self.typeDeGrad = simpledialog.askinteger("Entree", "Choisir le type de Gradient (1 pour Gradient Morphologique, autre pour Filtre Sobel)")
        flagGrad = 0
        self.first_grad = np.copy(self.reconstructed_images[0])
        self.last_grad = np.copy(self.reconstructed_images[-1])
        while flagGrad ==0:
            if self.typeDeGrad == 1:
                # Afficher la première image recuperee avec plt.imshow
                first_grad = np.copy(self.reconstructed_images[0])
                last_grad = np.copy(self.reconstructed_images[-1])
                if first_grad is not None:
                    self.tailleGrad = simpledialog.askinteger("Entree", "Choisir la taille de l'element structurant (de preference, chiffre impair)mettre une valeur enorme pour provoquer une erreur et revenir en arriere")
                    first_grad = cv2.morphologyEx(np.copy(first_grad), cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.tailleGrad, self.tailleGrad)))
                    first_grad = np.floor((first_grad / np.max(first_grad)) * 255).astype(np.int32)
                    last_grad = cv2.morphologyEx(np.copy(last_grad), cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.tailleGrad, self.tailleGrad)))
                    last_grad = np.floor((last_grad / np.max(last_grad)) * 255).astype(np.int32)
                    plt.figure()
                    plt.subplot(211)
                    plt.imshow(first_grad, cmap='gray')
                    plt.title("Première image")
                    plt.subplot(212)
                    plt.imshow(last_grad, cmap='gray')
                    plt.title("Dernière image")
                    plt.show()
                    flagGrad= simpledialog.askinteger("Entree", "Valider ce Gradient (1 pour oui) ")

            if self.typeDeGrad != 1:
                # Afficher la première image recuperee avec plt.imshow
                if self.first_grad is not None:
                    self.tailleGrad = simpledialog.askinteger("Entree", "Choisir la taille du filtre Sobel (1,3,5,7,11) mettre une valeur enorme pour provoquer une erreur et revenir en arriere")
                    self.dY = simpledialog.askinteger("Entree", "Choisir le DY")
                    self.dX = simpledialog.askinteger("Entree", "Choisir le dX")
                    temp_sobel_x = cv2.Sobel(np.copy(self.first_grad), cv2.CV_64F,self.dY,0,ksize=self.tailleGrad)
                    temp_sobel_y = cv2.Sobel(np.copy(self.first_grad), cv2.CV_64F,0,self.dX,ksize=self.tailleGrad)

                    tempx1 = cv2.Sobel(np.copy(self.first_grad), cv2.CV_64F,9,0,ksize=11) ###""
                    tempx2 = cv2.Sobel(np.copy(self.first_grad), cv2.CV_64F,0,9,ksize=11)####

                    temp_sobel_x = temp_sobel_x - np.where(tempx1/np.max(tempx1) >0.5,1,0) * temp_sobel_x####
                    temp_sobel_y = temp_sobel_y - np.where(tempx2/np.max(tempx2) >0.5,1,0) * temp_sobel_y#####
                    first_grad = np.sqrt(temp_sobel_x**2 + temp_sobel_y**2).astype(np.int32)
                    first_grad = np.floor((first_grad / np.max(first_grad)) * 255).astype(np.int32)

                    temp_sobel_x2 = cv2.Sobel(np.copy(self.last_grad), cv2.CV_64F,self.dY,0,ksize=self.tailleGrad)
                    temp_sobel_y2 = cv2.Sobel(np.copy(self.last_grad), cv2.CV_64F,0,self.dX,ksize=self.tailleGrad)
                    last_grad = np.sqrt(temp_sobel_x2**2 + temp_sobel_y2**2).astype(np.int32)
                    last_grad = np.floor((last_grad / np.max(last_grad)) * 255).astype(np.int32)
                    plt.figure()
                    plt.subplot(211)
                    plt.imshow(first_grad, cmap='gray')
                    plt.title("Première image")
                    plt.subplot(212)
                    plt.imshow(last_grad, cmap='gray')
                    plt.title("Dernière image")
                    plt.show()
                    flagGrad= simpledialog.askinteger("Entree", "Valider ce Gradient (1 pour oui)")

        self.segmented_images, exec_time = fonctionSegmentation(container = self.reconstructed_images, background = self.background, fond = fond,couleurFond= self.couleurs_fond, couleursPiece= self.couleurs_piece,coordsRect= self.coords_rect, markersBonus = self.markers_bonus,coordsRectFond= self.coords_rect_fond,typeDeGrad=self.typeDeGrad,tailleGrad = self.tailleGrad,dx=self.dX,dy=self.dY)
        messagebox.showinfo("Info", f"Segmentation terminee en {exec_time:.2f} secondes. Utilisez les boutons de navigation pour voir les resultats.")
        self.show_original = False
        self.show_image()
        self.result_button = tk.Button(self, text="Afficher le Resultat Segmentation", command=self.show_segmentation_result)
        self.result_button.pack()
        self.choseCorrectSeg = tk.Button(self, text="Corriger la segmentation", command=self.correctSegmentation)
        self.choseCorrectSeg .pack()
        self.chosezone_button = tk.Button(self, text="Localiser la zone de striction", command=self.select_zonemin)
        self.chosezone_button.pack()

    def show_segmentation_result(self):
        if self.segmented_images:
            self.switch_to_segmented = not self.switch_to_segmented
            self.show_image()

    def correctSegmentation(self):
        self.flagCorrectSeg = 0
        self.flagSegmentationCorrection = simpledialog.askinteger("Entree", " Corriger la segmentation ? (0 pour NON)")
        flagFondEprouv = simpledialog.askinteger("Entree", " Corriger l'arrière-plan ou l'eprouvette (1 pour eprouvette, autre pour fond)")
        if self.flagSegmentationCorrection  == 0:
            self.segmented_images = self.segmented_images
        else:
            while self.flagCorrectSeg == 0:
                self.tailleCorrect = simpledialog.askinteger("Entree", "Choisir taille correction de segmentation")
                tempSeg = np.copy(self.segmented_images[0])
                if flagFondEprouv==1:
                    tempSeg = cv2.morphologyEx(tempSeg,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.tailleCorrect,self.tailleCorrect)))
                else : 
                    tempSeg = np.max(tempSeg) - tempSeg
                    tempSeg = cv2.morphologyEx(tempSeg,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.tailleCorrect,self.tailleCorrect)))
                    tempSeg = np.max(tempSeg) - tempSeg

                tempSeg2 = np.copy(self.segmented_images[-1])
                if flagFondEprouv==1:
                    tempSeg2 = cv2.morphologyEx(tempSeg2,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.tailleCorrect,self.tailleCorrect)))
                else:
                    tempSeg2 = np.max(tempSeg2) - tempSeg2
                    tempSeg2 = cv2.morphologyEx(tempSeg2,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.tailleCorrect,self.tailleCorrect)))
                    tempSeg2 = np.max(tempSeg2) - tempSeg2

                plt.figure()
                plt.subplot(211)
                plt.imshow(tempSeg, cmap='gray')
                plt.title("Première image")
                plt.subplot(212)
                plt.imshow(tempSeg2, cmap='gray')
                plt.title("Dernière image")
                plt.show()
                self.flagCorrectSeg= simpledialog.askinteger("Entree", "Valider cette correction ?(1 pour oui)")
                if self.flagCorrectSeg== 1 :
                    if flagFondEprouv==1:
                        for m in range (len(self.segmented_images)):
                            self.segmented_images[m] = cv2.morphologyEx(self.segmented_images[m],cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.tailleCorrect,self.tailleCorrect)))
                    else : 
                        for m in range (len(self.segmented_images)):
                            self.segmented_images[m] = np.max(self.segmented_images[m]) - self.segmented_images[m]
                            self.segmented_images[m] = cv2.morphologyEx(self.segmented_images[m],cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.tailleCorrect,self.tailleCorrect)))
                            self.segmented_images[m] = np.max(self.segmented_images[m]) - self.segmented_images[m]
            
        
    def select_zonemin(self):
        self.zone_sectionmin = self.sectionPlusDeforme(self.segmented_images,self.original_imagesCopie)
        self.zoneminValide = simpledialog.askinteger("Entree", "Choisir cette section min (1 pour oui)")
        self.suivipoint_button = tk.Button(self, text="Lancer suivi de point", command=self.select_suivipoint)
        self.suivipoint_button.pack()

    def select_suivipoint(self):
        self.memoire = simpledialog.askinteger("Entree", "Activer memoire du pattern (1 pour oui):")
        self.tauxMaj = simpledialog.askinteger("Entree", "Taux de mise à jour du pattern")
        self.taillePattern = simpledialog.askinteger("Entree", "Taille du pattern")
        pointSectionMin = self.zone_sectionmin
        self.coords_suivi = self.suiviPoint(container= self.original_images_reversed, memoire=self.memoire, taillePattern=self.taillePattern, tauxMaj=self.tauxMaj,pointSectionMin=pointSectionMin,zoneminValide=self.zoneminValide)
        self.sommefinale_button = tk.Button(self, text="Lancer calcul des deformations", command=self.select_sommefinale)
        self.sommefinale_button.pack()

        
    def select_sommefinale(self):
        taillemoy = simpledialog.askinteger("Entree", "Entrer la taille de la moyenne (nombre impair)")
        self.taillemoy = taillemoy
        self.resultsomme_tranches,self.resultsomme_tranchesVraie, self.images_trait_suivi = self.sommeSectionFinale(listLpe=self.segmented_images,coordonnesPointMin = self.coords_suivi,taillemoy=taillemoy, containerImgVraie=self.original_imagesCopie)
        self.original_images = self.images_trait_suivi
        self.flagReverseDeltaL = simpledialog.askinteger("Entree", "(Delta L 2 points) Partir de la premiere images (1 oui, autre pour partir de la derniere)")
        if self.flagReverseDeltaL == 1:
            self.result_distancedeltaL,recupCoords,self.images_trait_suivi,self.coordsGTranches,self.coordsDTranches = correlation(memoire = self.memoire,taillePattern=self.taillePattern , tauxMaj=self.tauxMaj, containerImageTrait=self.original_imagesCopie, flagreverse=self.flagReverseDeltaL, imageVerif = self.images_trait_suivi)
        else:
            self.result_distancedeltaL,recupCoords, self.images_trait_suivi,self.coordsGTranches,self.coordsDTranches = correlation(memoire = self.memoire,taillePattern=self.taillePattern , tauxMaj=self.tauxMaj, containerImageTrait=self.original_images_reversed, flagreverse=self.flagReverseDeltaL,  imageVerif = list(reversed(self.images_trait_suivi)))
        
        self.manuel = simpledialog.askinteger("Entree", "(Tranches)Reglage manuel de la taille des tranches ? (1 pour oui)")
        if self.manuel == 1:
            self.flagReverseTranches = simpledialog.askinteger("Entree", "(Tranches) Partir de la premiere images (1 oui, autre pour partir de la derniere)")
        else : 
            self.flagReverseTranches=self.flagReverseDeltaL

        if self.flagReverseTranches==1:
            self.result_distanceTranches,self.images_trait_suivi = correlationTranches(memoire2 = self.memoire, tauxMaj=self.tauxMaj, containerImageTrait=self.original_imagesCopie,flagreverse=self.flagReverseTranches, recupCoords= recupCoords, imageVerif = self.images_trait_suivi,coordsGTranches=self.coordsGTranches,coordsDTranches=self.coordsDTranches,taillePattern=self.taillePattern,manuel=self.manuel )
        else:
            self.result_distanceTranches,self.images_trait_suivi = correlationTranches(memoire2 = self.memoire, tauxMaj=self.tauxMaj, containerImageTrait=self.original_images_reversed, flagreverse=self.flagReverseTranches, recupCoords= recupCoords, imageVerif = list(reversed(self.images_trait_suivi)),coordsGTranches=self.coordsGTranches,coordsDTranches=self.coordsDTranches,taillePattern=self.taillePattern,manuel=self.manuel )
        
        print(self.result_distanceTranches)
        self.plot_btn = tk.Button(self, text="Afficher le graphique", command=self.plot_results)
        self.plot_btn.pack()
        

    def toggle_images(self):
        self.show_original = not self.show_original
        self.show_image()
        
    def save_results(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            with open(save_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Index", "Deformation locale ING (programme)"])
                for index, result in enumerate(self.resultsomme_tranchesCopie):
                    writer.writerow([index, float(result)])
                
                writer.writerow(["Deformation axiale VRAIE (programme)"])
                for index,result in enumerate(self.resultsomme_tranchesVraieCopie):
                    writer.writerow([index,float(result)])

                writer.writerow(["Deformation moyenne(longitudinale) (suivi de deux points)"])
                for index,result in enumerate(self.result_distancedeltaLCopie):
                    writer.writerow([index,float(result)])


                writer.writerow(["Deformation moyenne(longitudinale) (tranches)"])
                for index,result in enumerate(self.result_distanceTranchesCopie):
                    writer.writerow([index,float(result)])
                    
            messagebox.showinfo("Info", "Resultats enregistres avec succès")

    
    def plot_results(self):
        fig, ax = plt.subplots()
        ax.plot(self.resultsomme_tranches, marker='.', linestyle='-', color='b')
        ax.plot(self.resultsomme_tranchesVraie, marker='.', linestyle='-', color='y')
        ax.plot(self.result_distancedeltaL, marker='.', linestyle='-', color='r')
        ax.plot(self.result_distanceTranches, marker='.', linestyle='-', color='g')
        
        ax.set_xlabel("Images")
        ax.legend(['Deformation locale(axiale) sur '+str(self.taillemoy)+' pixels','Deformation locale (axiale) vraie sur '+str(self.taillemoy)+' pixels', 'Deformation moyenne(longitudinale) (suivi de deux points)','Deformation moyenne(longitudinale) (Tranches)'])
        plt.show()
        lam = simpledialog.askinteger("Entree", "Entrer la valeur de lambda (regression lineaire)")
        self.resultsomme_tranchesCopie ,self.resultsomme_tranchesVraieCopie ,self.result_distancedeltaLCopie , self.result_distanceTranchesCopie = regression(self.resultsomme_tranches , lam),regression(self.resultsomme_tranchesVraie, lam), regression(self.result_distancedeltaL , lam) ,regression(self.result_distanceTranches , lam)
        fig, ax = plt.subplots()
        ax.plot(self.resultsomme_tranchesCopie, marker='.', linestyle='-', color='b')
        ax.plot(self.resultsomme_tranchesVraieCopie, marker='.', linestyle='-', color='y')
        ax.plot(self.result_distancedeltaLCopie, marker='.', linestyle='-', color='r')
        ax.plot(self.result_distanceTranchesCopie, marker='.', linestyle='-', color='g')
        ax.set_xlabel("Images")
        ax.legend(['Deformation locale(axiale) sur '+str(self.taillemoy)+' pixels','Deformation locale (axiale) vraie sur '+str(self.taillemoy)+' pixels', 'Deformation moyenne(longitudinale) (suivi de deux points)','Deformation moyenne(longitudinale) (Tranches)'])
        plt.show()
        self.save_btn = tk.Button(self, text="Enregistrer", command=self.save_results)
        self.save_btn.pack()

if __name__ == "__main__":
    app = ImageViewer()
    app.mainloop()