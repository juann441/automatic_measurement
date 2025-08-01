import cv2
import numpy as np
from utils.maths_utils import *

def correlationTranches(containerImageTrait, memoire2, tauxMaj, flagreverse, recupCoords,imageVerif, coordsGTranches,coordsDTranches,taille_pattern,manuel):
    """
    containerImageTrait : Liste contenant les images de base. En fin d'iteration, un trait vertical est dessine sur l'image aux points du suivi

    memoire2 : Si egal à 1, le rechargement de la tranche est active, c'est à dire, la tranche du suivi est reprise sur l'ième image

    tauxMaj : Frequence de rechargement de la tranche

    flagreverse : Sert à indiquer si le suivi se fait de la premiere image à la dernière (egal à 1) ou de la dernière à la premiere ( si different de 1)

    recupCoords : Recuperation des points selectionnes dans le suivi par imagettes, (seulement visible en cas de reglage manuel)

    imageVerif : Images contenant les traits dessines sur la zone de striction

    coordsGTranches,coordsDTranches : coordonnees des tranches selectionnees dans la fenetre opencv

    taille_pattern : taille de pattern/imagette , la taille sera de taille_pattern * taille_pattern

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
        fenetreGauche = np.array(imgBase[:, coordsGTranches[1] - int(taille_pattern/2):coordsGTranches[1] + int(taille_pattern/2)], dtype=np.float32) / np.max(imgBase)
        fenetreDroite = np.array(imgBase[:, coordsDTranches[1] - int(taille_pattern/2):coordsDTranches[1] + int(taille_pattern/2)], dtype=np.float32) / np.max(imgBase)



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


def correlation(memoire,taille_pattern , tauxMaj, containerImageTrait, flagreverse,imageVerif):

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
    carreeG = np.copy(img)[coordsG[0]-int(taille_pattern/2) : coordsG[0]+int(taille_pattern/2) , coordsG[1]-int(taille_pattern/2) : coordsG[1]+int(taille_pattern/2)] # Imagettes de taille_pattern ^2
    carreeD = np.copy(img)[coordsD[0]-int(taille_pattern/2) : coordsD[0]+int(taille_pattern/2), coordsD[1]-int(taille_pattern/2): coordsD[1]+int(taille_pattern/2)]
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
        maskCorr1 [corr1[0] - int(taille_pattern/2) : corr1[0] + int(taille_pattern/2), corr1[1] - int(taille_pattern/2): corr1[1] + int(taille_pattern/2)] = 1
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
        corr1Presque = [corr1Presque[0] + int(taille_pattern/2), corr1Presque[1] + int(taille_pattern/2)]

        counterItr = 0
        # Verification de la distance entre le point precedent et le i-ème point, s'il est trop eloigne c'est peut etre une erreur, donc on le met à 0 et on recherche encore
        while ((corr1Presque[1] - tempDeltaCorr1[1])**2 + (corr1Presque[0] - tempDeltaCorr1[0])**2)**0.5 > 20 and counterItr <= nbMaxItr :
            corr1_res[corr1Presque[0] + int(taille_pattern/2), corr1Presque[1] + int(taille_pattern/2)] = 0
            corr1Presque = np.where(corr1_res == np.max(corr1_res))
            corr1Presque = [corr1Presque[0][0] + int(taille_pattern/2), corr1Presque[1][0] + int(taille_pattern/2)]
            counterItr +=1
        if counterItr > nbMaxItr: # Si on depasse le nombre max d'iterations
            corr1Presque = tempDeltaCorr1
        corr1 = corr1Presque
        corr1 = [int(np.mean(corr1[0])) , int(np.mean(corr1[1]))]
        tempDeltaCorr1 = corr1 # Mise à jour du point de correlation precedent

        maskCorr2 = np.zeros(containerImageTrait[i].shape)
        maskCorr2[corr2[0] - int(taille_pattern/2): corr2[0] + int(taille_pattern/2), corr2[1] - int(taille_pattern/2): corr2[1] + int(taille_pattern/2)] = 1
        maskCorr2 = np.copy(containerImageTrait[i]) * maskCorr2
        maskCorr2 = np.array(maskCorr2, dtype=np.uint8)

        corr2_res = cv2.matchTemplate(np.copy(imgVisee), carreeD, method=cv2.TM_CCOEFF_NORMED)
        corr2_res[:,0:corr2[1] - 45] = 0
        corr2_res[:,corr2[1] + 55:corr2_res.shape[1]] = 0
        corr2_res[0:corr2[0] - 65,:] = 0
        corr2_res[corr2[0] + 45:corr2_res.shape[0],:] = 0

        corr2Presque = np.where(corr2_res == np.max(corr2_res))
        corr2Presque = [corr2Presque[0] + int(taille_pattern/2), corr2Presque[1] + int(taille_pattern/2)]
        counterItr = 0
        while ((corr2Presque[1] - tempDeltaCorr2[1])**2 )**0.5 > 15 and counterItr <= nbMaxItr :
            corr2_res[corr2Presque[0] + int(taille_pattern/2), corr2Presque[1] + int(taille_pattern/2)] = 0
            corr2Presque = np.where(corr2_res == np.max(corr2_res))
            corr2Presque = [corr2Presque[0][0] + int(taille_pattern/2), corr2Presque[1][0] + int(taille_pattern/2)]
            counterItr +=1
        if counterItr > nbMaxItr:
            corr2Presque = tempDeltaCorr2
        corr2 = corr2Presque
        corr2= [int(np.mean(corr2[0])) , int(np.mean(corr2[1]))]
        tempDeltaCorr2 = corr2

        # Dessiner un carre centre sur les coordonnees
        center_y, center_x = corr1
        size = taille_pattern  # Taille du côte du carre
        
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
        size = taille_pattern  # Taille du côte du carre
        
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
            carreeG = imgVisee[corr1[0]-int(taille_pattern/2):corr1[0]+int(taille_pattern/2),corr1[1]-int(taille_pattern/2):corr1[1]+int(taille_pattern/2)]
            carreeD =  imgVisee[corr2[0]-int(taille_pattern/2):corr2[0]+int(taille_pattern/2),corr2[1]-int(taille_pattern/2):corr2[1]+int(taille_pattern/2)]
 
    ###########################################################
    
    listDistance = np.array(listDistance, dtype=np.int64)
    
    if flagreverse !=1:
        listDistance = list(reversed(listDistance)) 
        imageVerif = list(reversed(imageVerif))
    L0 = np.int64(listDistance[0])
    print(L0,type(L0))
    listDistance = (listDistance - L0)/L0
    
    return  (listDistance),recupCoords,imageVerif,coordsGTranches,coordsDTranches
