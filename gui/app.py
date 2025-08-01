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

from utils.correlation import *
from utils.lpe import *
from utils.maths_utils import *
from utils.segmentation import *
from utils.setup_images import *
from utils.utils_cv2 import *
from utils.miscelanous_image import *

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
        self.taille_pattern = 0
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

            self.original_images,self.original_imagesCopie = load_images_from_folder(folder_selected)
            self.original_images_reversed = list(reversed(np.copy(self.original_imagesCopie)))
            self.show_image()
            self.process_button = tk.Button(self, text="Appliquer le pre-traitement les images", command=self.prepare_images)
            self.process_button.pack()
            self.status_label = tk.Label(self, text="Chargement des images termine")
            self.status_label.pack()


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
        self.coords_rect = coordsPointsRectangle(self.original_images)

    def select_rectangle_fond_points(self):
        self.coords_rect_fond = coordsPointsRectangleFond(self.original_images,self.coords_rect_fond)

    def select_colors(self):
        self.couleurs_fond, self.couleurs_piece, self.markers_bonus = selectColors(self.original_images)
        self.segment_button = tk.Button(self, text="Lancer la Segmentation", command=self.run_segmentation)
        self.segment_button.pack()



    

    def run_segmentation(self):
            fond = simpledialog.askinteger("Entree", "Entrez la valeur du fond (1 pour clair, autre pour sombre):")
            self.typeDeGrad = simpledialog.askinteger("Entree", "Choisir le type de Gradient (1 pour Gradient Morphologique, autre pour Filtre Sobel)")
            
            first_grad = self.reconstructed_images[0]
            last_grad = self.reconstructed_images[-1]
            flagGrad = 0

            while flagGrad == 0:
                if self.typeDeGrad == 1:
                    self.tailleGrad = simpledialog.askinteger("Entree", "Choisir la taille de l'element structurant (impair)")
                    first_grad_displayed = apply_morphological_gradient(first_grad, self.tailleGrad)
                    last_grad_displayed = apply_morphological_gradient(last_grad, self.tailleGrad)
                else:
                    self.tailleGrad = simpledialog.askinteger("Entree", "Choisir la taille du filtre Sobel (1,3,5,7,11)")
                    self.dY = simpledialog.askinteger("Entree", "Choisir le DY")
                    self.dX = simpledialog.askinteger("Entree", "Choisir le dX")
                    first_grad_displayed = apply_sobel_gradient(first_grad, self.tailleGrad, self.dX, self.dY)
                    last_grad_displayed = apply_sobel_gradient(last_grad, self.tailleGrad, self.dX, self.dY)

                if first_grad_displayed is not None:
                    display_gradient_images(first_grad_displayed, last_grad_displayed)
                    flagGrad = simpledialog.askinteger("Entree", "Valider ce Gradient (1 pour oui)")
            
            self.segmented_images, exec_time = fonctionSegmentation(
                container=self.reconstructed_images,
                background=self.background,
                fond=fond,
                couleurFond=self.couleurs_fond,
                couleursPiece=self.couleurs_piece,
                coordsRect=self.coords_rect,
                markersBonus=self.markers_bonus,
                coordsRectFond=self.coords_rect_fond,
                typeDeGrad=self.typeDeGrad,
                tailleGrad=self.tailleGrad,
                dx=self.dX,
                dy=self.dY
            )
            
            messagebox.showinfo("Info", f"Segmentation terminee en {exec_time:.2f} secondes.")
            self.show_original = False
            self.show_image()
            # Création des boutons, le reste de votre code...
            self.result_button = tk.Button(self, text="Afficher le Resultat Segmentation", command=self.show_segmentation_result)
            self.result_button.pack()
            self.choseCorrectSeg = tk.Button(self, text="Corriger la segmentation", command=self.correctSegmentation)
            self.choseCorrectSeg.pack()
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
        self.zone_sectionmin = sectionPlusDeforme(self.segmented_images,self.original_imagesCopie)
        self.zoneminValide = simpledialog.askinteger("Entree", "Choisir cette section min (1 pour oui)")
        self.suivipoint_button = tk.Button(self, text="Lancer suivi de point", command=self.select_suivipoint)
        self.suivipoint_button.pack()

    def select_suivipoint(self):
        self.memoire = simpledialog.askinteger("Entree", "Activer memoire du pattern (1 pour oui):")
        self.tauxMaj = simpledialog.askinteger("Entree", "Taux de mise à jour du pattern")
        self.taille_pattern = simpledialog.askinteger("Entree", "Taille du pattern")
        pointSectionMin = self.zone_sectionmin
        self.coords_suivi = suiviPoint(container= self.original_images_reversed, memoire=self.memoire, taille_pattern=self.taille_pattern, tauxMaj=self.tauxMaj,pointSectionMin=pointSectionMin,zoneminValide=self.zoneminValide)
        self.sommefinale_button = tk.Button(self, text="Lancer calcul des deformations", command=self.select_sommefinale)
        self.sommefinale_button.pack()

        
    def select_sommefinale(self):
        taillemoy = simpledialog.askinteger("Entree", "Entrer la taille de la moyenne (nombre impair)")
        self.taillemoy = taillemoy
        self.resultsomme_tranches,self.resultsomme_tranchesVraie, self.images_trait_suivi = sommeSectionFinale(listLpe=self.segmented_images,coordonnesPointMin = self.coords_suivi,taillemoy=taillemoy, containerImgVraie=self.original_imagesCopie,taille_pattern=self.taille_pattern)
        self.original_images = self.images_trait_suivi
        self.flagReverseDeltaL = simpledialog.askinteger("Entree", "(Delta L 2 points) Partir de la premiere images (1 oui, autre pour partir de la derniere)")
        if self.flagReverseDeltaL == 1:
            self.result_distancedeltaL,recupCoords,self.images_trait_suivi,self.coordsGTranches,self.coordsDTranches = correlation(memoire = self.memoire,taille_pattern=self.taille_pattern , tauxMaj=self.tauxMaj, containerImageTrait=self.original_imagesCopie, flagreverse=self.flagReverseDeltaL, imageVerif = self.images_trait_suivi)
        else:
            self.result_distancedeltaL,recupCoords, self.images_trait_suivi,self.coordsGTranches,self.coordsDTranches = correlation(memoire = self.memoire,taille_pattern=self.taille_pattern , tauxMaj=self.tauxMaj, containerImageTrait=self.original_images_reversed, flagreverse=self.flagReverseDeltaL,  imageVerif = list(reversed(self.images_trait_suivi)))
        
        self.manuel = simpledialog.askinteger("Entree", "(Tranches)Reglage manuel de la taille des tranches ? (1 pour oui)")
        if self.manuel == 1:
            self.flagReverseTranches = simpledialog.askinteger("Entree", "(Tranches) Partir de la premiere images (1 oui, autre pour partir de la derniere)")
        else : 
            self.flagReverseTranches=self.flagReverseDeltaL

        if self.flagReverseTranches==1:
            self.result_distanceTranches,self.images_trait_suivi = correlationTranches(memoire2 = self.memoire, tauxMaj=self.tauxMaj, containerImageTrait=self.original_imagesCopie,flagreverse=self.flagReverseTranches, recupCoords= recupCoords, imageVerif = self.images_trait_suivi,coordsGTranches=self.coordsGTranches,coordsDTranches=self.coordsDTranches,taille_pattern=self.taille_pattern,manuel=self.manuel )
        else:
            self.result_distanceTranches,self.images_trait_suivi = correlationTranches(memoire2 = self.memoire, tauxMaj=self.tauxMaj, containerImageTrait=self.original_images_reversed, flagreverse=self.flagReverseTranches, recupCoords= recupCoords, imageVerif = list(reversed(self.images_trait_suivi)),coordsGTranches=self.coordsGTranches,coordsDTranches=self.coordsDTranches,taille_pattern=self.taille_pattern,manuel=self.manuel )
        
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