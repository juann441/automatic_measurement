import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import numpy as np
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
        self.title("Mesure des déformations – Interface PRO")
        self.geometry("1200x800")
        self.configure(bg="#f2f2f2")

        # Variables
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


        # === TOP FRAME : sélection dossier ===
        self.top_frame = tk.Frame(self, bg="#f2f2f2")
        self.top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        self.label = ttk.Label(self.top_frame, text="Sélectionnez un dossier :")
        self.label.grid(row=0, column=0, sticky="w")

        self.select_button = ttk.Button(self.top_frame, text="Parcourir", command=self.select_folder)
        self.select_button.grid(row=0, column=1, padx=5)

        # === IMAGE DISPLAY FRAME ===
        self.image_frame = tk.Frame(self, bg="#ffffff", relief=tk.RIDGE, bd=2)
        self.image_frame.grid(row=1, column=0, padx=10, pady=10)

        self.image_label = tk.Label(self.image_frame, bg="#ffffff")
        self.image_label.pack()

        # === NAVIGATION BUTTONS ===
        self.nav_frame = tk.Frame(self)
        self.nav_frame.grid(row=2, column=0, pady=5)

        self.prev_button = ttk.Button(self.nav_frame, text="← Précédente", command=self.show_prev_image)
        self.play_button = ttk.Button(self.nav_frame, text="▶ Play", command=self.play_images)
        self.next_button = ttk.Button(self.nav_frame, text="Suivante →", command=self.show_next_image)
        self.next_ten_button = ttk.Button(self.nav_frame, text="+10", command=self.show_next_ten_image)

        self.prev_button.grid(row=0, column=0, padx=5)
        self.play_button.grid(row=0, column=1, padx=5)
        self.next_button.grid(row=0, column=2, padx=5)
        self.next_ten_button.grid(row=0, column=3, padx=5)

        # === STATUS ===
        self.image_number_label = ttk.Label(self, text="Image : --")
        self.image_number_label.grid(row=3, column=0, pady=5)

        # === CONTROL BUTTONS FRAME ===
        self.control_frame = tk.LabelFrame(self, text="Étapes de traitement", bg="#f2f2f2")
        self.control_frame.grid(row=4, column=0, padx=10, pady=10, sticky="ew")

        # === CONTROL REGLAGES BONUS ===
        self.bonus_settings = tk.LabelFrame(self, text="Réglages segmentation", bg="#f2f2f2")
        self.bonus_settings.grid(row=5, column=0, padx=10, pady=10, sticky="ew")

        # === CONTROL CALCULATIONS FRAME ===
        self.calcul_frame = tk.LabelFrame(self, text="Calcul des déformations", bg="#f2f2f2")
        self.calcul_frame.grid(row=6, column=0, padx=10, pady=10, sticky="ew")

        # === CONTROL CALCULATIONS FRAME ===
        self.final_frame = tk.LabelFrame(self, text="Résultats finaux", bg="#f2f2f2")
        self.final_frame.grid(row=7, column=0, padx=10, pady=10, sticky="ew")

    def select_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.original_images, self.original_imagesCopie = load_images_from_folder(folder_selected)
            if self.original_images:
                self.original_images_reversed = list(reversed(np.copy(self.original_imagesCopie)))
                self.show_image()
                self.process_button = ttk.Button(self.control_frame, text="Pré-traiter les images", command=self.prepare_images)
                self.process_button.grid(row=0, column=0, padx=5, pady=5)
                self.status_label = tk.Label(self, text="Chargement des images termine")
                self.status_label.grid(row=0, column=1, padx=5, pady=5)
            else:
                self.status_label.config(text="Aucune image trouvée dans le dossier.")


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
            self.image_number_label.config(text=f"Image numéro : {self.current_image_index + 1} / {len(self.original_images)}")
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

    def show_prev_image(self):
        if self.original_images:
            self.current_image_index = (self.current_image_index - 1) % len(self.original_images)
            self.show_image()

    def show_next_image(self):
        if self.original_images:
            self.current_image_index = (self.current_image_index + 1) % len(self.original_images)
            self.show_image()

    def show_next_ten_image(self):
        if self.original_images:
            self.current_image_index = (self.current_image_index + 10) % len(self.original_images)
            self.show_image()

    def prepare_images(self):
        # Affichage du status initial
        self.status_label.config(text="Chargement en cours...")
        self.update()

        # Validation interactive du flou
        flou_valide = False
        while not flou_valide:
            self.flou = simpledialog.askinteger("Flou", "Entrez la valeur de la puissance du flou :")
            if self.flou is None:
                return  # Annulation utilisateur

            # Prévisualisation
            img1 = preparationImagesUnique(np.copy(self.original_images[0]), self.flou)
            img2 = preparationImagesUnique(np.copy(self.original_images[-1]), self.flou)
            show_images_for_validation(img1, img2)

            valider = simpledialog.askinteger("Validation", "Valider ce flou ? (1 = Oui)")
            if valider == 1:
                flou_valide = True

        self.status_label.config(text="Application du flou à toutes les images...")
        self.update()

        # Traitement complet
        self.reconstructed_images, self.background, exec_time = preparationImages(self.original_images, self.flou)
        self.status_label.config(text=f"Images préparées en {exec_time:.2f} s")

        # Nettoyer les anciens boutons si besoin
        for widget in self.control_frame.winfo_children():
            widget.destroy()

        # Ajout des boutons dans une grille élégante
        self.rect_button = ttk.Button(self.control_frame, text="Rectangles Éprouvette", command=self.select_rectangle_points)
        self.rect_fond_button = ttk.Button(self.control_frame, text="Rectangles Fond", command=self.select_rectangle_fond_points)
        self.color_button = ttk.Button(self.control_frame, text="Couleurs", command=self.select_colors)
        self.segment_button = ttk.Button(self.control_frame, text="Segmentation", command=self.run_segmentation)

        self.rect_button.grid(row=1, column=0, padx=5, pady=5)
        self.rect_fond_button.grid(row=1, column=1, padx=5, pady=5)
        self.color_button.grid(row=1, column=2, padx=5, pady=5)
        self.segment_button.grid(row=1, column=3, padx=5, pady=5)
    
    def select_rectangle_points(self):
        self.coords_rect = coordsPointsRectangle(self.original_images)
    def select_rectangle_fond_points(self):
        self.coords_rect_fond = coordsPointsRectangleFond(self.original_images,self.coords_rect_fond)
    def select_colors(self):
        self.couleurs_fond, self.couleurs_piece, self.markers_bonus = selectColors(self.original_images)

    def select_zonemin(self):
        self.zone_sectionmin = sectionPlusDeforme(self.segmented_images,self.original_imagesCopie)
        self.zoneminValide = simpledialog.askinteger("Entree", "Choisir cette section min (1 pour oui)")

        self.suivipoint_button = tk.Button(self.calcul_frame, text="Lancer suivi de point", command=self.select_suivipoint)

        self.suivipoint_button.grid(row=1, column=0, padx=5, pady=5)

    
    def select_suivipoint(self):
        self.memoire = simpledialog.askinteger("Entree", "Activer memoire du pattern (1 pour oui):")
        self.tauxMaj = simpledialog.askinteger("Entree", "Taux de mise à jour du pattern")
        self.taille_pattern = simpledialog.askinteger("Entree", "Taille du pattern")
        pointSectionMin = self.zone_sectionmin
        self.coords_suivi = suiviPoint(container= self.original_images_reversed, memoire=self.memoire, taille_pattern=self.taille_pattern, tauxMaj=self.tauxMaj,pointSectionMin=pointSectionMin,zoneminValide=self.zoneminValide)

        self.sommefinale_button = tk.Button(self.calcul_frame, text="Lancer calcul des deformations", command=self.select_sommefinale)
        self.sommefinale_button.grid(row=1, column=1, padx=5, pady=5)


    def run_segmentation(self):
        fond = simpledialog.askinteger("Fond", "Entrez la valeur du fond (1 = clair, autre = sombre) :")
        if fond is None:
            return

        self.typeDeGrad = simpledialog.askinteger("Type de Gradient", "1 = Morphologique, autre = Sobel")
        if self.typeDeGrad is None:
            return

        grad_valide = False
        while not grad_valide:
            if self.typeDeGrad == 1:
                self.tailleGrad = simpledialog.askinteger("Morphologique", "Taille de l'élément structurant (impair) :")
                if self.tailleGrad is None:
                    return
                grad1 = apply_morphological_gradient(self.reconstructed_images[0], self.tailleGrad)
                grad2 = apply_morphological_gradient(self.reconstructed_images[-1], self.tailleGrad)
            else:
                self.tailleGrad = simpledialog.askinteger("Sobel", "Taille du filtre Sobel (1,3,5,7,11) :")
                self.dY = simpledialog.askinteger("Sobel", "Valeur de DY :")
                self.dX = simpledialog.askinteger("Sobel", "Valeur de DX :")
                if None in (self.tailleGrad, self.dY, self.dX):
                    return
                grad1 = apply_sobel_gradient(self.reconstructed_images[0], self.tailleGrad, self.dX, self.dY)
                grad2 = apply_sobel_gradient(self.reconstructed_images[-1], self.tailleGrad, self.dX, self.dY)

            display_gradient_images(grad1, grad2)
            valid = simpledialog.askinteger("Validation", "Valider ce gradient ? (1 = Oui)")
            if valid == 1:
                grad_valide = True

        # Paramètres de segmentation
        params = {
            'container': self.reconstructed_images,
            'background': self.background,
            'fond': fond,
            'couleurFond': self.couleurs_fond,
            'couleursPiece': self.couleurs_piece,
            'coordsRect': self.coords_rect,
            'markersBonus': self.markers_bonus,
            'coordsRectFond': self.coords_rect_fond,
            'typeDeGrad': self.typeDeGrad,
            'tailleGrad': self.tailleGrad,
            'dx': self.dX,
            'dy': self.dY
        }

        self.segmented_images, exec_time = fonctionSegmentation(**params)
        messagebox.showinfo("Info", f"Segmentation terminée en {exec_time:.2f} secondes.")

        self.show_original = False
        self.show_image()
        self.result_button = tk.Button(self.bonus_settings, text="Afficher le Resultat Segmentation", command=self.show_segmentation_result)
        self.result_button.grid(row=1, column=0, padx=5, pady=5)
        self.choseCorrectSeg = tk.Button(self.bonus_settings, text="Corriger la segmentation", command=self.correctSegmentation)
        self.choseCorrectSeg.grid(row=1, column=1, padx=5, pady=5)
        self.chosezone_button = tk.Button(self.bonus_settings, text="Localiser la zone de striction", command=self.select_zonemin)
        self.chosezone_button.grid(row=1, column=2, padx=5, pady=5)
    
    
    def correctSegmentation(self):
        # Utilisation de booléens pour plus de clarté
        corriger_eprouvette = simpledialog.askinteger("Entree", "Corriger l'arrière-plan ou l'eprouvette (1 pour eprouvette, autre pour fond)") == 1
        
        if simpledialog.askinteger("Entree", "Corriger la segmentation ? (0 pour NON)") != 0:
            validation_faite = False
            while not validation_faite:
                self.tailleCorrect = simpledialog.askinteger("Entree", "Choisir taille correction de segmentation")
                
                if self.tailleCorrect is None:
                    return # L'utilisateur a annulé
                
                # Appliquer la correction sur la première et la dernière image pour prévisualisation
                first_corrected = apply_segmentation_correction(np.copy(self.segmented_images[0]), self.tailleCorrect, corriger_eprouvette)
                last_corrected = apply_segmentation_correction(np.copy(self.segmented_images[-1]), self.tailleCorrect, corriger_eprouvette)
                
                plt.figure()
                plt.subplot(211)
                plt.imshow(first_corrected, cmap='gray')
                plt.title("Première image corrigée")
                plt.subplot(212)
                plt.imshow(last_corrected, cmap='gray')
                plt.title("Dernière image corrigée")
                plt.show()

                if simpledialog.askinteger("Entree", "Valider cette correction ?(1 pour oui)") == 1:
                    validation_faite = True
                    # Appliquer la correction sur toutes les images une fois pour toutes
                    self.segmented_images = [
                        apply_segmentation_correction(img, self.tailleCorrect, corriger_eprouvette) 
                        for img in self.segmented_images
                    ]
            
            self.show_image() # Mettre à jour l'affichage avec les images corrigées
            
    
    def show_segmentation_result(self):
        if self.segmented_images:
            self.switch_to_segmented = not self.switch_to_segmented
            self.show_image()
    
    def dummy(self):
        messagebox.showinfo("Info", "Fonction non implémentée dans cette version.")
    
    def select_sommefinale(self):
        taillemoy = simpledialog.askinteger("Entree", "Entrer la taille de la moyenne (nombre impair)")
        if taillemoy is None:
            return

        self.taillemoy = taillemoy
        self.resultsomme_tranches, self.resultsomme_tranchesVraie, self.images_trait_suivi = sommeSectionFinale(
            listLpe=self.segmented_images,
            coordonnesPointMin=self.coords_suivi,
            taillemoy=self.taillemoy,
            containerImgVraie=self.original_imagesCopie,
            taille_pattern=self.taille_pattern
        )
        self.original_images = self.images_trait_suivi

        # Passer à l'étape suivante, maintenant gérée par une autre méthode
        self.perform_correlation()

    def perform_correlation(self):
        self.flagReverseDeltaL = simpledialog.askinteger("Entree", "(Delta L 2 points) Partir de la premiere images (1 oui, autre pour partir de la derniere)")
        if self.flagReverseDeltaL is None:
            return

        if self.flagReverseDeltaL == 1:
            self.result_distancedeltaL, recupCoords, self.images_trait_suivi, self.coordsGTranches, self.coordsDTranches = correlation(
                memoire=self.memoire,
                taille_pattern=self.taille_pattern,
                tauxMaj=self.tauxMaj,
                containerImageTrait=self.original_imagesCopie,
                flagreverse=self.flagReverseDeltaL,
                imageVerif=self.images_trait_suivi
            )
        else:
            self.result_distancedeltaL, recupCoords, self.images_trait_suivi, self.coordsGTranches, self.coordsDTranches = correlation(
                memoire=self.memoire,
                taille_pattern=self.taille_pattern,
                tauxMaj=self.tauxMaj,
                containerImageTrait=self.original_images_reversed,
                flagreverse=self.flagReverseDeltaL,
                imageVerif=list(reversed(self.images_trait_suivi))
            )
        
        # Stocker recupCoords pour l'étape suivante
        self.recupCoords = recupCoords
        
        self.perform_correlation_tranches()

    def perform_correlation_tranches(self):
        self.manuel = simpledialog.askinteger("Entree", "(Tranches)Reglage manuel de la taille des tranches ? (1 pour oui)")
        if self.manuel is None:
            return
            
        if self.manuel == 1:
            self.flagReverseTranches = simpledialog.askinteger("Entree", "(Tranches) Partir de la premiere images (1 oui, autre pour partir de la derniere)")
            if self.flagReverseTranches is None:
                return
        else:
            self.flagReverseTranches = self.flagReverseDeltaL

        if self.flagReverseTranches == 1:
            self.result_distanceTranches, self.images_trait_suivi = correlationTranches(
                memoire2=self.memoire,
                tauxMaj=self.tauxMaj,
                containerImageTrait=self.original_imagesCopie,
                flagreverse=self.flagReverseTranches,
                recupCoords=self.recupCoords,
                imageVerif=self.images_trait_suivi,
                coordsGTranches=self.coordsGTranches,
                coordsDTranches=self.coordsDTranches,
                taille_pattern=self.taille_pattern,
                manuel=self.manuel
            )
        else:
            self.result_distanceTranches, self.images_trait_suivi = correlationTranches(
                memoire2=self.memoire,
                tauxMaj=self.tauxMaj,
                containerImageTrait=self.original_images_reversed,
                flagreverse=self.flagReverseTranches,
                recupCoords=self.recupCoords,
                imageVerif=list(reversed(self.images_trait_suivi)),
                coordsGTranches=self.coordsGTranches,
                coordsDTranches=self.coordsDTranches,
                taille_pattern=self.taille_pattern,
                manuel=self.manuel
            )

        print(self.result_distanceTranches)
        self.plot_btn = tk.Button(self.final_frame, text="Afficher le graphique", command=self.plot_results)
        self.plot_btn.grid(row=1, column=1, padx=5, pady=5)

    def toggle_images(self):
        self.show_original = not self.show_original
        self.show_image()
        
    def save_results(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            results = {
                "Deformation locale ING (programme)": self.resultsomme_tranchesCopie,
                "Deformation axiale VRAIE (programme)": self.resultsomme_tranchesVraieCopie,
                "Deformation moyenne(longitudinale) (suivi de deux points)": self.result_distancedeltaLCopie,
                "Deformation moyenne(longitudinale) (tranches)": self.result_distanceTranchesCopie
            }
            save_to_csv(save_path, results)

    
    def plot_results(self):
        # Création et affichage du premier graphique
        fig, ax = plt.subplots()
        plot_data(ax, self.resultsomme_tranches, 'Déformation locale', 'b', self.taillemoy)
        plot_data(ax, self.resultsomme_tranchesVraie, 'Déformation locale vraie', 'y')
        plot_data(ax, self.result_distancedeltaL, 'Déformation moyenne (2 points)', 'r')
        plot_data(ax, self.result_distanceTranches, 'Déformation moyenne (Tranches)', 'g')
        ax.set_xlabel("Images")
        ax.legend() # Utilisez l'argument 'label' dans plot_data
        plt.show()

        # Demande de paramètre et application de la régression
        lam = simpledialog.askinteger("Entree", "Entrer la valeur de lambda (regression lineaire)")
        if lam is None:
            return

        self.resultsomme_tranchesCopie = regression(self.resultsomme_tranches, lam)
        self.resultsomme_tranchesVraieCopie = regression(self.resultsomme_tranchesVraie, lam)
        self.result_distancedeltaLCopie = regression(self.result_distancedeltaL, lam)
        self.result_distanceTranchesCopie = regression(self.result_distanceTranches, lam)

        # Création et affichage du deuxième graphique (post-régression)
        fig, ax = plt.subplots()
        plot_data(ax, self.resultsomme_tranchesCopie, 'Déformation locale', 'b', self.taillemoy)
        plot_data(ax, self.resultsomme_tranchesVraieCopie, 'Déformation locale vraie', 'y')
        plot_data(ax, self.result_distancedeltaLCopie, 'Déformation moyenne (2 points)', 'r')
        plot_data(ax, self.result_distanceTranchesCopie, 'Déformation moyenne (Tranches)', 'g')
        ax.set_xlabel("Images")
        ax.legend()
        plt.show()

        self.save_btn = tk.Button(self.final_frame, text="Enregistrer", command=self.save_results)
        self.save_btn.grid(row=1, column=0, padx=5, pady=5)

if __name__ == "__main__":
    app = ImageViewer()
    app.mainloop()
