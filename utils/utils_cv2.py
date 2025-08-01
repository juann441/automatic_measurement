import cv2
import numpy as np


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
