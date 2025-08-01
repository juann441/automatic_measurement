import numpy as np

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