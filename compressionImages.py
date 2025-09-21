from PIL import Image
import os
import numpy as np


DOSSIERS = ["test","train","val"]
FORMES = ["circle","kite","parallelogram","rectangle","rhombus","square","trapezoid","triangle"]

def moyenne(Gmat,p,q,seuil):
    S = 0
    for i in range(4):
        for j in range(4):
            S+= Gmat[4*p+i][4*q+j]
    m = S/16
    if m >= seuil :
        return 1
    return 0

for dossier in DOSSIERS :
    for forme in FORMES :

        listeImageN = os.listdir(f"dataset/{dossier}/{forme}")
        listeMatriceImageN = []
        listeMatriceImageC = []

        for chImg in listeImageN :

            matImg = []
            img = Image.open(f"dataset/{dossier}/{forme}/"+chImg)
            for i in range(224):
                matImg.append([])
                for j in range(224):
                    if img.getpixel((i,j)) == (255,255,255) :
                        matImg[i].append(0)
                    else :
                        matImg[i].append(1)
            listeMatriceImageN.append(matImg)

        for matN in listeMatriceImageN :
            matC = []
            for k in range(56):
                matC.append([])
                for l in range(56):
                    matC[k].append(moyenne(matN,k,l,.9))
            listeMatriceImageC.append(matC)

        for mat in listeMatriceImageC :
            img = Image.new('1',(56,56))
            for k in range(56):
                for l in range(56):
                    img.putpixel((k,l),int(np.abs(mat[k][l]-1)))
            img.save(f"datasetC/{dossier}/{forme}/{forme}-{listeMatriceImageC.index(mat)}.png")
            print(f"enregistrement de l'image {listeMatriceImageC.index(mat)}")
        print(f"{forme} de {dossier} fini")