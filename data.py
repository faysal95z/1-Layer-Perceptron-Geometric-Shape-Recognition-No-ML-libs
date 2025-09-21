from PIL import Image
import os
import numpy as np

class Data :

    forme = ""
    matrice = []
    chImg = ""
    vecMat = []

    def __init__(self, forme, matrice, chImg):
        self.forme = forme
        self.matrice = matrice
        self.chImg = chImg
    
    def imageEnMatrice(self):
        self.matrice = []
        img = Image.open("datasetC/train/"+self.forme+"/"+self.chImg)
        for i in range(56):
            self.matrice.append([])
            for j in range(56):
                if img.getpixel((i,j)) == (255,255,255) :
                    self.matrice[i].append(0)
                else :
                    self.matrice[i].append(1)
        self.matrice = np.array(self.matrice)
    
    def entree(self):
        self.vecMat = []
        for i in range(56):
            for j in range(56):
                self.vecMat.append(self.matrice[i,j])
        self.vecMat = np.array(self.vecMat)

    def getForme(self):
        return self.forme
    
    def getMatrice(self):
        return self.matrice
    
    def getVecMat(self):
        return self.vecMat