from PIL import Image
import os
import numpy as np
import random as rd

from data import Data
from neurone import Neurone

class main :

    DOSSIERS = ["test","train","val"]
    FORMES = ["circle","kite","parallelogram","rectangle","rhombus","square","trapezoid","triangle"]

    listeImageCircle = os.listdir("datasetC/train/circle")
    listeImageKite = os.listdir("datasetC/train/kite")
    listeImageParallelogram = os.listdir("datasetC/train/parallelogram")
    listeImageRectangle = os.listdir("datasetC/train/rectangle")
    listeImageRhombus = os.listdir("datasetC/train/rhombus")
    listeImageSquare = os.listdir("datasetC/train/square")
    listeImageTrapezoid = os.listdir("datasetC/train/trapezoid")
    listeImageTriangle = os.listdir("datasetC/train/triangle")

    listeImageTrain = listeImageCircle + listeImageKite + listeImageParallelogram + listeImageRectangle + listeImageRhombus + listeImageSquare + listeImageTrapezoid + listeImageTriangle
    listeIndicesTrain = list(range(12000))

    def formeDe(k):
        if 0<= k <=1499 :
            return "circle"
        elif 1500 <= k <= 2999:
            return "kite"
        elif 3000 <= k <= 4499:
            return "parallelogram"
        elif 4500 <= k <= 5999:
            return "rectangle"
        elif 6000 <= k <= 7499:
            return "rhombus"
        elif 7500 <= k <= 8999:
            return "square"
        elif 9000 <= k <= 10499:
            return "trapezoid"
        elif 10500 <= k <= 11999:
            return "triangle"

    NEURONE = Neurone([])

    ENTREE = 0

##ENTRAINEMENT
    for j in range(20):

        print(f"epoch {j+1} / 20")

        listeAtraiter = []
        for k in range(600):
            i = rd.randint(0,len(listeIndicesTrain)-1)
            listeAtraiter.append(listeIndicesTrain[i])
            del(listeIndicesTrain[i])

        for l in range(len(listeAtraiter)) :

            print(f"{j*5 + int(100*5*l/len(listeAtraiter))/100}%")

            indice = listeAtraiter[l]

            chImg = listeImageTrain[indice]

            T = 0

            T = Data(formeDe(indice),[],chImg)

            T.imageEnMatrice()

            T.entree()

            ENTREE = T.getVecMat()

            SORTIE = NEURONE.sigmoide(NEURONE.sortie(ENTREE))

            def erreur(EntreE,NeuronE) :
                return NeuronE.sigmoide(NeuronE.sortie(EntreE))

            J = NEURONE.gradF(erreur,ENTREE,NEURONE)

            NEURONE.corfCombi(J)
    '''

    Pour prendre les valeurs de poids synaptiques au lieu d'entrainer un nouveau neurone.

    poidsSynaptiques = open("dataPoidsSynaptiques.txt", "w")
    poidsSynaptiques.write(str(NEURONE.getfCombi().tolist()))
    poidsSynaptiques.close()

    poidsSynaptiques = open("dataPoidsSynaptiques.txt", "r")
    P = poidsSynaptiques.read()
    poidsSynaptiques.close()

    p = []
    for m in range(len(P)):
        if m%2 != 0:
            p.append(float(list(P)[m]))
    '''


    ##TEST seuil ente ok 10**(-6.478335) et 10**(-6.47834) no
    '''
    TRAINEDNEURONE = Neurone([])

    TRAINEDNEURONE.setfCombi(p)
    '''

    seuil = .5

    TESTimg1 = Data("../test/circle",[],"circle-2.png")
    TESTimg2 = Data("../test/circle",[],"circle-3.png")

    TESTimg3 = Data("../test/rectangle",[],"rectangle-0.png")
    TESTimg4 = Data("../test/rectangle",[],"rectangle-15.png")

    TESTimg5 = Data("../test/triangle",[],"triangle-0.png")
    TESTimg6 = Data("../test/triangle",[],"triangle-20.png")

    TESTimg7 = Data("circle",[],"circle-1.png")

    TESTimg1.imageEnMatrice()
    TESTimg1.entree()
    ENTREE = TESTimg1.getVecMat()
    SORTIE = NEURONE.sigmoide(NEURONE.sortie(ENTREE))
    if SORTIE >= seuil :
        print("circle-2 est un cercle")
    else :
        print("circle-2 n'est pas un cercle")

    TESTimg2.imageEnMatrice()
    TESTimg2.entree()
    ENTREE = TESTimg2.getVecMat()
    SORTIE = NEURONE.sigmoide(NEURONE.sortie(ENTREE))
    if SORTIE >= seuil :
        print("circle-3 est un cercle")
    else :
        print("circle-3 n'est pas un cercle")

    TESTimg3.imageEnMatrice()
    TESTimg3.entree()
    ENTREE = TESTimg3.getVecMat()
    SORTIE = NEURONE.sigmoide(NEURONE.sortie(ENTREE))
    if SORTIE >= seuil :
        print("rectangle-0 est un cercle")
    else :
        print("rectangle-0 n'est pas un cercle")

    TESTimg4.imageEnMatrice()
    TESTimg4.entree()
    ENTREE = TESTimg4.getVecMat()
    SORTIE = NEURONE.sigmoide(NEURONE.sortie(ENTREE))
    if SORTIE >= seuil :
        print("triangle-15 est un cercle")
    else :
        print("triangle-15 n'est pas un cercle")

    TESTimg5.imageEnMatrice()
    TESTimg5.entree()
    ENTREE = TESTimg5.getVecMat()
    SORTIE = NEURONE.sigmoide(NEURONE.sortie(ENTREE))
    if SORTIE >= seuil :
        print("rectangle-0 est un cercle")
    else :
        print("rectangle-0 n'est pas un cercle")

    TESTimg6.imageEnMatrice()
    TESTimg6.entree()
    ENTREE = TESTimg6.getVecMat()
    SORTIE = NEURONE.sigmoide(NEURONE.sortie(ENTREE))
    if SORTIE >= seuil :
        print("rectangle-20 est un cercle")
    else :
        print("rectangle-20 n'est pas un cercle")
    
    TESTimg7.imageEnMatrice()
    TESTimg7.entree()
    ENTREE = TESTimg7.getVecMat()
    SORTIE = NEURONE.sigmoide(NEURONE.sortie(ENTREE))
    if SORTIE >= seuil :
        print("circle-1 est un cercle")
    else :
        print("circle-1 n'est pas un cercle")