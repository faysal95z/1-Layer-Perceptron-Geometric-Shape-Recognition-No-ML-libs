from PIL import Image
import os
import numpy as np
import random

class Neurone :

    fCombi = []
    epsilon = 10**(-3)
    H = []
    l = 0
    
    for l in range(56**2):
        h = []
        k = 0
        while k < l :
            h.append(0)
            k += 1
        h.append(epsilon)
        k += 1
        while k < 56**2 :
            h.append(0)
            k += 1
        h = np.array(h)
        H.append(h)


    def __init__(self, fCombi):
        for i in range(56**2):
            fCombi.append(random.gauss(0,1))
        self.fCombi = np.array(fCombi)

    def sortie(self,entree):
        return np.dot(entree,self.fCombi)
    
    def derivPartXi(self,i,f,x,a):
        h = self.H[i]
        return (f(x+h,a)-f(x,a))/self.epsilon
    
    def gradF(self,f,x,a):
        grad = []
        for i in range(56**2):
            grad.append(self.derivPartXi(i,f,x,a))
        grad = np.array(grad)
        return grad
    
    def corfCombi(self, J):
        self.fCombi = np.array(self.fCombi) - np.array(J)

    def getfCombi(self):
        return self.fCombi
    
    def setfCombi(self,list):
        self.fCombi = np.array(list)
    
    def sigmoide(self,x):
        return 1/(1+np.exp(-x))