# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 10:42:03 2016

@author: buttfive
"""
import numpy as np


class NeuNet:
    
        def __init__(self,layerSize,epsilon):
            self.layerSize = layerSize
            self.numLayers = np.shape(layerSize)[0]
            self.Theta = []
            for l in range(self.numLayers-1):
                self.Theta.append(np.random.rand(layerSize[l+1],layerSize[l]+1)*2*epsilon - epsilon)
               
        def loss(self,X,Y,Lambda):
            
            Loss = 1
            return Loss
        
        def forwardPropagation(self,x):
            a1 = np.zeros((self.layerSize[1]+1,))
            a1[0]=1
            a2 = np.zeros((self.layerSize[2],))
            z1 = np.dot(self.Theta[0],x)
            for j in range(np.shape(z1)[0]):
                a1[j+1]=1/(1+np.exp(-z1[j]))
            z2 = np.dot(self.Theta[1],a1)
            for j in range(np.shape(z2)[0]):
                a2[j]=1/(1+np.exp(-z2[j]))
            return x,a1,a2
        
        def backPropagation(self,x,y):
            a0,a1,a2 = self.forwardPropagation(x)
            delta2 = a2 - y
            delta1 = np.dot(self.Theta[1].T,delta2)*a1*(1-a1)
            delta1=delta1[1:]
            delta1 = np.reshape(delta1,(delta1.shape[0],1))
            delta2 = np.reshape(delta2,(delta2.shape[0],1))
            a0 = np.reshape(a0,(a0.shape[0],1))
            a1 = np.reshape(a1,(a1.shape[0],1))
            Delta0 = np.dot(delta1,a0.T)
            Delta1 = np.dot(delta2,a1.T)
            return Delta0,Delta1
        
        def globalBackPropagation(self,X,Y,Lambda):
            Delta0 = np.zeros((np.shape(self.Theta[0])[0],np.shape(self.Theta[0])[1]))
            Delta1 = np.zeros((np.shape(self.Theta[1])[0],np.shape(self.Theta[1])[1]))
            for i in range(np.shape(X)[0]):
                tempDelta0,tempDelta1 = self.backPropagation(X[i,:],Y[i])
                Delta0 += tempDelta0
                Delta1 += tempDelta1
            D0 = 1/(np.shape(X)[0])*Delta0 + Lambda*self.Theta[0]
            D1 = 1/(np.shape(X)[0])*Delta1 + Lambda*self.Theta[1]
            return D0,D1
        
        def train(self,X,Y,alpha,Lambda,tol,numIter):
            loop=0
            while(self.loss(X,Y,Lambda)>tol):
                self.Theta[0] -= alpha*self.globalBackPropagation(X,Y,Lambda)[0]
                self.Theta[1] -= alpha*self.globalBackPropagation(X,Y,Lambda)[1]
                loop += 1
                if (loop > numIter):
                    return False
            return True
        
        def predict(self,Z):
            P=np.zeros((Z.shape[0]))
            for i in range(Z.shape[0]):
                P[i] = np.argmax(self.forwardPropagation(Z[i,:])[2])
                #P = np.array(pd.get_dummies(P))
            return P
        
        def getCoefs(self):
            return self.Theta

        