import numpy as np
from numpy.linalg import inv, det
import math

class GaussianDiscriminant:
    """
    Multivariate Gaussian classifier assuming class-dependent covariance
    """
    def __init__(self,k=2,d=8,priors=None): # k is number of classes, d is number of features
        # k and d are needed to initialize mean and covariance matrices
        self.mean = np.zeros((k,d)) # mean
        self.S = np.zeros((k,d,d)) # class-dependent covariance
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d

    def fit(self,Xtrain,ytrain):
        # split data into two groups based on label
        x1 = []
        y1 = []
        
        x2 = []
        y2 = []
        
        for i in range(len(ytrain)):
            if ytrain[i] == 1:
                x1.append(Xtrain[i])
                y1.append(ytrain[i])
                
            else:
                x2.append(Xtrain[i])
                y2.append(ytrain[i])
        #reshape
        l1 = len(x1)
        f1 = len(x1[0])
        l2 = len(x2)
        f2 = len(x2[0])
        
        x1 = np.array(x1).reshape(l1,f1)
        x1 = x1.T #transpose x1
        x2 = np.array(x2).reshape(l2,f2)
        x2 = x2.T #transpose x2         
        
        # compute the mean for each class
        y1_sum = np.sum(y1) # get sum of all labels in class 1
        y2_sum = np.sum(y2) # get sum of all labels in class 2
        
        m1 = []
        for i in range(len(x1)):
            m1.append(np.sum(x1[i]*y1[i])/y1_sum)

        m2 = []
        for i in range(len(x2)):
            m2.append(np.sum(x2[i]*y2[i])/y2_sum)
        
        m1 = np.asarray(m1)
        m2 = np.asarray(m2)
        
        self.mean[0] = m1 #sample mean of the first class, 1x8 vector
        self.mean[1] = m2 #sample mean of the second class

        # compute the class-dependent covariance
        s1 = np.cov(x1,ddof=None) # pass in split data for this class
        
        #self.S[0,:,:] = s1 # store cov matrix in self.S
        self.S[0] = s1
        
        s2 = np.cov(x2,ddof=None)
        self.S[1] = s2
        
        pass


    def predict(self,Xtest):
        # predict function to get prediction for test set
        predicted_class = np.ones(Xtest.shape[0]) # placeholder, all 1's
        
        Xtest_len = len(Xtest)
        k = self.k
          
        for i in np.arange(Xtest.shape[0]): # for each test set example; there are 100
            g = np.zeros((Xtest_len, k))
            x = Xtest[i].reshape(8,1) # transpose to get 8x1 vector
            
            for c in np.arange(self.k): # calculate discriminant function value for each class, there are 2 classes
                #function values are scores    
                m = self.mean[c].reshape(8,1)
                dot = np.float(np.dot(np.dot((x - m).T, inv(self.S[c])),(x - m)))
                val = -0.5 * math.log(det(self.S[c])) - 0.5 * dot + math.log(self.p[c])
                
                if c == 0:
                    g[i][0] = val
                elif c == 1:
                    g[i][1] = val
                else:
                    pass

            # determine the predicted class based on the discriminant function values
            # if score is high, then it belongs to that class     
            if g[i][0] > g[i][1]:
                predicted_class[i] = 1
            elif g[i][0] < g[i][1]:
                predicted_class[i] = 2
            else:
                pass
        return predicted_class

    def params(self):
        return self.mean[0],self.mean[1],self.S[0,:,:],self.S[1,:,:]


class GaussianDiscriminant_Ind:
    """
    Multivariate Gaussian classifier assuming class-independent covariance
    """
    def __init__(self,k=2,d=8,priors=None): # k is number of classes, d is number of features
        # k and d are needed to initialize mean and covariance matrices
        self.mean = np.zeros((k,d)) # mean
        self.S = np.zeros((d,d)) # class-independent covariance
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d

    def fit(self,Xtrain,ytrain):
        # split data into two groups based on label
        x1 = []
        y1 = []
        
        x2 = []
        y2 = []
        
        for i in range(len(ytrain)):
            if ytrain[i] == 1:
                x1.append(Xtrain[i])
                y1.append(ytrain[i])
                
            else:
                x2.append(Xtrain[i])
                y2.append(ytrain[i])
        #reshape
        l1 = len(x1)
        f1 = len(x1[0])
        l2 = len(x2)
        f2 = len(x2[0])
        
        x1 = np.array(x1).reshape(l1,f1)
        x1 = x1.T #transpose x1
        x2 = np.array(x2).reshape(l2,f2)
        x2 = x2.T #transpose x2         
        
        # compute the mean for each class
        y1_sum = np.sum(y1) # get sum of all labels in class 1
        y2_sum = np.sum(y2) # get sum of all labels in class 2
        
        m1 = []
        for i in range(len(x1)):
            m1.append(np.sum(x1[i]*y1[i])/y1_sum)

        m2 = []
        for i in range(len(x2)):
            m2.append(np.sum(x2[i]*y2[i])/y2_sum)
        
        m1 = np.asarray(m1)
        m2 = np.asarray(m2)
        
        self.mean[0] = m1 #sample mean of the first class, 1x8 vector
        self.mean[1] = m2 #sample mean of the second class

        # compute the class-dependent covariance
        s1 = np.cov(x1,ddof=None) # pass in split data for this class
        s2 = np.cov(x2,ddof=None)
        
        #get one common covariance matrix
        self.S = s1*self.p[0] + s2*self.p[1] #the sum of cov matrices multiplied by their prior
        
        pass

    def predict(self,Xtest):
        # predict function to get prediction for test set
        predicted_class = np.ones(Xtest.shape[0]) # placeholder

        Xtest_len = len(Xtest)
        k = self.k
        for i in np.arange(Xtest.shape[0]): # for each test set example; there are 100
            g = np.zeros((Xtest_len, k))
            x = Xtest[i].reshape(8,1) # transpose to get 8x1 vector 
            
            for c in np.arange(self.k): # calculate discriminant function value for each class, there are 2 classes
                #function values are scores    
                m = self.mean[c].reshape(8,1)
                dot = np.float(np.dot(np.dot((x - m).T, inv(self.S)),(x - m)))
                val = - 0.5 * dot + math.log(self.p[c])
                
                if c == 0:
                    g[i][0] = val
                elif c == 1:
                    g[i][1] = val
                else:
                    pass

            # determine the predicted class based on the discriminant function values
            # if score is high, then it belongs to that class     
            if g[i][0] > g[i][1]:
                predicted_class[i] = 1
            elif g[i][0] < g[i][1]:
                predicted_class[i] = 2
            else:
                pass

            # determine the predicted class based on the discriminant function values

        return predicted_class

    def params(self):
        return self.mean[0], self.mean[1], self.S
