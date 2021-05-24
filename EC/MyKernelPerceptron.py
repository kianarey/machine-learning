import numpy as np

"""
Specify your sigma for RBF kernel in the order of questions (simulated data, digit-49, digit-79)
"""
sigma_pool = [0.00001,1.0,0.8] #0.0002

#sigma_pool = [0.5,0.1,0.1] #0.0002

class KernelPerceptron:
    """
    Perceptron Algorithm with RBF Kernel
    """
    def __init__(self,train_x, train_y, sigma_idx):
        self.sigma = sigma_pool[sigma_idx] # sigma value for RBF kernel
        self.train_x = train_x # kernel perceptron makes predictions based on training data
        self.train_y = train_y
        self.alpha = np.zeros([len(train_x),]).astype('float32') # parameters to be optimized

    def RBF_kernel(self,x):
        # Implement the RBF kernel
        val = np.linalg.norm(self.train_x - x, axis=1)
        return val

    def fit(self,train_x,train_y):
        # set a maximum training iteration
        max_iter = 1000
               
        for iter in range(max_iter):
            error_count = 0
            rbf_mat=[]
            for i in range(len(train_x)):
                rbf = self.RBF_kernel(train_x[i])
                rbf_exp = np.exp(-pow(rbf,2)/ (2 * self.sigma))
                rbf_mat.append(rbf_exp)
                   
            rbf_mat = np.asarray(rbf_mat)
            
            tot_y = []
            for i in range(len(rbf_mat)):
                val = np.sign(np.sum(self.alpha[i] * train_y[i] * rbf_mat[i]))
                tot_y.append(val)
            tot_y = np.asarray(tot_y)
            
            for i in range(len(train_y)):
                if tot_y[i] != train_y[i]:
                    
                    self.alpha[i] += 1
                    error_count += 1
            # print(tot_y)
            # print(self.alpha)
            
            if error_count == 0:
                break
                    

    def predict(self,test_x):
        # generate predictions for given data
        #pred = np.ones([len(test_x)]).astype('float32') # placeholder
        
        rbf_mat=[]
            
        for i in range(len(test_x)):
            rbf = self.RBF_kernel(test_x[i])
            rbf_exp = np.exp(-pow(rbf,2)/ (2 * self.sigma))
            rbf_mat.append(rbf_exp)
               
        rbf_mat = np.asarray(rbf_mat)
        #print(rbf_mat.shape)
        
        res = self.alpha * self.train_y
        tot_y = np.sign(np.dot(rbf_mat,res))
        
        # for i in range(len(rbf_mat)):
        #     val = np.sign(np.sum(self.alpha[i] * self.train_y[i] * rbf_mat[i]))
        #     tot_y.append(val)
        # tot_y = np.asarray(tot_y)
        #print(tot_y)

        return tot_y

    def param(self,):
        return self.alpha
