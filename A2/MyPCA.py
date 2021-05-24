import numpy as np

def PCA(X,num_dim=None):
    
    cov_mat = np.cov(X.T,ddof=None)
    
    w, v = np.linalg.eigh(cov_mat) #get the values for w (eigenvalues) and v (eigenvectors)
    w_flip = np.flip(w, axis = 0)
    v_flip = np.flip(v, axis = 1) #flip along column axis
    
    X_mean = np.mean(X, axis=0)
    X_new = X - X_mean
    
    # project the high-dimensional data to low-dimensional one
    if num_dim is None:
        
        # select the reduced dimensions that keep >90% of the variance
        tot = np.sum(w_flip)
        
        pov = 0
        i = 0
        while(pov < 0.9):
            val = w_flip[i]/tot
            pov += val
            i+=1
        
        num_dim = i
        v_s = v_flip[:,0:i] #keep only vectors in which their cumulative variance add to 90%     
        
        projZ = np.matmul(X_new, v_s)
        
    else: #num_dim is given
        v_s = v_flip[:,0:num_dim] #eigenvectors selected

        projZ = np.matmul(X_new, v_s)
    
    X_pca = projZ
    
    return X_pca, num_dim
