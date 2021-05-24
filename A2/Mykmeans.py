#import libraries
import numpy as np

class Kmeans:
    def __init__(self,k=6): # k is number of clusters
        self.num_cluster = k
        self.center = None
        self.error_history = []

    def run_kmeans(self, X, y):
        # initialize the centers of clutsers as a set of pre-selected samples
        init_idx = [1, 200, 1000, 1001, 1500, 2000]
        self.center = X[init_idx]
        num_iter = 0 # number of iterations for convergence

        # initialize cluster assignment
        prev_cluster_assignment = np.zeros([len(X),]).astype('int')
        cluster_assignment = np.zeros([len(X),]).astype('int')
        is_converged = False

        # iteratively update the centers of clusters till convergence
        while not is_converged:
            
            # iterate through the samples and compute their cluster assignment (E step)
            for i in range(len(X)):
                # use euclidean distance to measure the distance between sample and cluster centers
                all_dist = []
                for j in range(len(self.center)):
                    dist = np.linalg.norm(X[i]-self.center[j], axis=0)
                    #dist = distance.euclidean(X[i],self.center[j])
                    all_dist.append(dist)
                    # determine the cluster assignment by selecting the cluster whose center is closest to the sample
                closest_idx = np.argmin(all_dist) # returns index that specifies the closest cluster
                # print(closest_idx)
                cluster_assignment[i] = closest_idx
                
                pass
            
            # update the centers based on cluster assignment (M step)
            for i in range(len(self.center)): #length of 6
                data_x = [] # temp array to store the samples that are assigned to given cluster   
                for j in range(len(cluster_assignment)): #length of 3000
                    if cluster_assignment[j] == i:
                        data_x.append(X[j])
                    pass
                self.center[i] = np.mean(data_x, axis=0) #get the mean of each cluster and update center     
                pass
            
            # compute the reconstruction error for the current iteration
            cur_error = self.compute_error(X, cluster_assignment)
            self.error_history.append(cur_error)

            # reach convergence if the assignment does not change anymore
            is_converged = True if (cluster_assignment==prev_cluster_assignment).sum() == len(X) else False
            prev_cluster_assignment = np.copy(cluster_assignment)
            num_iter += 1

        # construct the contingency matrix
        """
        The contingency matrix represents the distribution of classes for each cluster. Suppose
        that we have 3 clusters and 300 samples (100 samples per class),
        and the algorithm generates perfect clusters (samples from different classes assigned
        to different clusters), the matrix could look like:
        [[100,0,0],
        [0,100,0],
        [0,0,100]]
        where each row corresponds to a cluster and each column for a class (in the order of digit 0,8,9
        for this assignment). Ideally, we would like the majority of samples in a cluster belong to a single
        class.
        """
        contingency_matrix = np.zeros([self.num_cluster,3])
        #counter = np.zeros([self.num_cluster,1])
        for i in range(self.num_cluster): #length of 6
            labels = np.zeros([3,1]).astype('int') 
            for j in range(len(cluster_assignment)): #3000, y is also this length
                
                if cluster_assignment[j] == i:            
                    #check y and see which class it belongs to, keep count
                    if y[j] == 0:
                        labels[0] += 1
                    elif y[j] == 8:
                        labels[1] += 1
                    elif y[j] == 9:
                        labels[2] += 1
                    else:
                        pass
                    
                for k in range(len(labels)):
                    contingency_matrix[i][k] = labels[k] # fill in table with this count
                    pass
                pass
                #counter[i] += 1
            pass

        return num_iter, self.error_history, contingency_matrix

    def compute_error(self,X,cluster_assignment):
        # compute the reconstruction error for given cluster assignment and centers
        error = 0 # placeholder
        
        for i in range(len(X)): #loop over all data samples
            for j in range(self.num_cluster): #loop over all clusters
                if cluster_assignment[i] == j: # if this data is assigned to this cluster
                    tot = np.sum((X[i] - self.center[j])**2) #sum of squares
                    error += tot
        
        return error

    def params(self):
        return self.center
