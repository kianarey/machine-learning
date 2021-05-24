# Machine Learning

This repository includes all assignments completed by me from graduate course `CSCI 5521 Machine Learning` offered by University of Minnesota - Twin Cities, MN during Spring 2021

### Assignment 1 - Multivariate Gaussian Classifiers
For this assignment, I implemented two multivariate Gaussian classifiers with different assumptions:

- S1 and S2 (covariance matrices) are learned independently (i.e. learned from the data from each class)

- S1 = S2 (learned from the data from both classes).

### Assignment 2 - K-means and Principal Component Analysis (PCA) Algorithms
For this assignment, I implemented and applied k-means algorithm to the provided dataset `Digits089.csv`. This algorithm iteratively updates the center of each cluster based on the input samples and records the reconstruction error after each iteration. After reaching convergence, it computes a contigency matrix that represents the class distribution for the final clusters, which is used to measure the quality of clustering results.

The process described above is repeated, but this time using low-dimensional data obtained by PCA. My PCA algorithm reduced the original samples to dimenions needed to capture >90% of variance. The number of dimensions required is 73. 

This process is repeated again, this time using only the first principal component. My results show that using only the first principal component produced poorer results.
  
### Assignment 3 - Multilayer Perceptron
For this assignment, I implemented a multilayer perceptron (MLP) for optical-digit classification. I trained my MLP, tuned the number of hidden units, and tested the prediction performance using the provided data. Data were normalized prior to training, tuning, and testing. The activation functions `sigmoid` and `softmax` were used. The error function used is the cross-entropy loss function. 

I tested my MLP with various hidden units; H = 4, 8, 12, 16, 20, and 24 hidden units. Results are visualized in my report.

### Assignment 4 - Univariate Decision Tree
For this assignment, I implemented a univariate decision tree for optical-digit classification. I trained my decision tree, tuned the minimum node entropy, and tested the prediction performance using the data provided. The minimum node entropy is used as a threshold to stop splitting the tree. Once splitting stopped, the node is set as a leaf.

My decision tree is implemented and tested with various minimum node entropy values: theta = 0.01, 0.05, 0.01, 0.2, 0.4, 0.8, 1.0, 2.0. The results and model complexity of my decision tree are discussed in my report.

### Extra Credit Assignment - Kernel Perceptron using Radial Basis Function (RBF) Kernel
For this extra credit assignment, I implemented a kernel perceptron using RBF kernel. Different values of sigma were used to provide best accuracy. The results for different sigma values are explained in my report. 

Once trained, I evaluated my implementation on the provided datasets. See my report for results and visualization.
