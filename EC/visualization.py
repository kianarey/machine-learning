from matplotlib import pyplot as plt
import numpy as np
#from planar_utils import plot_decision_boundary

def plot_boundary(clf,x,y):
    """
    Plot the decision boundary of the kernel perceptron, and the samples (using different
    colors for samples with different labels)
    """
    
    # #pred = clf.predict(x)
    
    # #X = np.meshgrid()
    
    # xlist = np.linspace(-3.0, 3.0, 20)
    # ylist = np.linspace(-3.0, 3.0, 20)
    
    # X, Y = np.meshgrid(x[:,0],x[:,1])
    # Z = clf.predict(x)
    # fig, ax = plt.subplots(1,1)
    # cp = ax.contourf(X, Y, Z)
    # fig.colorbar(cp)
    # plt.show()
    
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    h = 0.02
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha =0.8)
    #plt.axis(off)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    
    plt.show()

    pass
