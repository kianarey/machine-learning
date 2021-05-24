import numpy as np

def generate_data():
    # sampling with fixed seed for reproducibility
    np.random.seed(1); r1 = np.sqrt(np.random.random([100,]))
    np.random.seed(1); t1 = 2*np.pi*np.random.random([100,])
    # samples in class 1
    x1 = np.array([r1*np.cos(t1),r1*np.sin(t1)])

    # sampling with fixed seed for reproducibility
    np.random.seed(1); r2 = np.sqrt(3*np.random.random([100,])+2)
    np.random.seed(1); t2 = 2*np.pi*np.random.random([100,])
    # samples in class 2
    x2 = np.array([r2*np.cos(t2),r2*np.sin(t2)])

    x = np.concatenate((x1.reshape(-1,1),x2.reshape(-1,1)),axis=-1)
    y = np.ones([200,]) # create labels
    y[:100] = -1

    return x, y
