import numpy as np
import random
def process_data(data,mean=None,std=None):
    # normalize the data to have zero mean and unit variance (add 1e-15 to std to avoid numerical issue)
    if mean is not None:
        # mean and std is precomputed with the training data
        norm_data = (data - mean)/(std + 1e-15)
        return norm_data
    else:
        # compute the mean and std based on the training data
        mean = std = 0 # placeholder
        mean = np.mean(data, axis=0) # mean col-wise
        std = np.std(data, axis=0) # std col-wise with 1e-15 added
        
        norm_data = (data - mean)/(std + 1e-15)
        
        return norm_data, mean, std

def process_label(label):
    # convert the labels into one-hot vector for training
    one_hot = np.zeros([len(label),10])
    for i in range(0,len(label)):
        for j in range(0,10):
            if j == label[i]:
                one_hot[i][j] = 1
            else:
                one_hot[i][j] = 0 
    return one_hot

def sigmoid(x):
    # implement the sigmoid activation function for hidden layer
    f_x = 1/(1 + np.exp(-x))

    return f_x

def softmax(x):
    # implement the softmax activation function for output layer
    val = np.exp(x)
    f_x = val / val.sum()
    
    return f_x

class MLP:
    def __init__(self,num_hid):
        # initialize the weights
        self.weight_1 = np.random.random([64,num_hid]) #w
        self.bias_1 = np.random.random([1,num_hid]) #w0
        self.weight_2 = np.random.random([num_hid,10]) #v
        self.bias_2 = np.random.random([1,10]) #v0
        self.num_hid = num_hid #added

    def fit(self,train_x,train_y, valid_x, valid_y):
        # learning rate
        lr = 5e-3
        # counter for recording the number of epochs without improvement
        count = 0
        best_valid_acc = 0

        """
        Stop the training if there is no improvment over the best validation accuracy for more than 100 iterations
        """
        while count<=100:
            # training with all samples (full-batch gradient descents)
            # implement the forward pass
            zh = self.get_hidden(train_x)
            y = process_label(self.predict(train_x))
            
            rmy = train_y - y #get r_i minus y_i
            z_der = zh * (1 - zh) #derivative of z
            
            # implement the backward pass (backpropagation)
            # compute the gradients w.r.t. different parameters
            #get delta v
            delta_v = lr * np.dot(zh.T, rmy) #num_hidx10
            delta_v_bias = lr * np.sum(rmy, axis=0)
            
            #get delta_w
            rmy_v = np.dot(rmy,self.weight_2.T)
            prod = (rmy_v*z_der)
            delta_w = lr * np.dot(prod.T,train_x) #num_hidx64, need to transpose when updating
            
            delta_w_bias = lr* np.sum(prod, axis=0) 
            
            #update the parameters based on sum of gradients for all training samples
            self.weight_2 = self.weight_2 + delta_v  #update weight
            self.weight_1 = self.weight_1 + delta_w.T #update weight
            self.bias_2 = self.bias_2 + delta_v_bias  #update bias
            self.bias_1 = self.bias_1 + delta_w_bias #update bias
            
            # evaluate on validation data
            predictions = self.predict(valid_x)
            valid_acc = np.count_nonzero(predictions.reshape(-1)==valid_y.reshape(-1))/len(valid_x)

            # compare the current validation accuracy with the best one
            if valid_acc>best_valid_acc:
                best_valid_acc = valid_acc
                count = 0
            else:
                count += 1

        return best_valid_acc

    def predict(self,x):
        # generate the predicted probability of different classes
        # convert class probability to predicted labels
        
        zh = self.get_hidden(x)
        
        #get o and y
        ot = np.dot(self.weight_2.T,zh.T).T + self.bias_2
        
        y = softmax(ot)
        y_prob = np.argmax(y, axis=1)

            
        return y_prob

    def get_hidden(self,x):
        # # extract the intermediate features computed at the hidden layers (after applying activation function)
        w_T = self.weight_1.T # w transpose
        w_bias = self.bias_1.T
           
        zh=[]
        for t in range(len(x)): #for each sample in x
            #get z using sigmoid    
            z=[]   
            for h in range(self.num_hid): #for each hidden layer
                wtx = np.dot(w_T[h], x[t].T) + w_bias[h]
                z.append(wtx) 
            z=np.asarray(z)
            z_sig = sigmoid(z) # lenx X num_hid
            zh.append(z_sig)
            
        zh = np.asarray(zh).reshape(len(x),self.num_hid) 
        return zh

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2, self.num_hid
