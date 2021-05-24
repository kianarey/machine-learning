import numpy as np
from collections import Counter
class Tree_node:
    """
    Data structure for nodes in the decision-tree
    """
    def __init__(self,):
        self.feature = None # index of the selected feature (for non-leaf node)
        self.class_label = None # class label (for leaf node)
        self.left_child = None # left child node
        self.right_child = None # right child node

class Decision_tree:
    """
    Decision tree with binary features
    """
    def __init__(self,min_entropy):
        self.min_entropy = min_entropy
        self.root = None

    def fit(self,train_x,train_y):
        # construct the decision-tree with recursion
        self.root = generate_tree(train_x,train_y,self.min_entropy)

    def predict(self,test_x):

        prediction = np.zeros([len(test_x),]).astype('int') # placeholder
        
        # iterate through all samples
        for i in range(len(test_x)): #562 samples for test_x
            # traverse the decision-tree based on the features of the current sample
            node = self.root
            
            while node.left_child != None and node.right_child != None: #keep looping until no more children
                if test_x[i][node.feature] == 0.0:
                    node = node.left_child
                else:
                    node = node.right_child
            
            prediction[i] = node.class_label
        return prediction

def generate_tree(data,label,min_entropy):
    # initialize the current tree node
    cur_node = Tree_node()

    # compute the node entropy
    node_entropy = compute_node_entropy(label)

    # base case: determine if the current node is a leaf node
    if node_entropy < min_entropy:
        # determine the class label for leaf node
        classes, count = np.unique(label, return_counts=True)
        cur_node.class_label = classes[np.argmax(count)] #np.argmax returns the index of the class that has the most samples; then take classes[index]

        return cur_node

    # select the feature that will best split the current non-leaf node
    selected_feature = select_feature(data,label) # returns an index of feature (0-64)
    cur_node.feature = selected_feature

    # split the data based on the selected feature and start the next level of recursion    
    left_child_data = []
    left_child_y = []
    
    right_child_data = []
    right_child_y = []
    
    for i in range(len(data)): #loop through all samples
        if data[i][selected_feature] == 0.0: #if this sample is greater than 0, append to left, o.w. right.
            left_child_data.append(data[i])
            left_child_y.append(label[i])
        else:
            right_child_data.append(data[i])
            right_child_y.append(label[i])
    
    
    left_child_data = np.asarray(left_child_data)
    left_child_y = np.asarray(left_child_y)
    
    right_child_data = np.asarray(right_child_data)
    right_child_y = np.asarray(right_child_y)
    
    
    cur_node.left_child = generate_tree(left_child_data, left_child_y, min_entropy)
    cur_node.right_child= generate_tree(right_child_data, right_child_y, min_entropy)
    #print(cur_node.feature)
    return cur_node

def select_feature(data,label):
    # iterate through all features and compute their corresponding entropy
    best_feat = 0
    
    all_entropy = []
    for i in range(len(data[0])): #loop through all 64 features
        # compute the entropy of splitting based on the selected features
        left_y = []
        right_y = []
        for j in range(len(data)): #loop through all samples in each feature
            if data[j][i] == 0.0: #if this sample is greater than 0, append to left, o.w. right.
                left_y.append(label[j])
            else:
                right_y.append(label[j])
        
        cur_entropy = compute_split_entropy(left_y, right_y) # You need to replace the placeholders ('None') with valid inputs
        all_entropy.append(cur_entropy)
    # select the feature with minimum entropy
    best_feat = np.argmin(all_entropy)
    return best_feat

def compute_split_entropy(left_y,right_y):
    # compute the entropy of a potential split, left_y and right_y are labels for the two splits
    total = len(left_y) + len(right_y)
    left = compute_node_entropy(left_y)
    right = compute_node_entropy(right_y)
    
    split_entropy = (len(left_y)/total) * left + (len(right_y)/total) * right
    return split_entropy

def compute_node_entropy(label):
    # compute the entropy of a tree node (add 1e-15 inside the log2 when computing the entropy to prevent numerical issue)
    classes, count = np.unique(label, return_counts=True)
    entropy = np.sum([(-count[i]/np.sum(count))*np.log2(count[i]/np.sum(count) + 1e-15) for i in range(len(classes))])
    return entropy
