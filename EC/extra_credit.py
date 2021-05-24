#import libraries
import numpy as np
from MyKernelPerceptron import KernelPerceptron
from visualization import plot_boundary
from simulate_data import generate_data

# Apply kernel perceptron to simulated data
# generate the data
x, y = generate_data()
# define model
clf = KernelPerceptron(x,y,sigma_idx=0)
# training the perceptron
clf.fit(x,y)
# generate the prediction
pred = clf.predict(x)
acc = np.count_nonzero(pred==y)/len(pred)
print('Accuracy on the simulated data is %.2f' %acc)
# generate the plot
plot_boundary(clf,x,y)

#Apply kernel perceptron to digit classification with digits 4 and 9
#read in data.
train_data = np.genfromtxt("optdigits49_train.txt",delimiter=",")
train_x = train_data[:,:-1]
train_y = train_data[:,-1]
test_data = np.genfromtxt("optdigits49_test.txt",delimiter=",")
test_x = test_data[:,:-1]
test_y = test_data[:,-1]
# define model
clf = KernelPerceptron(train_x,train_y,sigma_idx=1)
# training the perceptron
clf.fit(train_x,train_y)
# generate the prediction
pred = clf.predict(test_x)
acc = np.count_nonzero(pred==test_y)/len(pred)
print('Accuracy on the digit classification-49 is %.2f' %acc)


# Apply kernel perceptron to digit classification with digits 7 and 9
# read in data.
train_data = np.genfromtxt("optdigits79_train.txt",delimiter=",")
train_x = train_data[:,:-1]
train_y = train_data[:,-1]
test_data = np.genfromtxt("optdigits79_test.txt",delimiter=",")
test_x = test_data[:,:-1]
test_y = test_data[:,-1]
# define model
clf = KernelPerceptron(train_x,train_y,sigma_idx=2)
# training the perceptron
clf.fit(train_x,train_y)
# generate the prediction
pred = clf.predict(test_x)
acc = np.count_nonzero(pred==test_y)/len(pred)
print('Accuracy on the digit classification-79 is %.2f' %acc)
