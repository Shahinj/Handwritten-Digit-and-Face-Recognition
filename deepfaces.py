from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
# from scipy.misc.pilutil import imread
# from scipy.misc.pilutil import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import time

import pickle as cPickle

import os
from scipy.io import loadmat

###
os.chdir("C:/Users/Shahin/Documents/School/Skule/Year 3 - Robo/second semester/csc411/Project_2_handwritten_digits/")
###

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

#Display the 150-th "5" digit from the training set
# imshow(M["train4"][150].reshape((28,28)), cmap=cm.gray)
# show()


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
    
def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output
    
def NLL(y, y_):
    '''
    y is p, what the network is giving us
    y_ is the actual y, what we want the network to give us
    '''
    return -sum(y_*log(y)) 

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, dCdL1.T ) 
    
##############################my code
PIC_SIZE = 28
LBL_SIZE = 10

def compute_o(x,w,b):
    '''
    input: xs and weights and biases
    output: the output of the network
    '''
    return np.matmul(w,x) + b
    
def p2(x,w,b):
    '''
    input: xs weights and biases
    output: output of the network, with softmax
    '''
    o = compute_o(x,w,b)
    res = softmax(o)
    return res
    
def gradient_b(x,y,w,b):
    '''
    input: xs and ys, weights and biases
    output: derivative w.r.t biases
    '''
    o_s = compute_o(x,w,b)
    p = softmax(o_s)
    p_y = p-y
    grad = np.sum(p_y, axis = 1)
    return grad
  
def gradient_w(x,y,w,b):
    '''
    input: xs and ys, weights and biases
    output: derivative w.r.t weights
    y is the label of the training set, each case is a column vector
    x is the training set, each case is a column vector
    '''
    o_s = compute_o(x,w,b)
    p = softmax(o_s)
    p_y = p - y
    grad = np.matmul(p_y,x.T)
    return grad
  
def finite_dif(func,xs,ys,w,h):
    '''
    finite difference computation of the gradient
    input: function and h
    output: gradient
    '''
    gradient = np.zeros((thetas.shape))
    for i in range(w.shape[0]):
        new_w = w.copy()
        new_w[i,:] += h
        u1 = func(xs,ys,new_w)
        u2 = func(xs,ys,w)
        derivative = (func(xs,ys,new_w) - func(xs,ys,w)) / float(h)
        gradient[i,:] = derivative
    return gradient
    
def grad_descent(dfw, dfb, x, y, init_w, init_b, alpha):
    '''
    input: derivative functions, xs and ys, initial weights and biases, learning rate
    output: optimized weights and biases, using gradient descent with momentum
    '''
    EPS = 1e-5   #EPS = 10**(-5)
    prev_w = init_w-10*EPS
    w = init_w.copy()
    prev_b = init_b-10*EPS
    b = init_b.copy()
    max_iter = 3000
    iter  = 0
    sub_w = np.zeros(init_w.shape)
    sub_b = np.zeros(init_b.shape)
    
    while (norm(w - prev_w) >  EPS or norm(b - prev_b) > EPS) and iter < max_iter:
        prev_w = w.copy()
        prev_b = b.copy()
        sub_w = alpha*dfw(x, y, w, b).reshape(sub_w.shape)
        sub_b = alpha*dfb(x, y, w, b).reshape(sub_b.shape)
        w -= sub_w
        b -= sub_b
        iter += 1
    return w, b        #w is the fitted weights
    
def grad_descent_param(dfw, dfb, x, y, init_w, init_b, alpha, iter):
    '''
    input: derivative functions, xs and ys, initial weights and biases, learning rate and max iter
    output: optimized weights and biases, using vanilla descent
    this is the same function as grad_descent, except it takes one more argument, maximum iterations
    '''
    EPS = 1e-5   #EPS = 10**(-5)
    prev_w = init_w-10*EPS
    w = init_w.copy()
    prev_b = init_b-10*EPS
    b = init_b.copy()
    max_iter = iter
    iter  = 0
    sub_w = np.zeros(init_w.shape)
    sub_b = np.zeros(init_b.shape)
    
    while (norm(w - prev_w) >  EPS or norm(b - prev_b) > EPS) and iter < max_iter:
        prev_w = w.copy()
        prev_b = b.copy()
        sub_w = alpha*dfw(x, y, w, b).reshape(sub_w.shape)
        sub_b = alpha*dfb(x, y, w, b).reshape(sub_b.shape)
        w -= sub_w
        b -= sub_b
        iter += 1
    return w, b        #w is the fitted weights
    
def train(training_set, label_set):
    '''
    input: training set
    output: the matrices of w and b for network
    '''
    init_w = np.zeros((LBL_SIZE,PIC_SIZE*PIC_SIZE))
    init_b = np.zeros((LBL_SIZE,1))
    learning_rate = 0.0001
    x = training_set
    y = label_set
    weights, bias = grad_descent(gradient_w, gradient_b, x, y,init_w, init_b, learning_rate)
    return weights, bias
    
def train_momentum(training_set, label_set, beta):
    '''
    input: training set
    output: the matrices of w and b for network
    '''
    init_w = np.zeros((LBL_SIZE,PIC_SIZE*PIC_SIZE))
    init_b = np.zeros((LBL_SIZE,1))
    learning_rate = 0.0001
    x = training_set
    y = label_set
    weights, bias = grad_descent_momentum(gradient_w, gradient_b, x, y,init_w, init_b, learning_rate, beta)
    return weights, bias
    
    

def train_param(training_set, label_set, alpha, iterations):
    '''
    input: training set
    output: the matrices of w and b for network
    this is the same function as train, except this will take alpha annd iterations as well
    '''
    init_w = np.zeros((LBL_SIZE,PIC_SIZE*PIC_SIZE))
    init_b = np.zeros((LBL_SIZE,1))
    learning_rate = alpha
    x = training_set
    y = label_set
    weights, bias = grad_descent_param(gradient_w, gradient_b, x, y,init_w, init_b, learning_rate,iterations)
    return weights, bias
    


def get_all_training_valid__test_set():
    '''
    input : nothing
    output: the training set with 80% of available training images, validation set with 20% of available training images, test set with all of the available test images
    some images are excluded (10) to make sure there are no overlaps
    '''
    
    train_label = ["train0","train1","train2","train3","train4","train5","train6","train7","train8","train9"]
    test_label = ["test0","test1","test2","test3","test4","test5","test6","test7","test8","test9"]
    i = 0
    train_set = np.array([[]])
    train_set_label = np.array([[]])
    valid_set = np.array([[]])
    valid_set_label = np.array([[]])
    for digit in train_label:
        up_t = int(round(0.8*M[digit].shape[0])) - 10
        low_v = up_t + 10
        size_v = M[digit].shape[0]-low_v
        x = np.zeros([PIC_SIZE*PIC_SIZE,up_t])
        labels_t = np.zeros([LBL_SIZE,up_t])
        v = np.zeros([PIC_SIZE*PIC_SIZE,size_v])
        labels_v = np.zeros([LBL_SIZE,size_v])
        
        np.random.seed(0)
        indices_t = np.random.random_integers(up_t, size = x.shape[1])
        np.random.seed(0)
        indices_v = np.random.random_integers(low_v,M[digit].shape[0]-1, size = v.shape[1])
        j = 0
        for index in indices_t:
            x[:,j:j+1] = M[digit][index].T.reshape((PIC_SIZE*PIC_SIZE,1)) / 255.0
            j += 1
        j = 0
        for index in indices_v:
            v[:,j:j+1] = M[digit][index].T.reshape((PIC_SIZE*PIC_SIZE,1)) / 255.0
            j += 1                
        labels_t[i,:] = 1
        labels_v[i,:] = 1
        i += 1
        if digit == 'train0':
            train_set = x
            train_set_label = labels_t
            valid_set = v
            valid_set_label = labels_v
        else:
            train_set = np.hstack((train_set, x))
            train_set_label = np.hstack((train_set_label, labels_t))
            valid_set = np.hstack((valid_set, v))
            valid_set_label = np.hstack((valid_set_label, labels_v))
        
    i = 0
    test_set = np.array([[]])
    test_set_label = np.array([[]])
    for digit in test_label:
        up = M[digit].shape[0]
        np.random.seed(0)
        test = np.zeros([PIC_SIZE*PIC_SIZE,up])
        labels_test = np.zeros([LBL_SIZE,up])
        indices_test = np.random.random_integers(up-1, size = test.shape[1])
        j = 0
        for index in indices_test:
            test[:,j:j+1] = M[digit][index].T.reshape((PIC_SIZE*PIC_SIZE,1)) / 255.0
            j += 1
        labels_test[i,:] = 1
        i += 1
        if digit == 'test0':
            test_set = test
            test_set_label = labels_test
        else:
            test_set = np.hstack((test_set, test))
            test_set_label = np.hstack((test_set_label, labels_test))
            

    return train_set,train_set_label,valid_set, valid_set_label, test_set,test_set_label


def get_training_set(train_set_size):
    '''
    input: desired size of the training set
    output: the training set
    '''
    np.random.seed(0)
    indices = np.random.random_integers(1000, size = (train_set_size))
    train_label = ["train0","train1","train2","train3","train4","train5","train6","train7","train8","train9"]
    x = np.zeros([PIC_SIZE*PIC_SIZE,train_set_size*len(train_label)])
    labels = np.zeros([LBL_SIZE,train_set_size*len(train_label)])
    i = 0
    j = 0
    for digit in train_label:
        for index in indices:
            x[:,j:j+1] = M[digit][index].T.reshape((PIC_SIZE*PIC_SIZE,1)) / 255.0
            j += 1
        labels[i,train_set_size*(i):train_set_size*(i+1)] = 1
        i += 1
    return x, labels
    
def get_valid_set(valid_set_size):
    '''
    input: desired size of the validation set
    output: the validation set
    '''
    np.random.seed(0)
    indices = np.random.random_integers(1001,2000, size = (valid_set_size))
    valid_label = ["train0","train1","train2","train3","train4","train5","train6","train7","train8","train9"]
    x = np.zeros([PIC_SIZE*PIC_SIZE,valid_set_size*len(valid_label)])
    labels = np.zeros([LBL_SIZE,valid_set_size*len(valid_label)])
    i = 0
    j = 0
    for digit in valid_label:
        for index in indices:
            x[:,j:j+1] = M[digit][index].T.reshape((PIC_SIZE*PIC_SIZE,1)) / 255.0
            j += 1
        labels[i,valid_set_size*(i):valid_set_size*(i+1)] = 1
        i += 1
    return x, labels
  
def get_figures_1():
    '''
    input : none
    output: 12 random images of each digit
    '''
    np.random.seed(0)
    indices = np.random.random_integers(5000, size = (12))
    train_label = ["train0","train1","train2","train3","train4","train5","train6","train7","train8","train9"]
    for digit in train_label:
        plt.figure()
        
        for i in range(0,12):
            plt.subplot(3,4, i+1)
            plt.imshow(M[digit][indices[i]].reshape((28,28)), cmap=cm.gray)
            
        plt.suptitle(digit)
        plt.savefig(digit, bbox_inches='tight')
        plt.show()
        
def get_test_set(test_set_size):
    '''
    input: desired size of the test set
    output: the test set
    '''
    np.random.seed(0)
    indices = np.random.random_integers( size = (test_set_size))
    test_label = ["test0","test1","test2","test3","test4","test5","test6","test7","test8","test9"]
    x = np.zeros([PIC_SIZE*PIC_SIZE,test_set_size*len(test_label)])
    labels = np.zeros([LBL_SIZE,test_set_size*len(test_label)])
    i = 0
    j = 0
    for digit in test_label:
        for index in indices:
            x[:,j:j+1] = M[digit][index].T.reshape((PIC_SIZE*PIC_SIZE,1)) / 255.0
            j += 1
        labels[i,test_set_size*(i):test_set_size*(i+1)] = 1
        i += 1
    return x, labels
        
def test_performance(x, label, w, b):
    '''
    input: a set and its labels, the weight matrix and bias matrix
    output: percentage correct classified
    '''
    correct = 0
    total = 0
    for i in range(x.shape[1]):
        o_test = np.matmul(w,x[:,i:i+1]) + b
        y_test = label[:,i]
        if (argmax(o_test) == argmax(y_test)):
            correct += 1
        total += 1
        
    return correct/float(total)*100
    
def plot_learning_curve(momentum='off'):
    '''
    input: (optional) learning curve of gradient descent with momentum
    output: the learning curve of gradient descent, using all the images
    '''
    
    train_set,train_set_label,valid_set, valid_set_label, test_set,test_set_label = get_all_training_valid__test_set()

    init_w = np.zeros((LBL_SIZE,PIC_SIZE*PIC_SIZE))
    init_b = np.zeros((LBL_SIZE,1))
    alpha = 0.0001
    beta = 0.9
    x = train_set
    y = train_set_label
    
    EPS = 1e-5   #EPS = 10**(-5)
    prev_w = init_w-10*EPS
    w = init_w.copy()
    prev_b = init_b-10*EPS
    b = init_b.copy()
    max_iter = 3000
    iter  = 0
    sub_w = np.zeros(init_w.shape)
    sub_b = np.zeros(init_b.shape)
    v_w = 0
    v_b = 0
    performance = np.zeros((max_iter,4))
    i = 0
    prev_cost = None
    roc = 0
    flag_inc = False
    if(momentum == 'off'):
        while (norm(w - prev_w) >  EPS or norm(b - prev_b) > EPS) and iter < max_iter:
            prev_w = w.copy()
            prev_b = b.copy()
            sub_w = alpha*gradient_w(x, y, w, b).reshape(sub_w.shape)
            sub_b = alpha*gradient_b(x, y, w, b).reshape(sub_b.shape)
            w -= sub_w
            b -= sub_b
            performance[i:i+1,0] = iter
            performance[i:i+1,1] = NLL(softmax(compute_o(train_set,w,b)),train_set_label)/train_set.shape[1]
            performance[i:i+1,2] = NLL(softmax(compute_o(valid_set,w,b)),valid_set_label)/valid_set.shape[1]
            performance[i:i+1,3] = NLL(softmax(compute_o(test_set,w,b)),test_set_label)/test_set.shape[1]
            iter += 1
            if flag_inc == False and prev_cost != None and NLL(softmax(compute_o(valid_set,w,b)),valid_set_label)/valid_set.shape[1] > prev_cost:
                roc = iter
                flag_inc = True
            else:
                prev_cost = NLL(softmax(compute_o(valid_set,w,b)),valid_set_label)/valid_set.shape[1]
            i += 1
    else:
        while (norm(w - prev_w) >  EPS or norm(b - prev_b) > EPS) and iter < max_iter:
            prev_w = w.copy()
            sub_w = gradient_w(x, y, w, b).reshape(sub_w.shape)
            prev_v_w = v_w
            v_w = beta * prev_v_w + alpha * sub_w
            w -= v_w
            prev_b = b
            sub_b = gradient_b(x, y, w, b).reshape(sub_b.shape)
            prev_v_b = v_b
            v_b = beta * prev_v_b + alpha * sub_b
            b -= v_b
            performance[i:i+1,0] = iter
            performance[i:i+1,1] = NLL(softmax(compute_o(train_set,w,b)),train_set_label)/train_set.shape[1]
            performance[i:i+1,2] = NLL(softmax(compute_o(valid_set,w,b)),valid_set_label)/valid_set.shape[1]
            performance[i:i+1,3] = NLL(softmax(compute_o(test_set,w,b)),test_set_label)/test_set.shape[1]
            iter += 1
            if flag_inc == False and prev_cost != None and NLL(softmax(compute_o(valid_set,w,b)),valid_set_label)/valid_set.shape[1] > prev_cost:
                roc = iter
                flag_inc = True
            else:
                prev_cost = NLL(softmax(compute_o(valid_set,w,b)),valid_set_label)/valid_set.shape[1]
            i += 1
            
    #need to plot
    plt.figure()
    plt.plot(performance[:,0],performance[:,1])   #train set plot
    plt.plot(performance[:,0],performance[:,2])   #valid set plot
    plt.plot(performance[:,0],performance[:,3])   #test set plot
    plt.plot(roc*np.ones(performance[:,0].shape),performance[:,1], '--')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend(['Cost of training set', 'Cost of validation set','Cost of test set','Point where cost goes up: ' + str(roc)])
    plt.show()

    return performance

def disp_weights(w):
    '''
    input: the weight matrix
    output: visualization of the weights 
    '''
    for i in range(10):
        vis = w[i:i+1,:].reshape((PIC_SIZE,PIC_SIZE))
        figure()
        imshow(vis, cmap = cm.gray)
        title("Visualized weights for " + str(i))
    show()
    
def get_optimum_param():
    '''
    input: none
    output: an array with the performances of different learning rates and max iterations
    '''
    size_per_digit = 100
    test_set_size = 50
    test_set, test_set_label = get_test_set(test_set_size)
    train_set, train_set_label = get_training_set(size_per_digit)
    alphas = [0.01, 0.001, 0.0001, 0.00001]
    iterations = [100, 1000, 10000, 100000]
    performance = np.zeros((len(alphas)*len(iterations),5))
    i = 0
    for l_r in alphas:
        for iter in iterations:
            start_time = time.time()
            w, b  = train_param(train_set, train_set_label, l_r, iter)
            performance[i,2] = float(time.time() - start_time)
            performance[i,0] = l_r
            performance[i,1] = iter
            performance[i,3] = test_performance(train_set, train_set_label, w, b)
            performance[i,4] = test_performance(test_set, test_set_label, w, b)
            i += 1
            print('done 1 nested loop')
            print(performance)
    return performance
    
def get_optimum_beta():
    '''
    input: none
    output: an array with the performances of different betas
    '''
    size_per_digit = 100
    test_set_size = 50
    test_set, test_set_label = get_test_set(test_set_size)
    train_set, train_set_label = get_training_set(size_per_digit)
    betas = [0.9, 0.99, 0.999, 0.9999]
    performance = np.zeros((len(betas),4))
    i = 0
    for beta in betas:
        start_time = time.time()
        w, b  = train_momentum(train_set, train_set_label, beta)
        performance[i,1] = float(time.time() - start_time)
        performance[i,0] = beta
        performance[i,2] = test_performance(train_set, train_set_label, w, b)
        performance[i,3] = test_performance(test_set, test_set_label, w, b)
        print performance
        i += 1
    return performance
    
def grad_descent_momentum(dfw, dfb, x, y, init_w, init_b, alpha, beta):
    '''
    input: derivative functions, xs and ys, initial weights and biases, learning rate and momentum term
    output: optimized weights and biases, using gradient descent with momentum
    '''
    EPS = 1e-5   #EPS = 10**(-5)
    max_iter = 3000
    iter  = 0
    
    prev_w = init_w-10*EPS
    w = init_w.copy()
    sub_w = np.zeros(init_w.shape)
    v_w = 0
    
    prev_b = init_b-10*EPS
    b = init_b.copy()
    sub_b = np.zeros(init_b.shape)
    v_b = 0
    
    while (norm(w - prev_w) >  EPS or norm(b - prev_b) > EPS) and iter < max_iter:
        prev_w = w.copy()
        sub_w = dfw(x, y, w, b).reshape(sub_w.shape)
        prev_v_w = v_w
        v_w = beta * prev_v_w + alpha * sub_w
        w -= v_w
        
        prev_b = b
        sub_b = dfb(x, y, w, b).reshape(sub_b.shape)
        prev_v_b = v_b
        v_b = beta * prev_v_b + alpha * sub_b
        b -= v_b
        
        iter += 1
    return w, b   
    
def plot_contour(disp='off'):
    '''
    input: (optional) display the contour or no
    output: w1 and w2 axis of the plot, z of the plot (contour)
    it also saves w1 and w2 and z into numpy files
    THIS TOOK 5 HOURS TO RUN, TEST ON YOUR OWN RISK (PATIENCE)
    '''
    train_set, train_set_label = get_training_set(400)
    w, b  = train(train_set, train_set_label)
    w1_pos = (5,13*14)      #weight 1 going to number 5, best near 0.7
    w2_pos = (5,15*14)      #weight 1 going to number 5, best near 1.1
    w1_list = np.arange(-10,10,0.01)
    w2_list = np.arange(-10,10,0.01)
    w1z , w2z = np.meshgrid(w1_list,w2_list)
    w_copy = w.copy()
    cost_data = np.zeros((len(w2_list),len(w1_list)))
    for i,w1 in enumerate(w1_list):
        for j,w2 in enumerate(w2_list):
            w_copy[w1_pos[0],w1_pos[1]] = w1
            w_copy[w2_pos[0],w2_pos[1]] = w2
            cost = get_diff_loss(train_set,train_set_label,b,w,w_copy)
            cost_data[j,i] = cost
            print 'w1: ' + str(w1) + ', w2: ' + str(w2) + ', Cost: ' + str(cost)
    z = cost_data
    if disp ==  'on':
        figure()
        CS = contour(w1z, w2z, z,levels = np.arange(z.min(),z.max(),abs(z.max()-z.min())/25))
        clabel(CS, inline=1, fontsize=10)
        title('Contour plot')
        xlabel('w1')
        ylabel('w2')
        show()
        
    i = 0
    saved = False
    while (saved == False):
        string_w1 = 'contour_w1_v'+str(i)+'.npy'
        string_w2 = 'contour_w2_v'+str(i)+'.npy'
        string_cost = 'contour_cost_v'+str(i)+'.npy'
        try:
            temp = np.load(string_w1)
            i += 1
        except IOError:
            np.save(string_w1[:-4],w1z)
            np.save(string_w2[:-4],w2z)
            np.save(string_cost[:-4],z)
            saved = True
    
    return w1z, w2z, z
    
    
def grad_descent_k_step(dfw, dfb, x, y, init_w, init_b, alpha, k, w1_pos,w2_pos,w1_val,w2_val,beta,type_of_descent = 'Vanilla'):
    '''
    input: derivative functions, xs and ys, initial weights and biases, learning rate and beta, k steps, location of the weights and their value, type of descent (optional)
    output: w and b, a list of w1 and w2 across the steps
    '''
    EPS = 1e-5   #EPS = 10**(-5)
    prev_w = init_w-10*EPS
    w = init_w.copy()
    prev_b = init_b-10*EPS
    b = init_b.copy()
    max_iter = k
    iter  = 0
    sub_w = np.zeros(init_w.shape)
    sub_b = np.zeros(init_b.shape)
    w_list = [(w1_val,w2_val)]
    
    
    v_w = 0
    v_b = 0

    if type_of_descent == 'Vanilla':
        while iter < max_iter:
            prev_w = w.copy()
            prev_b = b.copy()
            sub_w = alpha*dfw(x, y, w, b).reshape(sub_w.shape)
            sub_b = alpha*dfb(x, y, w, b).reshape(sub_b.shape)
            w -= sub_w
            b -= sub_b
            iter += 1
            w_list.append((w[w1_pos[0],w1_pos[1]],w[w2_pos[0],w2_pos[1]]))
    else:
        while iter < max_iter:
            prev_w = w.copy()
            sub_w = dfw(x, y, w, b).reshape(sub_w.shape)
            prev_v_w = v_w
            v_w = beta * prev_v_w + alpha * sub_w
            w -= v_w
            prev_b = b
            sub_b = dfb(x, y, w, b).reshape(sub_b.shape)
            prev_v_b = v_b
            v_b = beta * prev_v_b + alpha * sub_b
            b -= v_b
            iter += 1
            w_list.append((w[w1_pos[0],w1_pos[1]],w[w2_pos[0],w2_pos[1]]))
        
    return w, b,w_list        #w is the fitted weights

def get_diff_loss(train_set,train_set_label,b,w,w_new):
    '''
    input: training set, w and b, new w
    output: difference between the costs of the old w and new w
    '''
    cost_old = NLL(softmax(compute_o(train_set,w,b)),train_set_label)
    cost_new = NLL(softmax(compute_o(train_set,w_new,b)),train_set_label)
    return abs(cost_new - cost_old)
    
def plot_trajectory(version,read_from_file = 'on'):
    '''
    input: version of the numpy file to read (for the contour)
    output: the trajectory
    '''
    #get training set and optimum weights
    train_set, train_set_label = get_training_set(400)
    w, b  = train(train_set, train_set_label)
    
    w1_pos = (5,13*14)      #weight 1 going to number 5
    w2_pos = (5,15*14)      #weight 2 going to number 5
    
    #initialize weights away from optimum
    # w1_init = w[w1_pos[0],w1_pos[1]]+sign(w[w1_pos[0],w1_pos[1]])*0.03
    # w2_init = w[w2_pos[0],w2_pos[1]]+sign(w[w2_pos[0],w2_pos[1]])*0.03
    print 'optimum w1: ' + str(w[w1_pos[0],w1_pos[1]])
    print 'optimum w2: ' + str(w[w2_pos[0],w2_pos[1]]) 
    w1_init = -0.5
    w2_init = -0.5     
    beta = 0.9
    w_copy = w.copy()
    w_copy[w1_pos[0],w1_pos[1]] = w1_init
    w_copy[w2_pos[0],w2_pos[1]] = w2_init
    w_k,b_k,w_steps = grad_descent_k_step(gradient_w,gradient_b,train_set,train_set_label,w_copy,b,0.001,30,w1_pos,w2_pos,w1_init,w2_init,beta,'Vanilla')
    
    w_k_m,b_k_m,w_steps_m = grad_descent_k_step(gradient_w,gradient_b,train_set,train_set_label,w_copy,b,0.001,30,w1_pos,w2_pos,w1_init,w2_init,beta,'Momentum')
    
    if(read_from_file == 'on'):
        string = 'contour_w1_v' + str(version) + '.npy'
        cnt_w1 = np.load(string)
        string = 'contour_w2_v' + str(version) + '.npy'
        cnt_w2 = np.load(string)
        string = 'contour_cost_v' + str(version) + '.npy'
        cnt_data = np.load(string)
    else:
        cnt_w1,cnt_w2,cnt_data = plot_contour(disp='off')
    
    w_copy = w.copy()
    
    #display
    figure()
    CS = contour(cnt_w1, cnt_w2, cnt_data,levels = np.arange(cnt_data.min(),cnt_data.max(),(cnt_data.max()-cnt_data.min())/25))
    clabel(CS, inline=1, fontsize=10)
    title('Contour plot')
    xlabel('w1')
    ylabel('w2')
    plt.plot([a for a, b in w_steps], [b for a,b in w_steps], 'yo-', label="No Momentum")
    plt.plot([a for a, b in w_steps_m], [b for a,b in w_steps_m], 'gx-', label="Momentum")
    plt.legend(loc='top left')
    show()    
    return 

##############################
    

if (__name__ == "__main__"):
    '''
    #Load sample weights for the multilayer neural network
    snapshot = cPickle.load(open("snapshot50.pkl"))
    W0 = snapshot["W0"]
    b0 = snapshot["b0"].reshape((300,1))
    W1 = snapshot["W1"]
    b1 = snapshot["b1"].reshape((10,1))
    
    #Load one example from the training set, and run it through the
    #neural network
    x = M["train5"][148:149].T    
    L0, L1, output = forward(x, W0, b0, W1, b1)
    #get the index at which the output is the largest
    y = argmax(output)
    
    ################################################################################
    #Code for displaying a feature from the weight matrix mW
    #fig = figure(1)
    #ax = fig.gca()    
    #heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)    
    #fig.colorbar(heatmap, shrink = 0.5, aspect=5)
    #show()
    ################################################################################
    '''
    
    train_set,train_set_label,valid_set, valid_set_label, test_set,test_set_label = get_all_training_valid__test_set()
    w, b  = train(train_set, train_set_label)
    print test_performance(test_set,test_set_label,w,b)
    
    # train_set, train_set_label = get_training_set(50)
    # w, b  = train(train_set, train_set_label)
    # test_set, test_set_label = get_test_set(10)
    # print test_performance(test_set,test_set_label,w,b)
    
    # results = plot_learning_curve()
    # disp_weights()
    # results = get_optimum_param()
    
    # result = get_optimum_beta()
    
    # performance_momentum = plot_learning_curve('on')
    
    # q,w,r = plot_contour('on')
    
    # figure()
    # CS = contour(q, w, r,levels = np.arange(0,1,0.01))
    # clabel(CS, inline=1, fontsize=10)
    # title('Contour plot')
    # xlabel('w2')
    # ylabel('w1')
    # show()
    
    
    # plot_trajectory(0)
    