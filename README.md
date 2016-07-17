# back-propagation
Simple back-propagation algorithm for recognizing hand-written digits(MNIST).

# Description
This is a simple back-propagation algorithm. In this program, you can use an Multi-Layer Perceptron(MLP) with single-hidden-layer.
You can change some defined values of the architecture as following:
* LEARN...Training Epoch  
* SAMPLE...Number of Training Data Set  
* SAMPLE2...Number of Test Data Set  
* INPUT...Number of Input Layer Unit  
* HIDDEN...Number of Hidden Layer Unit  
* OUTPUT...Number of Output Layer Unit  
* ALPHA...Learning Rate  

# Requirement
You have to download "MNIST" dataset files that shown below:  
+ train-images-idx3-ubyte (training set images)  
+ train-labels-idx1-ubyte (training set labels)  
+ t10k-images-idx3-ubyte (test set images)  
+ t10k-labels-idx1-ubyte (test set labels)  
=> [MNIST](http://yann.lecun.com/exdb/mnist/)  
Then you have to make the directory '/MNIST/' in the same directory as '.cpp' files and move these above files to the directory '/MNIST'.  
