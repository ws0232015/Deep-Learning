&copy;Copyright 2017 Shuang Wu     
cite from the Neural Networks and Deep Learning book by Michael Nielsen     
Learning notes

# NN & DL

## Ch 1

### Using NN to recognize handwritten
Human vision involves entire series of visual cortices - V1, V2... V5, doing complex image processing. We are stupendously, astoundingly good at making sense of what our eyes show us, but done unconscilously.      
Computer recognize is more difficult. The shape of "9" which include a loop and a top is hard to transfer to algorithm. When try use this rule, always hopeless.       
Idea for NN is take large number of handwritten digits, training examples, and then develop a system learn from training examples. NN use examples automatically infer rules to do the recogniz. Increase the # of examples will increase the accuracy.         
NN are used by banks to process cheques and post offices to recognize addresses.          
Handwritting reognition is a good prototype example for learning NN in general. And can also apply to speech, natural language processing, etc..     

2 Important types of artificial neuron:
  1. Perceptron
  2. Sigmoid neuron
NN Standard learning algorithm: stochastic gradient descent.

#### Percetron


![My Formula](http://latex.codecogs.com/gif.latex?\sigma)
