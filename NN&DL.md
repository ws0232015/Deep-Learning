&copy;Copyright 2017 Shuang Wu     
cite from the Neural Networks and Deep Learning book by Michael Nielsen     
Learning notes

## NN & DL

# Ch 1

## Using NN to recognize handwritten
Human vision involves entire series of visual cortices - V1, V2... V5, doing complex image processing. We are stupendously, astoundingly good at making sense of what our eyes show us, but done unconscilously.      
![handW1](https://github.com/ws0232015/Deep-Learning/blob/master/imgs/handW1.jpg)
Computer recognize is more difficult. The shape of "9" which include a loop and a top is hard to transfer to algorithm. When try use this rule, always hopeless.       
Idea for NN is take large number of handwritten digits, training examples, and then develop a system learn from training examples. NN use examples automatically infer rules to do the recogniz. Increase the # of examples will increase the accuracy.      
![handW2](https://github.com/ws0232015/Deep-Learning/blob/master/imgs/handW2.jpg)
NN are used by banks to process cheques and post offices to recognize addresses.          
Handwritting reognition is a good prototype example for learning NN in general. And can also apply to speech, natural language processing, etc..     

2 Important types of artificial neuron:
  1. Perceptron
  2. Sigmoid neuron
NN Standard learning algorithm: stochastic gradient descent.

### Percetron
Perceptron takes binary inputs and produces a single binary output:     
![percep1](https://github.com/ws0232015/Deep-Learning/blob/master/imgs/percep1.jpg)
The neuron's output, 0 or 1, is determined by the weighted sum 
![My Formula](http://latex.codecogs.com/gif.latex?\Sigma_jw_jx_j)
is less than or greater than some threshold value, 0 or 1.       
![percep2](https://github.com/ws0232015/Deep-Learning/blob/master/imgs/percep2.jpg)
By varying the weights and the threshold, we can get diff. models of decision-making.      
Perceptron isn't a complete model of human decision-making, but a complex networks of perceptrons could make quite subtle decisions:     
![percep3](https://github.com/ws0232015/Deep-Learning/blob/master/imgs/percep3.jpg)
The 1st Layer making three decisions. 2nd making four decisions by weighing up the results from the 1st layer. 2nd will make more complex and abstract decisions than 1st layer. 3rd layer make more complex decision than 2nd. Many-layer network of percetrons can engage in sophisticated decision making.

Math:    

![My Formula](http://latex.codecogs.com/gif.latex?w\cdot{x}\equiv\Sigma_jw_jx_j)        
w and x are vectors for weights and inputs.

![My Formula](http://latex.codecogs.com/gif.latex?b\equiv{-threshold})     
b is perceptron's bias

![My Formula](http://latex.codecogs.com/gif.latex?{output}={\{}^{0\quad%20ifw\cdotx+b\leq0}_{1\quad%20ifw\cdotx+b%3E0})       

NAND, not AND:      
A perceptron with weights -2, -2 and a bias 3:
  1. when input 00, (-2) * 0 + (-2) * 0 + 3 = 3, positive 1. not (F&F) = T
  2. when input 10 or 01, (-2) * 1 + (-2) * 0 + 3 = 1, positive 1. not (F&T) = T
  3. when input 11, (-2) * 1 + (-2) * 0 + 3 = -1, negative 0. not (T&T) = F
Can use perceptron to compute simple logical functions. Can conput any logical function. NAND gates are universal for computtion, same as perceptrons.     

NAND to bitwise sum:
![percep4](https://github.com/ws0232015/Deep-Learning/blob/master/imgs/percep4.jpg)
![percep5](https://github.com/ws0232015/Deep-Learning/blob/master/imgs/percep5.jpg)
![percep6](https://github.com/ws0232015/Deep-Learning/blob/master/imgs/percep6.jpg)

![My Formula](http://latex.codecogs.com/gif.latex?\sigma)
