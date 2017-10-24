&copy;Copyright for [Shuang Wu] [2017]<br>
Cite from the [coursera] named [Neural network for Machine Learning] from [University of Toronto]<br>
Learning notes<br>
- [Overview of mini-batch gradient descent](#overview-of-mini-batch-gradient-descent)
    - [Reminder: The error surface for a linear neuron](#reminder-the-error-surface-for-a-linear-neuron)
    - [Convergence speed of full batch learning when the error surface is quadratic bowl](#convergence-speed-of-full-batch-learning-when-the-error-surface-is-quadratic-bowl)
    - [How the learning goes wrong](#how-the-learning-goes-wrong)
    - [Stochastic gradient descent](#stochastic-gradient-descent)
    - [Two types of learning algorithm](#two-types-of-learning-algorithm)
    - [A basic mini-batch gradient descent algorithm](#a-basic-mini-batch-gradient-descent-algorithm)
- [A bag of tricks for mini-batch gradient descent](#a-bag-of-tricks-for-mini-batch-gradient-descent)
    - [Be careful about turning down the learning rate](#be-careful-about-turning-down-the-learning-rate)
    - [Initializing the weights](#initializing-the-weights)
    - [Shifting the inputs](#shifting-the-inputs)
    - [Scaling the inputs](#scaling-the-inputs)
    - [A more thorough method: Decorrelate the input components](#a-more-thorough-method-decorrelate-the-input-components)
    - [Common problems that occur in multilayer networks](#common-problems-that-occur-in-multilayer-networks)
    - [4 ways to speed up mini-batch learning](#4-ways-to-speed-up-mini-batch-learning)
- [The momentum method](#the-momentum-method)
    - [The intuition behind the momentum method](#the-intuition-behind-the-momentum-method)
    - [The equations of the momentum method](#the-equations-of-the-momentum-method)

# Overview of mini-batch gradient descent

## Reminder: The error surface for a linear neuron
- The error surface lies in a space w/ a horizontal axis for each weight and one vertical axis for the error
    - For a linear neuron w/ a squared error, it is a quadratic bowl
    - Vertical cross-sections are ellipses
    - Horizontal cross-sections are ellipses
- For multi-layer, non-linear nets the error surface is much more complicated
    - but locally, a piece of a quadratic bowl is usually a very good approximation
- ![img1](imgs/img1.jpg)

## Convergence speed of full batch learning when the error surface is quadratic bowl
- Going downhill reduces the error, but the direction of steepest descent does not point at the minimum unless the ellipse is a circle
    - The gradient is big in the direction in which we only want to travel a small distance
    - The gradient is small in the direction in which we want to travel a large distance
    - ![img2](imgs/img2.jpg)
        - even for non-linear multi-layer nets, the error surface is locally quadratic, so the same speed issues apply   

## How the learning goes wrong
- If the learning rate is big, the weights slosh to and fro across the ravine
    - if the learning rate is too big, this oscillation diverges
- what we would like to achieve
    - move quickly in directions w/ small but consistent gradients
    - move slowly in directions w/ big but inconsistent gradients
- ![img3](imgs/img3.jpg)

## Stochastic gradient descent
- If the dataset is highly redundant, the gradient on the first half is almost identical to the gradient on the second half
    - so instead of computing the full gradient, update the weights using the gradient on the first half and then get a gradient for the new weights on the second half
    - the extreme version of this approach updates weights after each case. Its called "online"
- Mini-batches are usually better than online
    - less computation is used updating the weights
    - computing the gradient for many cases simultaneously uses matrix-matrix multiplies which are very efficient, especially on GPUs
- Mini-batches need to be balanced for classes

## Two types of learning algorithm
- If we use the full gradient computed from all the training cases, there are many clever ways to speed up learning (e.g. non-linear conjugate gradient)
    - The optimization community has studied the general problem of optimizing smooth non-linear functions for many years
    - Multilayer neural nets are not typical of the problems they study so their methods may need a lot of adaptation
- For large neural networks with very large and highly redundant training sets, it is nearly always best to use mini-batch learning
    - The mini-batches may need to be quite big when adapting fancy methods
    - Big mini-batches are more computationally efficient

## A basic mini-batch gradient descent algorithm
- Guess an initial learning rate
    - if the error keeps getting worse or oscillates wildly, reduce the learning rate
    - if the error is falling fairly consistently but slowly, increase the learning rate
- write a simple program to automate this way of adjusting the learning rate
- Towards the end of mini-batch learning it nearly always helps to turn down the learning rate
    - this removes fluctuations in the final weights caused by the variations between mini-batches
- turn down the learning rate when the error stops decreasing
    - use the error on a separate validation set


# A bag of tricks for mini-batch gradient descent

## Be careful about turning down the learning rate

- turning down the learning rate reduces the random fluctuations in the error due to the different gradients on different mini-batches
    - so we get a quick win
    - but then we get slower learning
- don't turn down the learning rate too soom
- ![img4](imgs/img4.jpg)

## Initializing the weights

- If two hidden units have exactly the same bias and exactly the same incoming and outgoing weights, they will always get exactly the same gradient
    -   so they can never learn to be different features
    -   we break symmetry by initializing the weights to have small random values
- If a hidden unit has a big fan-in, small changes on many of its incoming weights can cause the learning to overshoot
    - we generally want smaller incoming weights when the fan-in is big, so initialize the weights to be proportional to sqrt(fan-in)
- we can also scale the learning rate the same way

## Shifting the inputs
- when using steepest descent, shifting the input values makes a big difference
    - it usually helps to transform each component of the input vector so that it has zero mean over the whole training set
- The hyperbolic tangent (which is 2*logistic-1) produces hidden activations that are roughly zero mean
    - in this respect its better than the logistic
- ![img5](imgs/img5.jpg)

## Scaling the inputs
- when using steepest descent, scaling the input values makes a big difference
    - It usually helps to transform each component of the input vector so that it has unit variance over the whole training set
- ![img6](imgs/img6.jpg)

## A more thorough method: Decorrelate the input components
- For a linear neuron, we get a big win by decorrelating each component of the input from the other input components
- There are several different ways to decorrelate inputs. A reasonable method is to use Principal Components Analysis
    - Drop the principal components with the smallest eigenvalues
        - this achieves some dimensionality reduction
    - Divide the remaining principal components by the equare roots of their eigenvalues. For a linear neuron, this converts an axis aligned elliptical error surface into a circular one
- For a circular error surface, the gradient points straight towards the minimum

## Common problems that occur in multilayer networks
- If we start w/ a very big learning rate, the weights of each hidden unit will all become very big and positive or very big and negative
    -   The error derivatives for the hidden units will all become tiny and the error will not decrease
    -   this is usually a plateau, but people often mistake it for a local minimum
- In classification networks that use a squared error or a cross-entropy error, the best guessing strategy is to make each output unit always produce an output equal to the proportion of time it should be a 1
    - The network finds this strategy quickly and may take a long time to improve on it by making use of the input
    - This is another plateau that looks like a local minimum

## 4 ways to speed up mini-batch learning

- use "momentum"
    - instead of using the gradient to change the position of the weight "particle", use it to change the velocity
- use separate adaptive learning rates for each parameter
    - slowly adjust the rate using the consistency of the gradient for that parameter
- rmsprop: Divide the learning rate for a weight by a running average of the magnitudes of recent gradients for that weight
    - this is the mini-batch version of just using the sign of the gradient
- Take a fancy method from the optimization literature that makes use of curvature information (not this lecture)
    - Adapt it to work for NN
    - Adapt it to work for mini-batches

# The momentum method

## The intuition behind the momentum method
- Imagine a ball on the error surface. The location of the ball in the horizontal plane represents the weight vector
    - the ball starts off by following the gradient, but once it has velocity, it no longer does steepest descent
    - its momentum makes it keep going in the previous direction
- it damps oscillations in directions of high curvature by cobining gradients w/ opposite sign
- it builds up spped in directions w/ a gentle but consistent gradientee
- ![img7](imgs/img7.jpg)

## The equations of the momentum method
- $$v(t)=\alpha v(t)-\epsilon\frac{\partial E}{\partial w}(t)$$
    - The effect of the gradient is to increment the previous velocity. The velocity also decays by $\alpha$ which is slightly less than 1
- $$\Delta w(t)=v(t)$$
    - The weight change is equal to the current velocity
    - $$=\alpha v(t)-\epsilon\frac{\partial E}{\partial w}(t)$$
    - $$=\alpha \Delta w(t)-\epsilon\frac{\partial E}{\partial w}(t)$$
        - The weight change can be expressed in terms of the previous weight change and the current gradient
