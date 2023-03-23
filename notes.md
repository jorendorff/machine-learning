## tensorflow-1-public C1W1 (2023-03-03)

[keras-example.md](tensorflow-1-public/C1/W2/keras-example.md) is my version of the stuff in
[ungraded_lab](tensorflow-1-public/C1/W2/ungraded_lab).

The files on github have an "Open in Colab" button (it's not part of the GH UI,
it's markdown in the file itself).
<https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W1/ungraded_lab/C1_W1_Lab_1_hello_world_nn.ipynb>

Notice:

-   The answer varies from run to run of the entire notebook, but once trained,
    the model always predicts the same value for 10.0. Makes sense.

-   The explanation given in the text (and video) for the answer not being ==
    19 is "given the data that we fed the model with, it calculated that there
    is a very high probability that the relationship between x and y is y=2x-1,
    but with only 6 data points we can't know for sure. As a result, the result
    for 10 is very close to 19, but not necessarily 19." That can't possibly be
    an accurate descriptoin of what's going on. More like: it's just making
    guesses, and the best guess it made got it this close.


Wonder:

-   Why does model.predict take an array of inputs? Why does it return an array
    of arrays?

-   What parameters exist inside the single "neuron" we created? I take it that
    isn't a perceptron; those have boolean output.

    -   What are the implicit assumptions in the model?

    -   Is the model we made linear?

        A: No, the slope varies slightly from point to point. In fact the slope
        at plus or minus 10**5 is 0. (At plus or minus 1000, it's still 1.95.)
        Odd.

-   Why does it take so long for gradient descent to converge here? How is this
    like numeric solvers, which converge to a zero in a shockingly tiny number of
    steps (like 6)? How is it different?

Labs troubleshooting link: https://www.coursera.org/learn/introduction-tensorflow/supplement/M20rh/optional-downloading-your-notebook-and-refreshing-your-workspace



## tensorflow-1-public C1W2 (2023-03-03)

Again, the explanation of what this model is doing could not possibly be
correct. Going to drive me crazy.

Notice:

-   softmax will totally break if the numbers are large... right?

    A: It gives silly answers, yes.

-   softmax is sensitive to the base, e. Larger bases cause larger values to
    soak up more of the pie.
    
-   softmax is sensitive to the scale of the input vector. softmax(data * 2.0) is
    sharper than softmax(data). 
    
    To me, this suggests that there are some assumptions about how you're
    supposed to use this. It is a little weird that I wasn't told to normalize
    the data or something beforehand; like, you could compute the mean and
    standard deviation and scale the thing... hmm.

-   We didn't specify an activation function last time. Does that mean we got
    the identity function?
    
    A: Yes.

-   OK, so I think in the tensorflow worldview, "tensor" is a fancy word for
    "n-dimensional matrix", and I think the idea of tensorflow is to put a nice
    high-level API on these things, backed by a compiler that compiles your
    model down to machine code, taking advantage of the hardware, to make
    training and prediction efficient.
    
    A: This seems to be correct so far.

Wonder:

-   These two activation functions don't affect the shape of the data. Are
    there any that do?

## Logistic Regression (C1W2L02)

https://www.youtube.com/watch?v=hjrYrynGWGA&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=8

He says that it turns out to be easier to implement the bias vector b as a
separate thing rather than add an input that is pegged to 1. I wonder why.

Can I remember the loss function for logistic regression from memory?

He says they don't use the square of the error `sum((yhat - y)**2)` because it
makes the optimization problem non-convex.

    So I think what they use is `avg(-(y log yhat + (1-y) log (1-yhat)))`.


## Explanation of Logistic Regression's Cost Function (C1W2L18)

https://www.youtube.com/watch?v=k_S5fnKjO-4&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=24

This is a bit beyond me. The video is an odd blend of foundational proof and
random hacks.

I don't feel he has defined "cost" and "loss". Seems important. Internet says
they are synonymous.

In this lesson, we start with the assumption, or aspiration, that

    If y = 1:   p(y|x) = \hat{y}

That is, we want y-hat to be the probability that y = 1, given what we know
(the input x). Reasonable enough. And it follows that

    If y = 0:  p(y|x) = 1 - \hat{y}

(Note that here `p(y|x)` denotes the probability that y = 0, given x, so a
different number and in fact the opposite of what the same symbols denoted
above.)

And on the basis of this we make a leap of faith to this continuous function

    p(y=y|x) = \hat{y}^y * (1 - \hat{y})^{1 - y}

It is true that if you set y = 1, you get the first statement, and if you set y
= 0, you get the second (...unless y-hat = y, in which case both are undefined;
also p isn't continuous at those points),

----

But this makes a hash of the meaning I had attached to these symbols. Let me
rewrite all this using function notation for y-hat. First we need to understand
that there are many events, and for each event _x_ has some value, and _y_ is
either 0 or 1. It is not necessarily the case that _y_ is a function of _x_.

Now let _f_ be the function that maps x to y-hat:

    y-hat = f(x)

This is some instance of our model. And let us define another function,
unmotivated for now:

    g(x, y) = f(x)^y * (1 - f(x))^(1 - y)

(And I think what we need to do here is define 0^0 = 1, since the exponent is
not even really a continuous quantity.)

Clearly we can affect this function by making adjustments to _f_.

What happens if we optimize _f_ to maximize this function _g_? (That is, in
practice, we're maximizing the total _g_ across all training data. Abstractly
we could imagine maximizing the integral of _g_ over all events, all pairs
(_x_, _y_).)

Apparently what we are doing, when we optimize like that, is causing f(x) to
approximate p(y=1|x).

Do we benefit from wording it this way, continuous in y which is not actually a
continuous quantity? I don't think so. I think this is nonsense and we might as
well write:

    g(x, 0) = 1 - f(x)
    g(x, 1) = f(x)

----

Going back to the video, Ng has written

    p(y|x) = yhat ^ y * (1 - yhat) ^ (1 - y)

Ng points out that if you optimise the log of this, you will optimize this; and
optimizing the log, I guess, turns out to be easier. So:

    log p(y|x) = y log yhat + (1 - y) log (1 - yhat)

Or as I like to say,

    g(x, 0) = log (1 - f(x))
    g(x, 1) = log f(x)

Since we aim to maximize this, we will adopt the negative of this expression as
our loss function L(yhat, y).

Yeah, now having unraveled it, it looks like a lot of symbol-shuffling for no
benefit. The video has faults of its own: it's not clear which direction we're
reasoning, i.e. what is motivating what here.

----

One point he makes a bit later is that if you want to adjust f to maximize the
probability of the entire training set, what you want to do is maximize the
_product_ of the probabilities of the individual events. Taking the log turns
this into a sum, which is convenient, as sums are the only thing we actually
know how to tune for in practice.

It's all such motivated reasoning.

----

Saw an ad for learn-xpro.mit.edu. Neat stuff, very appealing. I doubt this
course is free.

## Discussion

Neural networks mix calculus and linear algebra in a way I don't understand.

Like, I can take a partial derivative of a function that has several
real-valued inputs and outputs... I can do the same for an expression where the
terms are matrices by "inlining away" the matrices and treating it as a
function with m*n inputs and outputs. But surely an ounce of mathematical
theory would help.

I can't tell if I like neural networks yet.

I *love* the trick of walking backwards through the network computing the
partial derivative of the loss with respect to everything. That's just
fantastic!!

On the other hand every little thing is such a mix of brilliant ideas and the
most hackety hacks

[...]

(graydon:) well, I am led to believe there was some kind of revolution in the
hinton lab around the late 2000s / early 2010s

but it's hard to nail a paper from browsing the timelines because they're often
talking way above my head

I believe https://www.deeplearningbook.org/ might be the most approachable
thing currently? like part 2 seems fairly .. well .. I can _read_ much of it,
which is better than I get out of a lot of material

ok so I _think_ the big-breakthrough set was this hinton paper
https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf followed by this pair by
bengio and lecun
https://www.iro.umontreal.ca/~lisa/pointeurs/bengio+lecun_chapter2007.pdf and
https://www.iro.umontreal.ca/~lisa/pointeurs/dbn_supervised_tr1282.pdf

yeah the first hinton one has 19059 academic citations

it's the watershed paper

http://www.iro.umontreal.ca/~lisa/pointeurs/TR1312.pdf is maybe a better bengio
followup that generalized

perhaps a more accessible summary of the breakthrough period is the most-cited
one, a nature publication in 2015 by all 3 (hinton, bengio, lecun):
http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf

(graydon, earlier:) ok. and the big deep-learning breakthrough was to figure
out some way that I never read the details of that lets you do this training
"layer by layer" such that the total wrongness to subtract can be somehow
decomposed to a single layer's wrongness plus .. uh .. here is where my
understanding falls apart

(me:) As far as I can tell, the trick is unexpectedly simple. Conceptually it
isn't one layer at a time, it seems to me.

Say you've got a neural net you want to train, some training data, and a loss
function. The loss function measures how far your neural net's output lies from
the correct answers.

Assume for a moment that you want to train one particular parameter of your
network, any one parameter from any layer. If you could compute the partial
derivative of loss with respect to that variable, you would know which way to
tweak that parameter to improve the loss. Then just iterate to a local minimum.

Combine this with the concept of gradient from multivariate calculus, and you
can, in each step, tweak all the parameters at once.

The gradient gives you a vector in parameter space that points in the direction
of steepest descent toward minimizing loss.

Take one baby step in that direction, then recompute all the derivatives and
repeat.

The only hard part is computing all those partial derivatives. That is the part
that proceeds backwards layer by layer. This video explains it.
https://www.youtube.com/watch?v=nJyUyKN-XBQ&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=14

Gradient descent only finds a local minimum. The Nature survey paper you posted
has a paragraph about "poor local minima", saying, effectively, "works fine in
practice, what do you want".

## Computing Neural Network Output (C1W3L03)

Oh, the single-neuron logistic regression isn't seen as a neural network
really. It must be a technique that sort of precedes neural networks.

Already knew all this. Skipped a lot.

## Vectorizing Across Multiple Examples, Explanation for Vectorized implementation

Skipping this one.

## Activation Functions

I'd like to understand why this is the case: The video says that for hidden
layers, tanh (which it describes as a shifted version of the sigmoid function)
is a better choice because the output is more likely to be centered at zero,
making learning easier for the next layer. I'm not sure why that should be
true.

As the output of the linear part of a neuron gets very large or very small, the
derivative gets very small, which can slow down gradient descent (I bet it
can!). Hence relu.

Leaky relu: sometimes people make the slope on the left a parameter of the
learning algorithm, but rarely. Seems a little silly to me.

Intuitively I thought one of the nice things about the sigmoid function was
that learning could make it into a sharp step function or could make use of the
nice curved parts -- it seemed versatile. Though maybe that kind of thing
creates concavities in the optimization problem.

## Why Non-linear Activation Functions (C1W3L07)

I already knew this, skipping

## Derivatives of Activation Functoins (C1W3L08)

Did I do the derivative of the sigmoid wrong?

```
g(z) = 1 / (1 + exp(-z))

D g z = - 1 / (1 + exp(-z))**2 * D (\z -> 1 + exp(-z))(z)
      = - 1 / (1 + exp(-z))**2 * - exp(-z)
      = exp(-z) / (1 + exp(-z))**2
      = 1 / (1 + exp(-z)) * (exp(-z) / (1 + exp(-z)))
      = g(z) * (exp(-z) / (1 + exp(-z)))
      = g(z) * (1 + exp(-z) - 1) / (1 + exp(-z))
      = g(z) * (1 - 1/(1 + exp(-z)))
      = g(z) * (1 - g(z))
```

No, I just missed a performance optimization.

```
D tanh z = q - (tanh z)**2
```

I will take this one on faith.

## Backpropagation Intuition (C1W3L10)

Lots of calculus. I did all this calculus for my example program, but it's hard
to verify (using normal software debugging techniques) that I got it right.
Unfortunately it is also hard to do that using the video, as slightly different
choices lead to quite different formulas. I'll leave it for now.

## Random Initialization (C1W3L11)

Hey, I was right. You can't initialize all parameters to zero for exactly the
reason I thought.

I wonder what this looks like in the parameter space, what the geometric
intuition is.

Small random values are better because we want to start in the neighborhood of
the activation function where its slope is not extremely close to zero.

## Forward Propagation in a Deep Network (C1W4L02)

Obvious.

## Getting Matrix Dimensions Right (C1W4L03)

I've already basically been doing all this.

Maybe the reason it's easier to have a separate vector of parameters `b` for
each layer rather than use a fake dimension that's always 1 is that you'd have
to add a 1 on every layer... or else you have to do something to make sure the
parameters preserve that 1 and never change.

## Why Deep Representations? (C1W4L04)

https://www.youtube.com/watch?v=5dWp1mw_XNk&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=39

1.  Each output in a network indicates "recognizing" some feature, and the
    intuition is that shallow features combine to form more sophisticated
    features.

2.  Theoretically a deep network can compute things that you _could_ compute
    with a shallow network, but it would take exponentially more units to
    compute!

The example given for #2 is XOR. To represent the XOR function on two inputs
might take, well, let's actually work this out (not in the video):

```
a[1,0] = relu(x[0] + x[1])
a[1,1] = relu(2*x[0] + 2*x[1] - 2)
a[2,0] = a[1,0] - a[1,1]
```

OK, so a network with two layers can implement this.

Well, suppose the function you want to implement is the XOR of n inputs. If n
is a power of 2, you can do this with `2*log2(n)` layers and a total of `n-1`
units. But what if you want to use fewer layers?

You can still do it. But it requires exponentially more units because each unit
must learn to recognize a particular case or family of cases.

(Training all those units will take more time and more training data. And you
can easily have errors where some of the cases simply did not get trained. The
deeper network is more likely to get it right, recognizing the overall XOR
pattern even though XOR is about as far from a neural network's building blocks
as you can get. This is something I could actually demonstrate by building
both.)

Ng's honesty in the final section here is balm for the soul.

## Parameters vs Hyperparameters (C1W4L07)

Hyperparameters:
- learning rate α
- number of iterations
- number of hidden layers
- number of hidden units
- choice of activation function

And in the second course we will add:
- momentum
- min-batch size
- regularizations

Applied deep learning is empirical. You try lots of different hyperparameters.

Intuitions about parameters do not necessarily transfer across domains (from
vision to speech to NLP to ads to search).

"I know this may seem unsatisfying"


## What does this have to do with the brain? (C1W4L08)

As of this video, it's still not well modeled what a neuron in the human brain
does.

And there's no evidence the human brain is "training" using anything like
backpropagation.

Computer vision has taken more inspiration from the human brain than other
fields.


# Course 2 - Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization

## Train/Dev/Test Sets (C2W1L01)

https://www.youtube.com/watch?v=1waHlpKiNyY&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc

"Mismatched train/test distribution" - cool idea. You don't have enough of the
target inputs.


## Bias/Variance (C2W1L01)

Guess: variance is when there are lots of concavities, places where the
derivative of your learned function is high; bias is whole classes/regions of
points for which your learned function is wrong

No, it's to do with train vs dev performance.

Low bias means you have good accuracy, i.e. you perform about as well on the
training set as you expect is possible.

Low variance means your performance on the training set transfers well to the
dev set. Overfitting leads to high variance.


## Basic recipe for machine learning (C2W1L03)

The point of this video is that the two different problems have different
remediation options. You have to take care of high bias first; that means your
model just isn't learning the function.

```
while high bias:
    try bigger network
    try training longer
    try a different NN architecture
# ...then...
while high variance:  # overfitting
    get more data
    try regularization
    try a different NN architecture
```

Pre-deep-learning, there was talk of the "bias-variance tradeoff" because it
was hard to improve bias without risking overfitting. Now actions to improve
bias mostly don't hurt variance and vice versa.


## Regularization (C2W1L04)

Regularization adds a term to the quantity we're trying to minimize.

For logistic regression, which has a single matrix of weights w, the extra term
is `λ/2m * Frob(w)**2` where Frob is the Frobenius norm. λ here is the
regularization parameter, yet another hyperparameter you get to choose.

The main effect is to discourage "gerrymandering" which intuitively requires a
strong weight to get a sharp change in output for a small change in input.

A side effect of L1 regularization can be to drive any parameters the model
isn't really using to 0; in theory this could help a sparse model compress
better but it doesn't seem to matter that much in practice.

But for logistic regression, it feels like there's a dimension along which this
gradient, however strong or faint you make it, is nonzero, even if the
derivative of all other terms in J is 0. In logistic regression, why doesn't
this push b toward 1/2 and your weights toward 0 until you reach the point of
floating-point instability? (open question)

Implementation: Theoretically, this adds a term to the loss function. So that
changes the derivatives when we work backwards. This affects the computation of
dw. It adds a term to dw (at each layer) of `λ/m w`. Therefore every training
cycle we end up subtracting a fraction `αλ/m` of w from w! For this reason this
is sometimes called "weight decay".

## Why regularization reduces overfitting (C2W1L05)

This is great. It's great that he gives multiple reasons it might have this
effect.

1.  It can zero out whole units, in fact if your network only needs a handful
    of units there could be thousands zeroed out (though this doesn't happen in
    practice).

2.  Using reasonable weights might restrict your model to the linear part of
    the activation function (but I'm not sure this is true either, since your
    activation function is often gonna be relu, and you've got `b` which can
    put you into other regions of the activation function's domain).


## Dropout regularization (C2W1L06)

The technique is to randomly eliminate some fraction of the nodes when training
for each example, train that way, then repeat.

Implementation: At each layer, after applying the activation function, mask out
some fraction of the units with

    mask = np.random.rand(a.shape) < keep_prob
    a *= mask
    a /= keep_prob

Note that a's shape is (n, N) where n is the number of units in the layer and N
is the number of training examples; so we are dropping out different units for
every example.

The last line is some normalization, it keeps the overall output of this layer
high despite the dropouts, so that this doesn't affect the expected value of z
in the next layer. (This bit is called the "inverted dropout technique".)

I am extremly surprised this works. It is not surprising that it disrupts
anything fancy from being learned; it's surprising that it allows anything good
to be learned.

So all of the above applies only during training. At test time, you run it with
`keep_prob=1`. (The inverted dropout step makes things more likely to work
about the same when you change this parameter.)


## Understanding dropout (C2W1L07)

Why does dropout work? The system can't rely on any one feature, so it is
trained to spread out the weights. (It seems like that would mainly train
redundancy, but the system doesn't know dropout is happening; there is no
gradient in the direction of "gee it would be nice to have another copy of that
thing that totally missing input that doesn't exist right now".)

But it turns out that dropout can formally be shown to be an adaptive form of
L2 regularization, but L2 penalty on different weights are different depending
on the size of the activations being multiplied that way. Similar effect.

One last note -- it's feasible to vary `keep_prob` by layer, particularly of
`1` in layers where you are not worried about overfitting.

What made people invent dropout? Computer vision. Megapixel inputs -- a
million-dimensional input space? It is not possible to have enough inputs to
avoid overfitting. You can even apply a `keep_prob` to the input layer where it
makes some intuitive sense as a way of multiplying your inputs.

A big downside is that the cost function J is no longer well-defined. It won't
go down on every iteration. Implementation trick: Since `keep_prob` is a
parameter, you can set it to 1 just fora single training run, and make sure the
cost decreases (just to verify that your code is working), then turn on dropout
to train for real.


## Other regularization methods (C2W1L08)

https://youtu.be/BOCLq2gpcGU

Data augmentation - using script to generate extra training data from what
you've got. This can be pretty junky and still help with training.

Early stopping - Ng doesn't love this one because it entwines the two problems
of variance and bias. "Orthogonalization".

## Normalizing inputs (C2W1L09)

Make the mean 0. Then make the variance 1.

I think the code is:

```
x -= np.mean(x, axis=1)
x /= np.mean(x**2, axis=1)
```

where x is an `(num_inputs, len(training_examples))` 2D array.

The given reason for this is that the function will be more round and easy for
gradient descent to optimize. Does this make sense? Is gradient descent
*not* invariant to changes in coordinate systems like this?

It is not.


## Vanishing and exploding gradients (C2W1L10)

Simply, activations can grow or shrink exponentially with the depth of the
network. But no argument is made as to why this should happen or why gradient
descent wouldn't just leave the bad region.

Does relu make vanishing more likely, by cutting out half the signal?


## Weight initialization in a deep network (C2W1L11)

It's better to start with properly balanced random weights.

Suppose our goal is to keep the variance of the activations throughout the
network roughly at 1.

Say a layer has n inputs. Then the output `z = w1 x1 + w2 x2 + ... + wn xn`.

If the inputs have variance 1, and the weights have variance 1, the output has
variance n (or really n/2 if we're using relu).

So we need the weights to have variance roughly 2/n instead.

The code to initialize the weights this way is

```
# note n is number of inputs to this layer, not number of units in the layer
w = np.random.randn(shape) * np.sqrt(2 / n)
```

This doesn't solve but helps reduce the problem.

If using `tanh` function, it's better to use `1` instead of `2` in the
numerator there. Called Xavier initialization.


## Numerical approximations of gradients (C2W1L12)

For technical reasons, he claims,
`(f(x + eps) - f(x - eps)) / (2 * eps)`
is a better approximation of the derivative `D f (x)` than
`(f(x + eps) - f(x)) / eps`.

Why is this? Could be some floating-point thing?
There must be a simple explanation. When the second derivative is high...?
But surely with small epsilon it almost doesn't matter.
(I think this is the part that isn't accurate. It isn't on the scale of
f'(x) but it is on the scale of the error we're concerned with eliminating.)

Well, let's say f(x) = x^3. Then computing the first way you'd get

```
((x + ε)³ - (x - ε)³) / 2ε
= (x³ + 3εx² + 3ε²x + ε³ - (x³ - 3εx² + 3ε²x - ε³)) / 2ε
= (6εx² + 2ε³) / 2ε
= 3x² + ε²
```

for an error of ε².

The second way, you'd get

```
((x + ε)³ - x³) / ε
= (x³ + 3εx² + 3ε²x + ε³ - x³) / ε
= 3x² + 3εx + ε²
```

for an error of 3εx + ε².

This would be fun to investigate by itself. Wikipedia on "Numerical
differentiation" says only that "the first-order errors cancel" if you take a
symmetric difference.


https://www.youtube.com/watch?v=y1xoI7mBtOc&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=12


## Gradient checking (C2W1L13)

Yeah, this is exactly what I should be doing that I was too lazy to do so far,
with added implementation advice.

Reshape and concatenate all parameters into a single array. Likewise reshape
all computed dW and db and concatenate into a single array.

Write J as a function of the vector of parameters (this function runs the
entire network on all the training data, gets the answers, computes the loss
function, and returns the loss). (I guess we might want to do all the math,
perhaps both J and the actual training, at double precision when validating in
this way. Or not, for verisimilitude? He does not mention this.)

For each parameter, nudge numbers, do all the work, do a symmetric difference
quotient to approximate the derivative. Then compare against the ∂L/∂wᵢ and
∂L/∂bᵢ values you computed.

Check `norm2(approx - computed) / (norm2(approx) + norm2(computed))`.

The size of the nudge he calls ε and offhand I would have used the square root
of `f64::EPSILON`, about 1.5e-8. He says he often uses 1e-7, so we'll go with
that. Then an error on the order of 1e-7 is great, 1e-5 is inconclusive, and
1e-3 is reason to worry and debug.


## Gradient checking implementation notes (C2W1L14)

-   You may have put a regularization term into your loss function. Don't
    forget it here!

-   This doesn't work with dropout. But you can just turn off dropout (set your
    `keep_prob` to 1.0), test everything that way, and this at least verifies
    everything except your dropout-related code.

-   It's possible to have a bug that would not bite as long as w and b are
    close to 0, but would get worse with training. So consider checking
    gradients at init time and again after some training (I guess just as part
    of every training run).

## Lab

- make the notebook reasonably presentable and send a link to jim - done
    https://colab.research.google.com/drive/1hgIxGGjqVfsVbHPbATyQzkWZDfDIkI-D?usp=sharing
- implement gradient checking, debug - done
- implement multiple layers
- hook up the MNIST handwritten digits training data, i guess

Checking my derivative formulas:

```
      z = w @ x + b
    # ^(no,n)
    #     ^(no,ni)
    #         ^(ni,n)
    #             ^(no,1)
    dz[o,k] = sum(w[o,i]*x[i,k] for i) + b[o,0]
    dz[o,k]/dw[o,i] = x[i,k]
    dz[o,k]/db[o,0] = 1

    dL/dw[o,i] = sum(dL/dz[o,k] * dz[o,k]/dw[o,i] for k)
```

but it was all correct! My bug was that my gradient descent would get the loss
down to 0.25 or so and then progress decreased to a ludicrously slow rate. I
put in a gruesome hack to crank the learning rate in that case, and it worked.


## Mini batch gradient descent (C2W2L01)

Obvious.

https://www.youtube.com/watch?v=4qJaSmvhxi8&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=15


## 

