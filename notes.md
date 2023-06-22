# The Coursera course - C1 weeks 1 and 2

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
    an accurate description of what's going on. More like: it's just making
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



# Course 1 - Neural networks and deep learning

https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0


## Logistic regression (C1W2L02)

https://www.youtube.com/watch?v=hjrYrynGWGA&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=8

He says that it turns out to be easier to implement the bias vector b as a
separate thing rather than add an input that is pegged to 1. I wonder why.

Can I remember the loss function for logistic regression from memory?

He says they don't use the square of the error `sum((yhat - y)**2)` because it
makes the optimization problem non-convex.

So I think what they use is `avg(-(y log yhat + (1-y) log (1-yhat)))`.


## Explanation of logistic regression's cost function (C1W2L18)

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


## Computing neural network output (C1W3L03)

Oh, the single-neuron logistic regression isn't seen as a neural network
really. It must be a technique that sort of precedes neural networks.

Already knew all this. Skipped a lot.


## Vectorizing across multiple examples (C1W3L04), Explanation for vectorized implementation (C1W3L05)

Skipping this one.


## Activation functions (C1W3L06)

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


## Why non-linear activation functions (C1W3L07)

I already knew this, skipping


## Derivatives of activation functions (C1W3L08)

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
intuition is. A: The parameter space is exactly symmetric about planes on which
any two neurons are the same. If we start on that line, there will never be any
gradient in a direction away from that plane, so we are stuck there. The two
neurons can't diverge.

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


# Course 2 - Improving deep neural networks: hyperparameter tuning, regularization and optimization

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


## Understanding mini-batch gradient descent (C2W2L02)

The main problem with a mini-batch size of 1 is not the noise, but that you
lose the speedup from vectorization!

Guidelines: if small training set, m ≤ 2000, just use batch gradient descent.

Typical mini-batch sizes are powers of 2 from 64 = 2⁶ to 512 = 2⁹.

Make sure your mini-batch fits in CPU/GPU memory.


## Exponentially weighted averages (C2W2L03)

Given a data series θ, produce a smoothed data series v:

    v[t] = β v[t - 1] + (1 - β) θ[t]

Very roughly speaking, it finds something like
the average over the last `1 / (1 - β)` days.

If β is 0.99, the weight of the latest data point is 1/100.


## Understanding exponentially weighted averages (C2W2L04)

It's a convolution.

The reason it's like the average of the last `1/(1-β)` days is that

    β ^ (1 / β)

is an approximation of 1/e, less than a third.
The most recent `1/(1-β)` days
account for more than 2/3 of the current weighted average.


## Bias correction of exponentially weighted averages (C2W2L05)

Instead of taking v[t] from the recurrence relation above, use

    v[t] / (1 - β^t)

The denominator quickly goes to 1.


## Gradient descent with momentum (C2W2L06)

In a narrow valley, gradient descent may oscillate,
because the step size is too big for that valley.

The fix for this is to use a weighted average of momentum instead of the
current gradient. That will push us in the right direction, maybe. It will tend
to eliminate the movement up and down the steep walls of the valley without
reducing movement in the direction of the river.

The intuition is that this is "like" the partial derivatives are being used as
"acceleration" terms, and the v[t] is the "velocity". Ball rolling downhill.
The derivative imparts acceleration to this ball. β plays the role of friction,
preventing the ball from accelerating without limit.

    vdw := β vdw + (1 - β) dw
    vdb := β vdb + (1 - β) db
    w -= α vdw
    b -= α vdb

In practice β = 0.9 works well and bias correction is not worth it for this algorithm.

It's common to omit the `1 - β` term, but this means if you change β you'd have
effectively changed α (scaled your step size by the sum of an infinite series
that turns out to be `1 / (1 - β)` a fairly large number); the formulation
above keeps them uncoupled.

This almost always works better than gradient descent.


## RMSProp (C2W2L07)

    sdw = β sdw + (1 - β) dw^2
    w -= α (dw / sqrt(sdw + ε))

This is just bizarre to me. Damps out movement in steep directions, but this is
not what we want - is it?

I guess all this is the distillation of practical wisdom on high-dimension
gradient descent in general. It just seems bizarre that a smarter algorithm
would not work better.

Fun fact, RMSProp was first proposed in a Coursera course.


## Adam optimization algorithm (C2W2L08)

Adam = Adaptive moment estimation (vdw, mean of derivatives, is the first
moment; sdw, mean of squares, is called the second moment)

    vdw = 0
    sdw = 0
    for t, minibatch:
        vdw = β1 vdw + (1 - β1) dw
        sdw = β2 sdw + (1 - β2) dw^2
        vdw_corrected = vdw / (1 - β1^t)
        sdw_corrected = sdb / (1 - β2^t)
        w -= α vdw / sqrt(sdw + ε)

Hyperparameters:

    α needs to be tuned.
    β1 = 0.9
    β2 = 0.999
    ε = 10^-8 (doesn't matter much)


## Learning rate decay (C2W2L09)

1 epoch = 1 pass through the data. Good to know.

One thing you can do is set

    α = α₀ / (1 + decay_rate * epoch_num)

New hyperparameter.

Other formulas:

    α = α₀ * 0.95 ^ epoch_num

    α = α₀ * k / sqrt(epoch_num)

Manual decay is also done sometimes! If your model trains over many days, you
can just watch it get stuck and manually decrease α.

Next week: systematic hyperparameter tuning


## The problem of local optima (C2W3L10)

This was misnumbered and belongs at the end of week 2.

This repeats the part of what the survey paper said that makes sense, which is
that almost all points of zero gradient are saddle points (not local minima).

Sure. The intuition is that at any point with zero gradient, on each dimension,
the second derivative is either positive or negative. The point is a local
minimum only if all of them are positive. If any are negative, it's a saddle
point. Knowing nothing more about our functions, saddle points have to be the
more common case.


## Tuning process (C2W3L01)

α is the most important metaparameter to tune.

Then β (if not using Adam), number of hidden units, and mini-batch size.

Third, number of layers and learning rate decay.

If using Adam, Ng never tunes β1, β2, or ε.

Do not use a grid. Instead choose random points in the hyperparameter space.
This way, worst case, you will have tried N different values for α! You can
toss in lots of hyperparameters that do not matter much and that's fine.

Another thing you can do is course-to-fine search. Zoom in after initial
sampling of the space.


## Using an appropriate scale (C2W3L02)

Use a log scale for picking random values of α and other hyperparameters for
which reasonable values span orders of magnitude.

Likewise, for β, which might range from 0.9 to 0.999,
instead pick a number from 0.001 to 0.1 on a log scale and set `β = 1 - that`.
(irritating - this reveals that `1 - β` is the actual hyperparameter)

## Hyperparameter tuning in practice (C2W3L03)

Re-test your hyperparameters occasionally!

Panda vs. caviar approach. If you have the CPUs, do caviar.

## Normalizing activations in a network (C2W3L04)

Batch normalization makes your training more robust to choice of
hyperparameters.

It works by adding up the values of z (or a) at each layer over a batch, then
using that to compute the mean and variance of each of those values (what I
think of as "signals"); then normalize (each signal separately);

    z_norm = (z - μ) / sqrt(σ² + ε)     # ε term in case variance is zero

and after this add a layer that does:

    z~ = γ z_norm + β

where this γ and β are learnable parameters of the model, and z~ is passed to
the activation and downstream, instead of z.

This way the model _may_ choose γ and β so as to undo the normalization if
that's best. But as a default the output of each layer is normalized, to help
the next layer have an easier time training.

https://www.youtube.com/watch?v=AXDByU3D1hA&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=25

## Fitting batch norm into neural networks (C2W3L05)

It gets rid of `b` of `(w, b)` fame, because the learnable mean parameter is
the same thing but kind of better. `b` has no effect on the output, therefore
the derivative of output with respect to b is 0, and we never learn anything
for it.

`β` and `γ` each have the same shape that `b` had.

`tf.nn.batch_normalization`.

Does this require larger mini-batches? It seems you could get unlucky and have
an input go haywire just from bad luck. (The layer normalization paper says
that this does depend on batch sizes.)


## Why does batch norm work? (C2W3L06)

Nice insight: it makes later tiers of your neural net more robust to changes
early in the net. Consider the perspective of a neuron in tier 3. Because
earlier tiers are always learning and changing, the _meaning_ and _distribution_
of the inputs to tier 3 are always changing. The neuron in tier 3 is suffering
from covariate shift. The input is different now.

With batch norm, at least the mean and variance will remain the same (unless
some part of the network finds it valuable to change them!).

There is a second effect: batch norm as regularization. Similar to dropout, it
adds some noise to each hidden layer's output (because each mini-batch is
scaled by its mean/variance which will be somewhat random). This forces
downstream units not to be too dependent on any one upstream unit. This is a
slight effect.

A larger mini-batch size reduces this regularization effect.

Don't turn to batch norm for regularization though.


## Batch norm at test time (C2W3L07)

Problem: when learning, batch norm works on one mini-batch at a time. When
using the network in production, you might not have mini-batches! What then?

Prediction: You're going to keep a weighted exponential average of μ and σ². It
becomes part of your model, but not learned through gradient descent like
everything else.

Yup.

This turns out to be pretty robust to how exactly you estimate μ and σ², so
don't sweat that.


## Softmax regression (C2W3L08)

Hey, this is where we came in!

Softmax is an activation function for the last layer. Unusual in that it is not
a function of individual outputs but a function of all outputs.

Suppose the last layer has 10 outputs, the 10 digits 0-9, and the output of the
linear part of those neurons is z. z's shape is (10, n) where n is the mini-batch size.

    t = exp(z)
    a = t / sum(t)

This forces the elements of a to add up to 1 as desired.

Pictures show what softmax can do -- very nice presentation.

Why are the boundaries linear? Because these pictures do not show softmax
probability values; rather, each pixel tells you which output has the greatest
strength. But this is not softmax. It's max.


## Training softmax classifier (C2W3L09)

Explains why it's called "softmax".

Will it be different from what I'm doing, using the loss function he gave me
for logistic regression back in C1W2L18?

Claims if C = 2, softmax reduces to logistic regression.

    Loss(yh, y) = -sum(y[j] * log(yh[j]) for j in 0..C)

When y is made up of zeros and ones,
such that `y[j] = 1` and `y[k] = 0` where `k /= j`,
we get

    Loss(yh, y) = -log(yh[j])

so that only the value of yh[j] matters. I'm surprised. This is not what I was
thinking of doing.

Backprop is nice and easy in this case, because

    dz = yh - y

I don't see why that should be the case.


## The problem of local optima (C2W3L10)

See above at the end of C2W2.

## TensorFlow (C2W3L11)

```
w = tf.Variable(0, dtype=tf.float32)
cost = tf.add(tf.add(w**2, tf.multiply(-10., w)), 25)  # tensorflow knows how to take derivatives
cost = w**2 - 10*w + 25  # tf has operator overloading of course
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()

# These lines of code are idiomatic
with tf.Session() as session:
    session.run(init)
    print(session.run(w))

session.run(train)
print(session.run(w))

for i in range(1000):
    session.run(train)
print(session.run(w))
```

```
x = tf.placeholder(tf.float32, [3, 1])

session.run(train, feed_dict={x: coefficients})
```

Not really following this part.



# The Coursera course - C1W4

## `C1_W4_Lab_1_image_generator_no_validation.ipynb`

from the coursera course "Introduction to TensorFlow for Artificial
Intelligence, Machine Learning, and..." from DeepLearning.AI, here:
https://www.coursera.org/learn/introduction-tensorflow/home/week/4

Since this will be on the test:

```
!wget <URL>
```

This appears to be a normal code cell. Then normal python stuff works on the
VM's filesystem.

```
import zipfile

# Unzip the dataset
local_zip = './horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./horse-or-human')
zip_ref.close()
```

```
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        './horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')
```

> Deprecated: `tf.keras.preprocessing.image.ImageDataGenerator` is not
> recommended for new code. Prefer loading images with
> `tf.keras.utils.image_dataset_from_directory` and transforming the output
> `tf.data.Dataset` with preprocessing layers. For more information, see the
> tutorials for loading images and augmenting images, as well as the
> preprocessing layer guide.

OK, so the new thing would be like

```
from tensorflow.keras.utils import image_dataset_from_directory

ds = dataset_from_directory(
    "./horse-or-human",
    label_mode='binary',
    batch_size=128,
    image_size=(300, 300),
)
```

and `model.fit` needs a new `steps_per_epoch=8` argument to limit the epoch to
the approximate size of the data we have, because the dataset is basically
going to generate an infinite stream of images, from `model.fit`'s perspective.

## `C1_W4_Lab_2_image_generator_with_validation.ipynb`

```
history = model.fit(
    ...
    validation_data=validation_generator,
    validation_steps=8
)
```

## `C1_W4_Lab_3_compacted_images.ipynb`

-   Did we lose quality at all? If so, it should be reflected in the validation
    scores.

-   Did we lose quality because the information content in the shrunk images is
    less, or because we skipped a bunch of convolution layers? Or some other
    reason?

I immediately switched to adam since it seems to give better results.

I choose as my metric the second-best validation score in epoch 5 or later.

1.  With three convolutional layers, we barely break 80% on validation data,
    even though training accuracy goes to 99.9%. Metric: 80.86%.

    18496 inputs to the final dense layers.
    9,494,561 params.

2.  With a fourth convolutional layer: 87.89%, and validation accuracy was over
    80% consistently.

    3136 inputs to the final dense layers.
    1,667,169 params.

3.  With the fifth convolutional layer: 83.59%. I ran it again. 84.38%. Maybe
    this architecture is just worse -- the last pooling layer reduces the
    resolution to 2x2.

    256 inputs to the final dense layers.
    229,537 params.

4.  With the fifth convolutional layer, but no fifth pooling layer: 86.72%.

    1600 inputs to the final dense layers.
    917,665 params.

5.  For a lark, I went back to the notebook that used 300x300 images, switched
    to adam, and did a training run there. The number of params and CNN outputs
    is very similar to case 2 above, the best-performing of the bunch. 82.81%.

    3136 inputs to the final dense layers.
    1,704,097 params.

Switching to 150x150 images is good actually. We can save a lot of work by
doing the math on fewer pixels, and the results are as good or better if we
choose an appropriate architecture.


# MyML implementation of cross-entropy

I am thinking about softmax and a subsequent loss function called "categorical
cross-entropy".

Softmax is:

        ex = np.exp(x)
        yh = ex / np.sum(ex, axis=0)

Categorical cross-entropy is:

        loss = np.mean(-np.log(yh[y, np.arange(n)]))

I tried making these separate layers but got into trouble computing the
derivatives for the gradient. The problem is that softmax destroys a dimension
of space, and I don't know enough calculus to get the right answer despite
having decomposed the function in that way.

Unpacking the numpy magic, we get:

    # n = number of examples
    # no = number of outputs (== number of categories)
    ex[i,j] = exp(x[i,j])
    yh[i,j] = ex[i,j] / sum(ex[k,j] for k in range(no))
    loss = mean(-log(yh[y[j], j]) for j in range(n))

So what is the derivative of loss with respect to each x?

    dyh[i,j] = -1/(n * yh[i,j]) if i == y[j] else 0

    dex[i,j] = ∂loss/∂yh[y[j],j] * ∂yh[y[j],j]/∂ex[i,j]
    dex[i,j] = dyh[y[j], j] * (
        if i == y[j]
        then (sum(ex[k,j] for k in range(no)) - ex[y[j],j]) / sum(ex[k,j] for k in range(no)) ** 2
        else -ex[y[j],j] / sum(ex[k,j] for k in range(no)) ** 2
    )

    dx[i,j] = exp(x[i,j]) * dex[i,j]

There is a mathematically easier, computationally harder way that is correct
regardless of the loss function. Hey how 'bout it.

    dex[i,j] = sum(for k in 0..no: ∂loss/∂yh[k,j] * ∂yh[k,j]/∂ex[i,j])
             = sum(for k in 0..no: dyh[k,j] * (
                   if i == k
                   then (sumex[j] - ex[k,j]) / sumex[j]**2
                   else -ex[k,j] / sumex[j]**2
               )
             = sum(for k in 0..no:
                   dyh[k,j] * ((if i == k then sumex[j] else 0) - ex[k,j]) / sumex[j]**2)

This formula assumes something about linearity of partial derivatives that I
don't even know how to articulate. But I suspect it's true. It is definitely
true for the case we've got, where only one element of yh actually contributes
to a change in loss.

----

With this formula, the program actually works.


# CS224U - The Stanford NLP Podcast

Rishi Bommasani had an academic team that spent six months independently
reproducing GPT-2 (because they thought it would be easy™). He says a
surprising fact about LLMs is that they just crash when you train them long
enough. Parameters "blow up". I guess it just diverges. He says they've talked
to practitioners and the standard solution is to save checkpoints, detect the
crash, roll back, *shuffle the training data* and try again.


# Course 3 - Structuring machine learning projects

## Orthogonalization (C3W1L02)

https://www.youtube.com/watch?v=UEtvV1D6B3s&list=PLkDaE6sCZn6E7jZ9sN_xHwSHOdjUxUW_b&index=2


Chain of assumptions in ML

- Fit training set well on cost function
    - bigger network
    - better optimization algorithm
- Fit dev set well on cost function
    - regularization: weight decay, dropout, data augmentation
    - find more/better training data
- Fit test set well on cost function
    - Bigger dev set
- Performs well in real world
    - Change dev set; or
    - Change cost function!

Early stopping affects two things, the first two. Not an orthogonal knob to
twist.


## Single number evaluation metric (C3W1L03)

Precision is a measure of false positives, what proportion of what we found
really matches the search criteria. Recall is a measure of false negatives,
what percentage of true matches did we find.

Using these as your evaluation metrics you run into tradeoffs. Having a single
number settles all the tradeoffs in advance so you focus on iterating.

Instead use an F1 score, harmonic mean of P and R.

Haha this video is dumb but may be incredibly valuable


## Satisficing and optimizing metrics (C3W1L04)

You should have 1 optimizing metric and any number of satisficing metrics (i.e.
constraints).

OK this is not genius stuff


## Train/dev/test set distributions (C3W1L05)

https://www.youtube.com/watch?v=M3qpIzy4MQk&list=PLkDaE6sCZn6E7jZ9sN_xHwSHOdjUxUW_b&index=5


## Why human level performance? (C3W1L08)

In short, humans are good enough at many tasks; and while your ML is worse than
humans you can get help from humans (labeling data, etc.).


## Avoidable bias (C3W1L09)

"Bias" in the sense of "Bias/Variance" (C2W1L01).

Human-level error as a proxy for Bayes error.

Nonstandard terms: difference between Bayes and your training error is
"avoidable bias". Distinction matters when Bayes error is nonzero and your
model's performance gets close to ideal. If your variance is greater than the
*avoidable* bias then it may be time to work on variance.


## Understanding human-level performance? (C3W1L10)

Snore-fest. Where's the good stuff?


## Surpassing human-level performance (C3W1L11)

If you have surpassed human-level performance, you have no remotely plausible
estimate of Bayes error. This makes it harder to use the heuristic of C3W1L09
to decide what to work on next.

OMG he thinks machine learning is good at product recommendations. Literally
the garbage process that used to populate Borders is better. F minus.

Common threads among fields where computers are doing better than people

- structured data
- not natural perception
- lots of data

there are a few perception tasks where ML does better:

- speech recognition
- some image recognition tasks
- some medical tasks

in some cases -- but Ng thinks it has been harder.


## Improving your model performance (C3W1L12)

https://www.youtube.com/watch?v=zg26t-BH7ao&list=PLkDaE6sCZn6E7jZ9sN_xHwSHOdjUxUW_b&index=12

Once again:

To improve avoidable bias:
- train a bigger model; or
- train longer; or
- use a different optimization algorithm; or
- use a different NN architecture; or
- hyperparameter search

To improve variants
- get more data
- use regularization (L2, dropout)
- data augmentation
- more NN architecture / hyperparameter search


## Carrying out error analysis (C3W2L01)

Manually examining the mistakes your algorithm is making can help.

Error analysis plan:
- Get ~100 mislabeled dev set examples.
- Count how many the work you're considering doing would actually fix.

Use a spreadsheet and evaluate multiple ideas in parallel.

## Cleaning up incorrectly labeled data (C3W2L02)

Deep learning algorithms are quite robust for *random* errors in the training
set. Not systemic errors.

If you're worried about errors in the dev or test set, add a column
"incorrectly labeled" to your error analysis spreadsheet.

It is often not worth fixing incorrectly labeled data.

Advice:

-   Make sure you apply the same process to dev and test! They have to be from
    the same distribution.

-   Consider examples your classifier got right as well as ones it got wrong.


## Build first sytsem quickly, then iterate (C3W2L03)

Part of the value in building some working system and test/dev sets is the
ability to use error analysis to guide the team's attention.


## Training and testing on different distributions (C3W2L04)

Good sense.


## Bias and variance with mismatched data (C3W2L05)

If your train and dev sets are different distributions,
then when there is a big gap in accuracy between the two stes,
you don't know if it's because of variance (overfitting) or distribution.

Ng uses "variance" to refer specifically to the network "not generalizing well"
from the training data to dev, not to the other effect.

Solution: training-dev set with the same distribution as the training set, but
used for evalutaion, not training.

From training to training-dev is variance.

From training-dev to dev is the data mismatch problem.

It sometimes happens that your dev/test examples are "easier" than the training
examples from a different source, and so the error is much lower. To confirm
this is happening you can evaluate your network on specifically the examples in
your training set that are representative of the dev/test distribution. To
illustrate this Ng drew a 2d matrix where the two distributions are the
columns, and the rows are "human level / bayes error", "error on examples you
trained on", and "error on examples not trained on".


## Addressing data mismatch (C3W2L06)

Error due to mismatch between training and test/dev distributions.

There are not systematic ways to do this.

*   Manual error analysis - try to understand the differences.

*   Find ways to make training data more similar; or collect more.

Artificial data synthesis; limitations of. In speech recognition has seen it
work. CGI pictures of cars, hmm.


## Transfer learning (C3W2L07)

1.  Pre-train a deep neural net for task A, using one data set.
2.  Replace the last layer(s) with new layers, initialized with random weights.
3.  Train the new layers (and others if desired) on the actual target task, task B.

Makes sense when
- tasks A and B have the same type of input
- you have a lot more data for task A than B
- low-level features learned for A are useful for B

This is useful when you have a lot of training data for the first task and not
so much for the second task.

https://www.youtube.com/watch?v=sn_QSB7T1xo&list=PLkDaE6sCZn6E7jZ9sN_xHwSHOdjUxUW_b&index=18


## Multi-task learning (C3W2L08)

Train for multiple tasks at once.

Oddly a lot of this is about designing a loss function to train a multiple
classifier as opposed to, say, softmax on digit recognition

Makes sense when
- tasks can benefit from shared lower-level features
- amount of data you have for each task is pretty similar
- you can train a big enough neural network to do well on all tasks

Not super common. Computer vision object detection are the main applications.


## What is end-to-end deep learning? (C3W2L09)

Speech recognition used to involve a pipeline:

    audio -> features -> phonemes -> words -> transcript

End-to-end deep learning replaces all this with a single neural network. A
large training set is required -- on the order of 100,000 hours of audio for
transcription! (This obsoleted many years of research in speech recognition.)

**It doesn't always work.**

Example: face recognition turnstile

This is best decomposed into two subtasks because there's lots of data for each
of them, not end-to-end (this is 2017). Finding a face in a photo, lots of
examples exist, likewise there are hundreds of millions of examples for
detecting if two mugshots are the same person.

Example: Machine translation. End-to-end works very well because it's possible
to get enough (x, y) pairs.

Example: Estimate the age of a child from an x-ray (used to see if a patient is
developing normally)

`image --(segmentation)--> bones -> age` decomposition works well; there is not
enough data to do the task end-to-end.


## Whether to use end-to-end learning (C3W2L10)

Pros:

-   Let the data speak, not human preconceptions.
    (For example, Ng thinks phonemes are a fantasy of linguists!)

-   Less hand-designing of components.

Cons:

-   May need a lot of data

-   Excludes potentially useful hand-designed components (which allows you to
    manually inject knowledge into the system -- can help or hurt).

Really the question is if you have enough data.

Imagine the problem of self-driving cars. You must take the input, find
objects, route, and then steer, brake, etc. End-to-end deep learning is not the
approach that works best.

This concludes course 3. Course 4 should get back to interesting stuff.


# CS480/680 Lecture 19: Attention and Transformer Networks

Pascal Poupart, U Waterloo, Spring 2019. The name of the course is
"Introduction to Machine Learning".

https://www.youtube.com/playlist?list=PLdAoL1zKcqTW-uzoSVBNEecKHsnug_M0k

-   Attention Is All You Need 2017 (when was the podcast I heard made?)

-   attention on images gives "heatmaps"

-   useful for evaluating understanding - a network classifies a picture as
    containing a building. well, which pixels told you that? useful for the
    system to be able to tell you.

-   2015: machine translation - this mechanism lets a network "peek back"
    at previous tokens so it doesn't lose track of what it's translating,
    and doesn't have to remember evyrething - great for long sentences

-   2017: language modeling with transformer networks

    (many tasks can be cast as language modeling -- yeah no kidding)

-   RNNs are hard to train. They blow up, large # of training steps needed.
    Recurrence (correlated parameters) makes optimization hard, prevents
    parallel computation.

-   Transformer networks have fewer layers in practice.

He has not gotten to what the idea is. Also I didn't watch any previous videos
so I don't know what attention is really. Oh wait, he recaps at 11:50.

-   Attention mimics the retrieval of a value for a query based on a key, like
    a select in a database.

        attention(q, k, v) = sum(for i, similarity(q, k[i]) * v[i])

    Q: The similarity function ought to sum to at most 1, I think, or there
    should be a weighted average instead?

This is great, will watch more later.

# The Coursera course - C2W1

https://colab.research.google.com/drive/1PcfV25kctKwBSgDpzpVXFC13luLOcOq2


# Course 4 - Convolutional neural networks

## Convolutional neural networks (C4W1)

-   Computer vision (C4W1L01)

-   Edge detection examples (C4W1L02)

-   More edge detection (C4W1L03)

    At the beginning of computer vision, there was a lot of debate about which
    numbers to use in your edge detection matrix. Sobel filter, Scharr filter.
    But with deep learning you just learn them.

-   Padding (C4W1L04)

    By convention, when you pad, you pad with zeros. (In the later course, we
    never bother.)

    "Valid" - no padding

    "Same" - pad so that output size = input size

-   Strided Convolutions (C4W1L05)

-   Convolutions over volumes (C4W1L06)

    "Depth" - number of layers in a neural network
    "Channels" - dimension through different kinds of information in each pixel

    (640, 480, 3) * (64, 3, 3, 3) => (640-2, 480-2, 64)

-   One layer of a convolutional net (C4W1L07)

    Bias for a conv2d layer is a single parameter per channel,
    added to every pixel;
    relu applied to every pixel

    number of params is independent of image size
    thus less prone to overfitting

    notationL `(n_H, n_W, n_C)` for height, width, channels/filters

    shape of weights is `(f[l], f[l], n_C[l-1], n_C[l])`

    some authors put channels first (in the shape tuple)

-   A simple convolutional network example (C4W1L08)

    His example has 1960 outputs from the convlutional layers to the dense layers.

-   Pooling layers (C4W1L09)

    I've only seen `f=2, stride=2` but `f=3, stride=2` is sometimes used.

    Almost never uses padding, though there is one exception we'll see next week
    (it is in C4W2L05, the inception network).

    Pooling applies to each channel independently (each channel is a feature).

-   CNN example (C4W1L10)

    LeNet-5 was a pioneering example of a convolutional neural network with two
    convolutional layers (with 5x5 filters but a tiny number of channels) and
    three dense layers.

    It was 1989, so the sigmoid activation function was used, and average
    pooling rather than max pooling.

    Total number of activations decreases gradually as you progress through the
    layers (though the convolutional layers tend to increase the number
    slightly, especially the first, where you go from 3 channels to maybe 8 or
    16 or 100). If it drops too suddenly that tends to be bad for performance.

-   Why convolutions (C4W1L11)

    - parameter sharing
    - sparse connections

    CNNs are sometimes said to be very good at capturing translation invariance.
    Convolutional structure helps the network encode this. I haven't seen it yet,
    with these tiny images we're using, but OK.

-   Why look at case studies? (C4W2L01)

## Classic networks (C4W2L02)

### LeNet-5

LeCun et al., 1998. Gradient-based learning applied to document recognition.

About 60,000 parameters. To save computation, some filters in each
layer saw some channels of the input, but not all. These days you would
just connect them all.

### AlexNet

Krizhevsky et al., 2012. ImageNet classification with deep CNNs.

Input is 227x227x3 images. Layers:

-   conv2d(96 filters, 11x11, stride=4)
-   maxpool2D 3x3, stride=2
-   conv2d(256 filters, 5x5, padding='same')
-   maxpool2d 3x3, stride=2
-   conv2d(384 filters, 3x3, padding='same')
-   conv2d(384 filters, 3x3, padding='same')
-   conv2d(256 filters, 3x3, padding='same')
-   maxpool2d(3x3, stride=2)
-   flatten (9216 outputs)
-   Dense(4096)
-   Dense(4096)
-   Dense(1000, activation='softmax')

Like LeNet but much bigger. About 60M parameters. Used ReLU. Multiple GPUs.

Local response normalization (LRN). Not used anymore.

This is the paper that convinced the computer vision community that deep
learning worked. One of the more accessible papers.

### VGG-16

Simonyan & Zisserman 2015. Very deep convolutional networks for large-scale
image recognition.

16 refers to 16 layers with parameters to tune.

All Conv2D layers in this network use 3x3 filters, stride of 1, and "same" padding.
All MaxPool2D layers are just 2x2 with a stride of 2.

-   Conv2D(64) twice
-   MaxPool2D
-   Conv2D(128) twice
-   MaxPool2D
-   Conv2D(256) three times
-   MaxPool2D
-   Conv2D(512) three times
-   MaxPool2D
-   Conv2D(512) three times
-   MaxPool2D
-   Dense(4096)
-   Dense(4096)
-   Dense(1000, activation='softmax')

138M parameters. Size goes down as number of features goes up. Relative
uniformity of the architecture meant it felt like less just-so magic. But, a
lot of parameters to train.

(Second paper to read, after AlexNet.)


## ResNets (C4W2L03)

He et al., 2015. Deep residual networks for image recognition.

A residual block is a series of layers in which inputs are "fast-forwarded" and
added to a later layer. In the example, the inputs `a[l]` are added to the output of the linear transform in layer `l+2`, like this:

    z[l+1] = W[l+1] @ a[l] + b[l+1]
    a[l+1] = g(z[l+1])

    z[l+2] = W[l+2] @ a[l+1] + b[l+2]
    a[l+2] = g(z[l+2] + a[l])         <---- note extra a[l] term here

(Why is the shortcut injected before the nonlinearity? It makes sense.)

(Wait, isn't the new addition adding values with very different meanings? Maybe
the intermediate layers l+1 and l+2 are just "fine-tuning", or touching up the
output of layer l.)

The addition means layers must have the same dimension, so you see a lot of
"same" padding. If they are different, you can add another matrix to change the
size of the fast-forwarded data:

    a[l+2] = g(z[l+2] + Ws @ a[l])

where `Ws` is a matrix of parameters to be learned, or any number of other things.

Stack residual blocks to make your network.

Why is this valuable? It allows you to train much deeper networks.

Empirically, the more layers you add, past a point, training error goes back
up. But residual networks help with vanishing and exploding gradients, so
deeper networks continue to improve loss.


## Why ResNets work (C4W2L04)

Intuitively, it's very easy for the network to learn the identity function for
a residual block. Just let all weights decay to 0 and that is what you get.

So it makes sense a residual block probably shouldn't hurt performance.

Me: It's a way to let the network learn some fine-tuning around every layer.


## Network in network (C4W2L05)

Lin et al., 2013. Network in network.

A 1x1 Conv2D layer is a linear combination of the channels. So it's like a
Dense layer but applied to each pixel in the manner of a Conv2D layer.

Can be used to reduce the number of channels any amount.


## Inception network motivation (C4W2L06)

Szegedy et al. 2014. Going deeper with convolutions.

Just do everything in every layer, and stack up the output channels, which all
need to have the same size. (Here if you want to try pooling among the other
things you're trying, you'll use MaxPool2D with stride 1 and "same" padding, an
unusual case.)

Computational cost is a problem. Convolutions with large filters *and* lots of
input channels *and* lots of output channels cost `f*f*ni*no` to compute _per
pixel_. So there's a trick: use a 1x1 convolution (a "bottleneck layer") to
reduce the number of channels first. This saves 90%, doesn't hurt performance
in practice. I wonder if they wouldn't have been better off using the hack from
LeNet-5.


## Inception network (C4W2L07)

Szegedy et al., 2014, Going deeper with convolutions.

An inception module takes the previous activation, does in parallel these
things. For the same of example, suppose the input has 192 channels.

-   1x1 conv (producing, for example, 64 channels)
-   1x1 conv bottleneck into 3x3 conv (128 channels)
-   1x1 conv bottleneck into 5x5 conv (32 channels)
-   max pooling 3x3, stride=1 but in our example this produces 192 channels of
    output, and we don't want pooling to dominate the output. so pipe the result
    into a 1x1 conv to reduce output dimensionality (32 channels)

Then do channel concatenation to produce one big volume of outputs (256
channels) each channel the same size.

The inception network is just a stack of these modules, with occasional pooling
layers to reduce size.

In the paper, there are side branches that use a hidden layer to try to make a
prediction. Ensures that even features in the middle of the network are not too
bad for predicting what we want to know about an image. "This appears to have a
regularizing effect" on the system, prevents overfitting.

This was developed at google and called "GoogLeNet", in homage to LeNet.

_Inception_ movie meme. Ugh.


## What is Word2Vec? A simple explanation

https://www.youtube.com/watch?v=hQwFeIupNP0

I watched the first 13:40 of this. The vector representation of a word is the
set of weights for the *last* layer of the network, for the neuron selecting
that word as output. In the next 3' of the video, he uses a different task, and
there the vector representation is the set of weights for the *first* layer of
the network (which is what I had anticipated). Either one works.

(There must be a way to use the same weights in reverse on input. But never
mind.)

This is clearly what (Bengio 2003) was trying to tell me. The task they chose
was language modeling. This is pretty rad!

You can search the vector space for gender pairs of words. You can figure out
the past tense relationship and look for instances of that or use it to find
the past tense of any verb. Or the country-capital relation.



## Convolutional neural networks (C4W2)

-   Using open source implementations (C4W2L08)

-   Transfer learning (C4W2L09)

    "something you should always do" in computer vision

    Replace the final softmax layer with your own new one.

    `trainableParameter=0` or `freeze=1` to stop training layers

    For training, precompute the frozen part on each example input and save them to disk.

    If you have more data, freeze fewer layers. Keep later layers and their
    parameters, but retrain them on your data; or replace them with your own layers.

-   Data augmentation (C4W2L10)

    For most computer vision problems, we just can't get enough data.

    - mirroring
    - random cropping (as long as reasonably large)
    - rotation, shearing, local warping - used less perhaps because of complexity
    - color shifting (using PCA, details in AlexNet paper)

    This is a little wild to me -- data augmentation is possible because of
    symmetries in the problem space, but instead of incorporating those
    symmetries into the network, we train it in with this cumbersome mechanism.

-   State of computer vision (C4W2L11)

    As of November 2017:

    - we have a nice amount of data for speech recognition
    - can never get enough for image recognition
    - even less for object detection

    When you have lots of data, you can "get away with" using simpler
    algorithms and less hand-engineering. Less data, more hacks.

    You see a lot of complex hyperparameter choices in computer vision.

    (Transfer learning fits on this data-hacks axis too.)

    Fun dumb things you only do to excel on benchmarks:
    -   ensembling - train 3-15 networks independently and average their outputs
        (good for 1-2%)
    -   multi-crop at test time - run classifier on several versions of test image
        and average the results (too slow/expensive, 10x, in production)


## Labs for computer vision week 2

my assignment to myself

-   grab a data set with labeled images, train a plain CNN on it, maybe exactly AlexNet
-   implement data augmentation via mirroring and random cropping, measure improvement
-   implement resnet, measure without data augmentation
-   and measure with both

CIFAR100 is a data set to use. low resolution.

Resume computer vision playlist at week 3:
https://www.youtube.com/watch?v=GSwYGkTfOKk&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=23


## Word2vec plan

-   My training set will be Programming Rust, which is 213,551 words. i'll
    start by not stripping out anything, counting newlines and punctuation as
    tokens. there are some `V`=12,000 distinct words counting all that garbage

    ```
    cat ../../rust-book/atlas/*.md | sed 's/$/ <NL>/' | sed -E 's/([.,:;!?()"[`~^*%+<>&@\/|{}-]|\])/ \1 /g' | sed 's/ /\n/g' | sed 's/^ *//' | sort | uniq -c | wc -l
    ```

-   problem is prediction based on previous `n` words.

-   Network architecture: experiment with it, but I believe word2vec did
    something very simple with just 2 layers, an input layer to compress the
    `n * V` inputs to `n_vec` activations, and a softmax output layer to expand
    those to `n` outputs.

-   Implement in Keras.

    -   Why didn't the Keras `one_hot` gadget work for me?

    -   Start a document: Keras shapes. Log all errors.

    -   Find out how to save weights.

    -   Babble as we go.

-   Write code to generate babble from this.

-   Save the weights.
    -   Implement knn.
    -   Implement the analogy game. Have the computer generate analogy puzzles for me.
    -   Find the words nearest to one another.
    -   Have the computer search for word pairs whose average is near a word.

-   Learn vectors again from scratch and try to evaluate whether the two
    vectorizations agree.

-   Learn vectors from another Rust text written in Markdown (?) and evaluate
    whether the two agree.

    -   Qualitatively, does the babble look the same?

-   What about other corpora?

-   What happens if the number of dimensions of the vector space is extremely
    impoverished, like 3 or 16? What happens to loss and what happens
    qualitatively? How does it compare to doing PCA (or whatever) on the
    100-dimensional vectors?

-   What happens if we add more layers to the model? (To loss, to the learned
    vectors?)

-   Q: What is the mapping of `{word --> vector}` called?

-   Q: Why is the vector called an "embedding"?

-   Make another neural net for the same task that *starts* with vectors, i.e.
    fewer inputs because we begin by mapping each word not to a huge pile of
    zeros but a vector embedding. But keep the output layers the same. Can we
    now include more words of context? Does training this net change the
    weights of the output layer much?


## Object localization (C4W3L01)

Wow you just have the nn spit out the coordinates and size of the bounding box.

This seems bonkers.

Output of the net is `[any_object_is_present, ...bbox, ...classes]`.

Loss function is `(yhat_1 - y_1)^2` if y_1 = 0, otherwise the norm2 of the whole vector.

Or, squared error for bbox coordinates, log likelihood loss for the classes,
logistic regression loss for yhat_1, etc.


## Landmark detection (C4W3L02)

Output can be points you care about, where things are, on a picture

Uses:
- detecting emotion on a face
- AR stuff in snapchat works, putting a crown or puppy nose on someone
- pose detection


## Object detection (C4W3L03)

Sliding windows detection - sounds stupid, is in fact too slow


## Convolutional implementation of sliding windows (C4W3L04)

*   Sermanet et al., 2014, OverFeat: Integrated recognition, localization and
    detection using convolutional networks.

The idea is to turn the dense layers of your network into convolutional layers.
Just make them be convolutional with a fairly large size, maybe 7x7 or more.
Result is a whole image of probabilities, stride = the amount of pooling you've done.

Possibly still with bounding box info, but now a bounding box for every offset
in the image where you've run the algorithm. So kind of a lot of bounding boxes
and probabilities left over at the end, see L07.

Not much discussion of how actually to implement the loss function and
backpropagation for this. I wonder if you're supposed to train on nicely
cropped images, then rewire your model to do sliding windows.


## Intersection over union (C4W3L06)

Way of evaluating a bounding box produced by a neural network, relative to the
bounding box in the label.

Area of intersection divided by area of union. Kaggle often calls your output
"correct" if IoU ≥ 1/2.

No relation to IOUs as debt markers. Heh.


## Non-max Suppression (C4W3L07)

You'll end up with lots of bounding boxes. Select the max; and eliminate
overlapping bounding boxes that are not the max.

First standardize coordinates.

Discard all boxes with probability < 0.6. While any boxes remain, pick the box
with largest probability. Output that box. Discard any remaining box with IoU ≥
0.5 with the box just output.

If you are at the same time classifying things, errrr, well - he recommends
just running the algorithm n_categories times.


## Anchor boxes (C4W3L08)

https://www.youtube.com/watch?v=RTlwl2bv0Tg&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=29

*   Redmon et al., 2015, You Only Look Once: Unified real-time object detection.

Network has multiple output "ports", called anchor boxes.

    Output shape = (width, height, 1 + 4 + num_categories, num_anchor_boxes)

where the 4 is because bounding boxes are represented as (cx, cy, bw, bh).

But each one is a different shape in order to make there be a single "right
answer" where the object should go, so training works. One anchor box gets good
at recognizing objects in a horizontal orientation, and so on. When constructing training data, use IoU to decide which anchor box a given bounding-box 

This lets different output units specialize for different shapes of object.
Also lets you detect multiple things centered in the same grid cell.


## Putting it together: YOLO algorithm (C4W3L09)

YOLO = "You Only Look Once"

Hey, we got there. Object detection.

- how to label training data
- how to interpret output using non-max suppression


## Region proposals (C4W3L10)

*   Girshik et al., 2013, Rich feature hierarchies for accurate object
    detection and semantic segmentation.

*   Girshik, 2015. Fast R-CNN.

*   Ren et al., 2016. Faster R-CNN. Towards real-time object detection with
    region proposal networks.

An alternative to convolutional sliding windows.

Run a segmentation algorithm to find, say, 2000 windows in the image on which
to run your classifier. Improves your response to scale I guess.

R-CNN: Propose regions. Classify proposed regions one at a time. Output label +
bounding box. (Bounding box is within the region.)

Fast R-CNN: Propose regions. Use convolutional sliding windows to classify all
proposed regions.

Faster R-CNN: Use convolutional network to propose regions.

Still slower than YOLO.


# Course 5 - Sequence models

Two full weeks of content missing from the Youtube playlist. Tragic. I know a
little bit about RNNs and LSTM so I'll give it a shot.


## Basic models (C5W3L01)

*   Sutskever et al., 2014. Sequence to sequence learning with neural networks

*   Cho et al., 2014. Learning phrase representations using RNN encoder-decoder
    for statistical machine translation.

Video is from 2018 and talks about RNNs; I imagine that is all gone now.

Sequence-to-sequence models are useful for audio and NLP.

Encoder network reads the sentence and outputs a vector encoding the whole
sentence. Decoder network generates the translation in the target language, one
token at a time. No discussion of how to train this.

Image captioning: take a convolutional net trained to recognize things, cut off
the last layer which is about nicely formatting the output; take the raw output
of the next-to-last layer as your vector representation of what's in the image.
Feed that to a decoder RNN.

*   Mao et al., 2014. Deep captioning with multimodal recurrent neural
    networks.

*   Vinyals et al., 2014. Show and tell: neural image caption generator.

*   Karpathy and Li, 2015. Deep visual-semantic alignments for generating image
    descriptions.


## Picking the most likely sentence (C5W3L02)

P(y|x) is the probability of output sentence y given input x.

Goal: Find the y that maximizes P(y|x) for a given x. Beam search is how we do
this - next video (which is missing from the playlist).

Greedy search, picking the most likely first word first, does not work well.
(Ng offers a hand-wavy argument.)


## Beam search (C5W3L03)

Not in the playlist, ugh - found it though: https://www.youtube.com/watch?v=RLWuzLLSIgw

Parameter B = "beam width" - consider multiple likely sequences. At each
location in the output, try extending each of the B stored sequences-so-far
with each possible next word; select the B most likely combinations to carry
forward.

    P(y1,y2|x) = P(y1|x) * P(y2|x,y1)

You will be storing the whole internal state of the network times B, not just
the most likely sequences.

B=1 is greedy search.

I guess after you reach EOS in one fork, you can set that aside and proceed
with the other possibilities as long as they are more likely 


## Refining beam search (C5W3L04)

https://www.youtube.com/watch?v=gb__z7LlN_4

1.  Instead of actually multiplying probabilities (producing vanishingly small
    numbers), add up logs (producing reasonable negative numbers).

2.  To reduce the penalty for longer sentences, which is a bit extreme,
    multiply by `1 / len(y)**α` where α is a parameter in 0..1.

    Normalized log probability objective.

In production systems, beam width of 10 is common. For benchmarks, 100, 1000,
3000, but it's wasteful.

Not guaranteed to find maximum -- but you could prune instead... hrmmmm.


## Error analysis of beam search (C5W3L05)

https://www.youtube.com/watch?v=ZGUZwk7xIwk

Suppose the output of beam search `yhat` is much worse than some known good
output `y`.

You can directly compute P(y|x) and P(yhat|x). If P(y|x) is greater, the
problem is that beam search failed to find the most likely output.

If P(yhat|x) is greater, the RNN simply isn't learning the desired function.

You can do this for many examples, make a spreadsheet, and see if beam search
or the RNN is more often to blame.


## Bleu score (C5W3L06)

*   Papineni et al., 2002. Bleu: A method for automatic evaluation of machine
    translation. Recommended paper.

Bleu: bilingual evaluation understudy

This requires one or more "reference translations".

Precision: Does the machine-translation output appear in any reference?

Modified precision: Only give credit for a word appearing up to the maximum
number of times it appears in any reference; so if the output is "the the the
the the the the" and the reference is "The cat is on the mat", the modified
precision will be 2/7.

Generalizes to n-grams:

    p[n] = sum[g∈n-grams in yhat] count_clip(g) / number of n-rams in yhat

Max score is 1.0.

Bleu score on bigrams, example:

    Reference 1: The cat is on the mat.
    Reference 2: There is a cat on the mat.
    MT output: The cat the cat on the mat.

Of 6 bigrams in the output, 4 appear in a reference ("The cat" in reference 1
but only once; and "cat on" "on the" "the mat" in reference 2). Score = 4/6.

The combined Blue score is: `BP exp(1/4 * sum(p[n] for n=1 to 4))`.

The brevity penalty BP penalizes very short translations, since otherwise you
could get a score of 1.0 by guessing a single word that appears in a reference:

    min_len = min(len(r) for r in references)
    BP = 1                             if len(yhat) > min_len
         exp(1 - len(yhat) / min_len)  otherwise.

This accelerated the entire field of machine translation by creating a metric
of competition.

You can just download code for this and use it to evaluate your system.

This is not used in speech recognition as there's usually a single right answer.

## Recurrent neural networks (C5W1)

https://www.youtube.com/watch?v=IV8--Y3evjw


### Why sequence models? (C5W1L01)

sure


### Notation (C5W1L02)

-   inputs and outputs `x<t>` and `y<t>`, with the `<t>` as superscript
    (I will continue to just write `x[t]`)

-   `T_x` and `T_y`, number of tokens in the input/output

-   In a batch, `x(i)<t>` is term `t` of training example `i`,
    `T_x(i)` its length, etc.

Word representations: one-hot vectors. finite vocabulary, special token for
anything not in the vocabulary ("`<UNK>`").


### Recurrent neural network model (C5W1L03)

Why: A standard network wouldn't automatically share features learned at
different positions in text.

Explanation of the architecture. The initial input is typically a vector of
zeros. Some researchers initialize it randomly.

Ng is not going to draw the circuit diagrams. Prefers unrolled. Fine I guess.

Weights in the first example are `W_ax`, `W_aa`, and `W_ya`.

    a[0] = 0
    a[t] = g_a(W_aa @ a[t-1] + W_ax @ x[t] + b_a)

    yh[t] = g_y(W_ya @ a[t] + b_y)

Or, smooshing the weight matrices together:

    a[t] = g_a(W_a @ [a[t-1], x[t]] + b_a)

tanh apparently a pretty common choice in RNNs. "other ways to address vanishing gradient problem which we'll talk about later this week".


### Backpropagation through time (C5W1L04)

Define elementwise loss `L<T>`. The example here is named entity recognition,
so output is a 0 or 1 per token (i.e. per time step) and he uses the logistic
loss function, a.k.a. binary cross-entropy loss.

Overall loss is the sum of these terms.

To train I guess you forward-propagate a full example (complete sentence)
through the entire unrolled network, collecting intermediate computed values;
then backpropagate from the loss values, particularly backward through `W_a`.

Here there's an output at every step, which means there is a potential signal
from any word we get wrong to our weight matrices. Kind of surprising if this
works well, but let's continue.


### Different types of RNNs (C5W1L05)

"The unreasonable effectiveness of recurrent neural networks", Karpathy blog
post.

Many examples of different architectures. I wonder if the machine translation
type is hard to train, the encoder weights being so far removed from the loss.

encoder = from input natural language to vector representation

decoder = from vector representation to output natural language


### Language model and sequence generation (C5W1L06)

Probability of a sentence.

Jumps straight into an RNN actually trying to predict. Later loops back and
shows a little probability-of-x-given-y notation, kind of assumes you know it
though.

We feed `x[1]=0` to the first cycle of the RNN, predicting P(y[0]=i)
for all i; from then on `x[t]=y[t-1]`.

Softmax loss function at each time step; total loss is the sum.


### Sampling novel sequences (C5W1L07)

Basically what I did. Means my network is not learning well, not surprising
since it is not an RNN and only gets 4 tokens of context.

A disadvantage of a character-level language model is that you end up with much
longer sequences. Expensive to train and run. Oh, I guess this explains the
existence of tiktoken. Can represent all strings, more or less efficiently.

Up next, some challenges of training RNNs, like vanishing gradients.


### Vanishing gradients with RNNs (C5W1L08)

I wonder: The problem brings to mind resnets.

I wonder what structure these complex neural nets are inferring and how we
would crack them open and find out. Surely people are working on this right
now.

I wonder: You could visualize this by drawing a heatmap of ∂L/∂p for params
across the network. Would it really bear out? And is the cause that there are
just too many inputs (I doubt it; they all have an impact, one having an impact
does not take away from the others having an impact) or that the same
parameters are used many times and have contradictory effects on the output?

Vanishing gradients are the more common problem.

Exploding gradients are easier to spot and can be fixed with "gradient
clipping", a relatively robust solution. (!)


### Gated recurrent unit (GRU) (C5W1L09)

*   Cho et al., 2014. On the properties of neural machine translation.
    Encoder-decoder approaches.

*   Chung et al., 2014. Empirical evaluation of gated recurrent neural networks
    on sequence modeling.

https://www.youtube.com/watch?v=IV8--Y3evjw&t=3911s

"Modification to the RNN hidden layer
that makes it much better at capturing long-range connections
and helps a lot with the vanishing gradient problem."

There is a memory cell c being carried forward through time.

There is at each time step a candidate c~ for replacing c.

    c~[t] = tanh(w_c @ r_[c[t-1], x[t]] + b_c)

There is an update gate in [0,1].

    Γ_u = σ(w_u @ r_[c[t-1], x[t]] + b_u)

    c[t] = Γ_u * c~[t] + (1 - Γ_u) * c[t-1]

I wonder: Doesn't the signal decay slightly (but exponentially) with every time
step?

This is a "GRU unit". I imagine you have these alongside the normal units in
your RNN. As usual you can have a bunch of these units, and then `w_c` and
`w_u` are matrices and the `*`s in the last line are element-wise multiplication.
`Γ_u` tells which bits of the memory to update.

Full GRU has an additional gating factor Γ_r that controls how relevant c[t-1]
is to the computation of c~[t].

    Γ_r = σ(w_r @ r_[c[t-1], x[t]] + b_r)
    c~[t] = tanh(w_c @ r_[Γ_r * c[t-1], x[t]] + b_c)
    Γ_u = σ(w_u @ r_[c[t-1], x[t]] + b_u)
    c[t] = Γ_u * c~[t] + (1 - Γ_u) * c[t-1]

This seems unmotivated. It was discovered through experimentation with many
alternatives that this one works pretty well.

Literature contains much inconsistent notation.


### LSTM (long short-term memory) unit (C5W1L10)

*   Hochreiter & Schmidhuber 1997. Long short-term memory. "[G]oes quite a lot
    into the theory of vanishing gradients".

The equations are:

    c~[t] = tanh(w_c @ r_[a[t-1], x[t]] + b_c)
    Γ_u = σ(w_u @ r_[a[t-1], x[t]] + b_u)        # update gate
    Γ_f = σ(w_f @ r_[a[t-1], x[t]] + b_f)        # forget gate
    Γ_o = σ(w_o @ r_[a[t-1], x[t]] + b_o)        # output gate
    c[t] = Γ_u * c~[t] + Γ_f * c[t-1]
    a[t] = Γ_o * tanh(c[t])

A common variation ("peephole connection") also feeds `c[t-1]` to the four
computations that use `a[t-1]` and `x[t]`, in such a way that element 5 of
`c[t-1]` affects only the fifth element of `Γ_u` and so on.

This gives the network the option of at the same time retaining `c[t-1]` (by
setting `Γ_f = 1`) and adding `c~[t]` to it (by setting `Γ_u = 1`).

Disappointingly thin on explanation and theory. I need help understanding
what's actually going on.


### Bidirectional RNN (BRNN) (C5W1L11)

Two separate neural networks, one of which receives the input reversed. Combine
them to determine yh.

Forward propagation in the reverse half of a bidirectional RNN goes backward in
time. The graph is acyclic.

    yh[t] = g(w_y @ r_[a_fwd[t], a_rev[t]] + b_y)

The diagram has two boxes for each time step. These blocks can be RNN units, or
GRU or LSTM units.

As of 2017, BRNN w/LSTM was a reasonable first thing to try for NLP.


### Deep RNNs (C5W1L12)

It's a grid, with RNN/GRU/LSTM cells, arrows pointing "up" through the sequence
of layers from input to prediction, and "right" from earlier to later time steps.

It's also common for the final layers of the network to *not* be connected
horizontally, just a regular non-recurrent output net producing the final
prediction.

Because RNNs are very expensive to train (think about it) we don't see as many
deep recurrent layers as you would in a conventional deep neural net.

This concludes course 5 week 1.


## NLP and word embeddings (C5W2)

https://www.youtube.com/watch?v=36XuT5c9qvE

copyright struck just before i got to watch it :( The content does not appear
to be available anywhere.


## Convolutional neural networks: Applications (C4W4)

More content I didn't even know was there.

https://www.youtube.com/watch?v=-FfMVnwXrZ0&list=PLpFsSf5Dm-pd5d3rjNtIXUHT-v7bdaEIe&index=131

### Face recognition

#### What is face recognition? (C4W4L01)

Two problems:
- verification (binary yes/no)
- recognition (check an image against a database of faces)

Recognition is dramatically harder and requires verification to be very solved


#### One-shot Learning (C4W4L02)

Learning to recognize a person from a single example picture.

There is not enough training data to do this at all.

Instead, learn a "similarity" function on pictures of people.


#### Siamese network (C4W4L03)

*   Taigman et al., 2014. DeepFace: closing the gap to human level performance.

Introduces the notion of vector encodings.

The idea of running two identical CNNs on two images, and then comparing the
output, is sometimes called a Siamese network. (Silly term.)

The network learns parameters so that norm2(f(x[i]) - f(x[j])) reflects
similarity.

I wonder: Is there some theory that predicts how many dimensions you need to be
confident such a function exists?

I wonder: What's the loss function you use to learn this function? What's the
training data? I imagine a series of input pairs and boolean outputs (x[0],
x[1], y) and a single output neuron that makes use of only the norm, which has
only 2 parameters (and they might be of interest at the end) and a sigmoid
activation function. This means that all "large enough" norm2 values are mapped
extremely close to 0.

But I'm concerned about having enough training data for the most difficult
cases, where two faces are quite similar but not the same person.


#### Triplet loss (C4W4L04)

*   Schroff et al., 2015. FaceNet: A unified embedding for face recognition and
    clustering.

Oh, weird. So this is different, you're just trying to make triplets work out.
I don't know about this honestly. Shouldn't you train extra on the most
difficult cases?

OK, so you pick some margin α, and your objective is to have a margin of α
between any two pictures of two different faces. (Ng mentions support vector
machines.)

I wonder: How good are humans at this task? I would have guessed not so very
good, say 99% or 99.9% -- on the verification task. Might not be good enough
for the recognition task. I'm personally terrible at the recognition task.

Given 3 images A, P, N (anchor, positive, negative), the loss function is:

    L(A, P, N) = max(d(A, P) - d(A, N) + α, 0)
    where d(A, B) = norm2(f(A) - f(B))**2

This means if the margin is at least α, there's no gradient at all. Only
failures train.

However... I'm a little iffy on turning this into a working system, because
there's information missing. You're training so that the margin exists; but the
"volume" of the space a person's face occupies can vary from person to person.
You could train the network to 0 loss, but still fail to answer the question!

You need pairs (A, P), so the training data must include pictures that are the
same person.

("that're" spotted in the wild)

Again, only failures train. So if you choose N randomly, it will typically be a
person who looks nothing like A, your network will easily get this right, and
then you learn nothing from the example. Ng hints that you need to do adaptive
training, using the parameters in the network to find "difficult" cases in your
data set and train on those.

Some face recognition systems are trained on tens of millions of pictures. It's
very hard to find enough training data. You can download a pretrained model.


#### Face verification and binary classification (C4W4L05)

(back to the Taigman DeepFace paper)

Another way to learn the similarity function: take two pictures and do logistic
regression.

This was the most obvious idea, which I actually wrote down after watching L03.
It is slightly different in that:

-   they apply some weights to individual dimensions of the vector space;
    I assumed you'd rather have the network learn to scale those in upstream
    layers so that they're all nicely scaled in the vector representations,
    and you can actually use the norm2 on those vectors.

-   They use `abs(u[k] - v[k])` i.e. the norm1 on each dimension rather than
    the norm2. I remember from a previous video that the norm1 is sometimes
    used where tiny differences matter a lot!

    chi-squared difference along each dimension is sometimes used,
    `(u[k] - v[k])**2 / (u[k] + v[k])`, and other possibilities explored in the
    DeepFace paper.

Both this approach and the triplet loss function work quite well.


### Neural style transfer

#### What are deep ConvNets learning? (C4W4L07)

*   Zeiler and Fergus, 2013. Visualizing and understanding convolutional networks.

How to visualize:

For each unit in layer 1, go through your training data and find the nine image
patches that maximize that unit's activation. Display those.

Ng showed examples from the paper. Fascinating, it doesn't seem possible
really. Tempted to do it with the very stupid examples from the Coursera class.


#### Cost function (C4W4L08)

*   Gatys et al., 2015. A neural algorithm of artistic style.

The problem is to take a content image C and a style image S, and produce a
single image G that has the content of C and the style of S.

The loss is

    J(G) = α * J_content(C, G) + β * J_style(S, G)

Now

1. Initialize the image G randomly.

2. By gradient descent, minimize J(G).


#### Content cost function (C4W4L09)

Prediction: I think you need to train a single net on lots of images, not just
the two target images, and use that net to define `J_content` and `J_style`.
Alternatively you could train a net from scratch solely with random crops from
the two images but I don't think that will work well.

Prediction: Both `J_content` and `J_style` can be defined in terms of the
activation of all the cells in the network. Obviously `J_content` will weight
cells later in the last layers of the net more heavily while `J_style` will
focus on earlier layers. Similarity is the dot product; loss is -log that or
something. So that

    J_content(C, G) = -log dot(late_layer_activations(C), late_layer_activations(G))
    J_style(S, G) = -log dot(early_layer_activations(S), early_layer_activations(G))

Actual:

- instead of dot product (similarity) the paper uses squared norm2 (distance) so no `-log`
- not weighting layers or neurons because there's nothing to train those weights


#### Style cost function (C4W4L10)

Style is defined as the correlation between activations across channels (at a
given layer).

Style matrix:

Let `a[l][i,j,k]` = activation of layer l at pixel (i, j), channel k.

Compute a style matrix G for layer l and style image S, and it will be an
nc x nc (nc = number-of-channels) square matrix:

    G[l](S)[k1,k2] = sum_i sum_j a[l][i,j,k1] * a[l][i,j,k2]

(in linear algebra, the "G" is for "gram matrix")

"I've been using the term 'correlation'. Technically this is the unnormalized
cross-covariance."

Compute the style matrix also for G.

So the style cost function is just the Frobenius norm of the matrices squared, and the
original authors use a normalization factor `1/(2 * W_l * H_l * nc_l)**2`.

You get prettier results if you use the style function on multiple layers, weighted.

I notice: this is a lot more complicated than the content cost function. I
wondered if this means style is more sophisticated than content. But then the
network will have been trained to find the content.


### Convolutional networks in 1D or 3D

#### 1D and 3D generalizations of models (C4W4L11)

I knew all this.



## NLP: Sequence-to-sequence models (C5W3, continued)

### Attention model intuition (C5W3L07)

*   Bahdanau et al., 2014. Neural machine translation by jointly learning to
    align and translate.

OK, the basic idea here is that we will use a bidirectional RNN on the input
token sequence to generate some activations, then feed those activations to
another RNN which uses them to decide what part of the sentence to focus on.

It's still somewhat fuzzy to me because it seems like it could not possibly
work this way. There is an inherent lock-step quality to the relationship
between input tokens and output tokens.


### Attention model (C5W3L08)

*   Xu et al., 2015. Show attention and tell: neural image caption generation
    with visual attention.

OK, so you have an input bidirectional RNN and an output RNN. The output RNN
needs data, which comes in the form of a *context vector* `c`, build from the
input RNN's activations like so:

    c_i = sum_t'  α<1, t'> * a<t'>

where `a<t'>` consists of both forward and backward activations in the
bidirectional RNN at time step `t'`.

α<i, j> gives the attention we pay to input step `j` at output step `i`. These
α are called the *attention weights*, and they're computed using a small neural
network that takes `s<t-1>` (activations of the previous time step of the
output RNN) and `a<t'>` as inputs.

The softmax activation is applied to the attention weights at each output time
step.

There are two notions of time in play here; we must compute activations for the
entire input series (across all `t'`) before starting to crank the output RNN
(`t`).

The algorithm is quadratic (`nt * nt'` attention weights must be computed).

Attention is easily visualized with an input-time-vs-output-time heatmap.


### Speech recognition (C5W3L09)

A common pre-processing step is to run an FFT, "Filter-bank outputs" etc.

The human ear does something similar.

This problem was once broken down into producing phonemes and then producing
transcripts from phonemes, but that isn't how it's done now.

(as of 2017) The best commercial systems are trained on 100,000+ hours of
transcribed speech.

An attention model works for this problem. Or,

*   Graves et al., 2006. Connectionist temporal classification: Labeling
    unsegmented sequence data with recurrent neural entworks.

This is some kind of network, maybe a deep bidirectional RNN, that produces
output like `ttth_eee___ __qqqqqqquu___iii_` for the start of "the quick brown
fox". (collapse repeated characters not separated by blanks `_`). The idea is
that you'll always have plenty of output slots for all the letters you want to
spit out. After the frequency pre-processing, you'll have maybe 10 samples
(each one giving energy levels at a range of frequencies) per second.


### Trigger word detection (C5W3L10)

"literature still evolving" as of 2017.

I notice: Can't to bidirectional, at least not very much...

First slide shows an RNN that outputs 0 if the trigger word hasn't been said, 1
if it has been "recently" said; training is easier if you leave it at 1 for
some time, so training isn't so dependent on exact timing and there are
reasonable numbers of both 0s and 1s in the training data output.

That's it.


### Conclusion (C5W3L11)

OK, and it's over. I really wish I could have seen the NLP week 2 videos.
That was the part most relevant to my interests. Missed them by literally days.



## Stanford CS224N: NLP with Deep Learning, Winter 2021

### Lecture 1 - Intro & Word Vectors

https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ&index=1

"In discussions about artificial intelligence, a lot of the time people focus
on human brains, and the neurons buzzing by, and this intelligence that's meant
to be inside people's heads, but I just wanted to focus for a moment on the
role of language.

"This is kind of controversial, but it's not necessarily the case that humans
are much more intelligent than some of the higher apes like chimpanzees or
bonobos. Chimpanzees and bonobos have been shown to be able to use tools, to
make plans; and in fact chimps have much better short-term memory than human
beings do. Relative to that, if you look through the history of life on earth,
human beings developed language really recently. How recently, we kind of
actually don't know, because there's no fossils that say, here's a language
speaker. But most people estimate that language arose in human beings somewhere
in the range of 100,000 to a million years ago. Compared to the process of
evolution of life on earth, that's blinking an eyelid. But that power --
communication between human beings -- quickly set off our ascendancy over other
creatures. It's interesting that the ultimate power turned out not be posionous
fangs or being super fast or super big, but having the ability to communicate
with other members of your tribe.

"It was much more recently again that humans developed writing which allowed
knowledge to be communicated across distances of time and space. That's only
about 5,000 years old, the power of writing. In just a few thousand years, the
ability to preserve and share knowledge took us from the bronze age to the
smartphones and tablets of today."

https://www.youtube.com/watch?v=rmVRLeJRkl4&t=361s or just after, echoing _The
Origins of Life_ by Smith and Szathmary.

The lecturer's purpose here, I suppose, is to aggrandize NLP, the topic of this
course and his area of study.

----

word2vec skip-grams: Given a center word `c`, the model predicts context word
`o` with probability

    P(o|c) = softmax(dot(u[o], v[c]), dot(u, v[c]))

Note that we are learning _two_ vectors per word w: `v[w]` when w is a center
word, `u[w]` when w is a context word. This simplifies the math.

Let me just think through the entire proposed model here.

The function being computed is the likelihood of each `w` given center word `c`.

So the input to the network is a one-hot vector with `c` lit up. The first
layer produces an embedding `v[c]`, so the parameters to the first layer are
simply the vectors `v`, a matrix with `n_vec` rows and `n_words` columns;
and no nonlinear activation function.

The second layer computes dot products of `v[c]` with all vectors `u[c]`. So
the parameters to the second layer are simply the vectors `u`, a matrix with
`n_words` rows and `n_vec` columns.

Softmax is then applied to that; and categorical cross-entropy to the resulting
probability distribution. I have all those pieces in MyML.

----

Let me rethink it using only what I saw in the paper.

Let
D = dimensionality, the size of the word representation vectors;
V = vocabulary, the number of distinct tokens.

Take CBOW first. You maintain as parameters a vector per word, `D*V` numbers.
In the feedforward pass, you simply take the context words, average their
vectors, then (conceptually) compute the similarity of this average vector with
each word's representation; do softmax on this V-vector to obtain the output
distribution. I could write that. They have a hack from this paper:

*   Morin & Bengio, "Hierarchical probabilistic neural network language
    model", 2005.
    https://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf

Now skip-grams.

I think it must be that you mark 2R results as the "correct" answers, and
compute something that's like categorical cross-entropy (which I don't really
understand), but different in that multiple answers can be right.

The loss L(input, weights, expected) is

    sum(for w in words: -log(1 - abs(actual[w] - expected[w])))

where actual = the softmax above, only ...

----

I wonder if the difference between singular and plural words, i.e. the average
of `v(sock) - v(socks)` over many singular-plural noun-pairs, is mainly along a
single dimension of the vector. I think it would make sense either way. It
would be interesting to see how that plays out for the other relations Mikolov
et al. used for evaluation in the original word2vec paper. Can you rotate the
space so that each relation is on an axis? And if you do: then what?

It seems clear that 300-vectors can record only 300 orthogonal distinctions for
a given class of words... I wonder how much of the syntactic precision of GPT
is down to these vectors.

I wonder how much of the available surface area of the 300-sphere is actually
occupied with words. I wonder if you can create a random projection from that
hypersphere to the 3-sphere, for the purpose of making a UI to explore the
vectors. I wonder if that produces different results from PCA or whatever.

I wonder if you can tell when you've used too many dimensions, by the result
being flat. There must be some measure of that -- whole dimensions of a
point-set being _almost_ linearly dependent and thus superfluous. But it also
seems like training would try pretty hard to make something of those
dimensions, pushing them around in arbitrary ways. In the Bengio 2003
architecture, you could tell by the larger model having learned to ignore those
dimensions. In word2vec I don't know.

I wonder if the sphere-iness of the vectors is (a) mathematically convenient;
(b) a consequence of how they're computed; (c) theoretically relevant to the
information being represented. I don't have good intuitions for how the surface
of a sphere is really a super-rich manifold with "plenty of space" for complex
relationships to stretch out across n-1 dimensions. I keep thinking kind of
dumb thoughts, like that it doesn't make any sense to do vector math on these
things, since it's not possible for A - B + C to be on the surface of the
sphere.

----

At 41:59 he puts all the formulas on the board.

    p(o|c) = softmax (\w -> dot(u[w], v[c])) o

    J(θ) = -mean (\t -> sum (\j, -m <= j <= m, j ≠ 0 -> log P(w[t + j] | w[t])))

So putting those together

    J(u, v) = -mean (\t -> sum (\j, -m <= j <= m, j ≠ 0 ->
                  log (softmax (\w -> dot(u[w], v[j])) (t + j))))

Find the partial derivative of loss J with respect to every parameter. The ones
for v are the easy ones. I won't repeat it here. It's just the chain rule
forever.

STANFORD POINT

## Digit-vision project

A dense network can get you to 97%. A convolutional net is much slower, but even
a poor one can get you to 96% in *a single epoch*, and 99% in 5 epochs.


## Readings

### "Attention is all you need" 2017

https://arxiv.org/pdf/1706.03762v5.pdf

Page 3 of the paper clearly lays out the model. The boxes that show Multi-Head Attention
have three inbound arrows because they have three inputs: keys, values, and queries.

In the famous diagram, is it showing that the *final* output of the encoder is
used as the keys and values for each player of the decoder? That would make
sense; but then, it wouldn't necessarily have to be N=6 for both...

Section 3 explains in detail.


#### Interlude: Encoders, decoders

How to remember which is the "encoder" and which is the "decoder": In a
sequence-to-sequence model, the encoder comes first. You encode, then decode.
That's the same order as the jargon term "codec" (from "coder/decoder"). So the
"encoded-ness" is from the human's perspective.

The reason I get this mixed up is that of course from the model's perspective
it's the opposite: it has to parse (decode) tokens or what-have-you into an
internal representation it can work with, then at the end re-encode them in
some human-friendly form.

Since then I've seen the words "encode" and "represent" used in a paper about
the evolution of linguistics. So these may terms come from linguistics or
cognitive science originally.


### Ba et al. Layer normalization. 2016.

https://arxiv.org/pdf/1607.06450.pdf

An alternative to batch normalization.

Background: Batch normalization is described in Andrew Ng's course. It can be
applied to any layer in a neural net and it has two components. First, for each
mini-batch, for each neuron in the layer, compute the mean and variance of the
input to that neuron across the training samples in that mini-batch, and use
this to normalize the input so that it's centered on 0 and has a variance of 1.
Second, add learnable parameters to multiply and add to change the mean and
variance of the input. (It could instead be seen as applying to the output
of the previous layer; the present paper talks about normalizing inputs.)

The paper notes two drawbacks of batch norm:

-   It requires running averages of the input statistics. (I don't think this
    means you use these during training, because the previous paragraph talks
    about "the stochasticity from the *batch* statistics". I think it means you
    are going to need these overall statistics in order to make predictions
    after training is done. But then it wouldn't necessarily say *running*
    average. So who knows.) In an RNN, the statistics often vary with sequence
    length.

-   Can't be applied if the mini-batches are small (and sometimes they have to
    be small).

So generally the idea is, transpose this idea and normalize using the
statistics across the whole layer, separately for each training case, rather
than statistics across training cases but separately for each unit in the
model.


### Iandola et al. SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size. 2016.

https://arxiv.org/abs/1602.07360

Techniques:

-   Use the "inception" technique of using kernels of _different_ sizes in the
    same layer, and concatenate along the channel axis. This allows SqueezeNet
    to use a lot of 1x1 kernels which have 1/9 as many parameters.

    (Keras has a "Concatenate" layer that probably can do this trick, but it's
    effectively undocumented.)

-   Reduce the number of channels that flow into 3x3 filters. The basic module
    of the architecture uses alternating layers of 1. a few 1x1 kernels to
    reduce the number of channels by a factor of like 8, then 2. a mix of 1x1
    and 3x3 kernels that increases the number of channels again.

-   Postpone down-resolution (to keep precision while reducing number of
    parameters). That makes the thing more computationally expensive, right?
    Slower to train.

This alone gets them to 50x, and then they apply other compression techniques
from the literature -- as it happens, they work fine on the smaller network.
With that they get to 500x.

I'm not clear on how much stock we should put in the accuracy. It was not
explicitly stated that they had separate dev and test sets...

One cool thing I learned in this paper is a way to adapt the ResNet idea to
situations where the block shrinks the data. You can take the bypassing data
and run it through a dense layer (or if it's an image, a set of 1x1 filters)
because that can produce exactly the number of outputs (channels) you want.


### LeCun. Generalization and network design strategies. 1989.

http://yann.lecun.com/exdb/publis/pdf/lecun-89.pdf

The title is hopeful that this design has lessons that could apply in other
contexts. Today this paper would have been called "OcuLORD: A deep
convolutional network for identifying hand-written digits".

"Although various successful applications of [backpropagation] have been
described in the literature, the conditions in which good generalization
performance can be obtained are not understood." Heh. Yeah.


### Samuel et al. Trained on 100 million words and still in shape: BERT meets British National Corpus. 2023.

https://arxiv.org/pdf/2303.09859v3.pdf

I can't tell if it was trained in a single pass over the data, though, or
multiple passes. I have read elsewhere that large models like GPT-4 typically
never see the same data twice during training.


### BLiMP: The Benchmark of Linguistic Minimal Pairs for English

I looked up BLiMP, one of the instruments the previous paper used to evaluate
the model, and it strikes me as quite strange. Here are some good and bad
sentences from BLiMP:

    1.  A - There were most brothers of Aaron baking.
        B - There were many brothers of Aaron baking.

    2.  A - Some guest knows one doctor and Patricia knows two nice.
        B - Some guest knows one nice doctor and Patricia knows two.

    3.  A - Only ladies ever wave.
        B - Even ladies ever wave.

    4.  A - Diana conceals who this actress that talked fled from.
        B - Diana conceals that this actress that talked fled from.

    5.  A - Every nephew of Carol has examined one book about and William's
            husband has examined more smooth book about.
        B - Every nephew of Carol has examined one purple book about and
            William's husband has examined more smooth.

Every sentence is quite improbable a priori.

The problem is that both sentences in 5 are total gibberish.

Another whole suite of questions, I think about 0.7% of the test, check whether
you think "they" can have a singular referent; to get the questions right you
have to indicate that "he" or "she" is better.

There is something prescriptivist about any grammar test. They tried to confirm
that the test is linked to real English usage by validating it with a
crowd-sourced survey, and to their credit the repo is public and includes the
raw results of the survey.

Less flatteringly, the survey covers less than 400 of the 67,000 questions, and
still there are a handful of questions in the survey where human subjects got
them "wrong" more than half the time:

    1. (45%)
      A - It's himself that this cashier attacked.
      B - It's himself that attacked this cashier.

    2. (45%)
      A - Kayla imagines Randolf sounds like herself.
      B - Randolf imagines Kayla sounds like herself.

    3. (45%)
      A - Gary imagined most reports upset himself.
      B - Gary imagined most reports upset themselves.

    4. (40%)
      A - All convertibles weren't there existing.
      B - There weren't all convertibles existing.

    5. (40%)
      A - Kayla concealed who a lot of guests that were scaring many people
          complain about.
      B - Kayla concealed that a lot of guests that were scaring many people
          complain about.

    6. (40%)
      A - Patricia had forgotten.
      B - Patricia had changed.

    7. (43%)
      A - Some lake was passed by some cashiers.
      B - Some lake was passed by some phenomena.

Apart from these examples just being super bad, the principle supposedly being
tested in the last question is that the agent in a passive-voice sentence has
to be animate — a distinction of meaning — whereas a ton of the generated
sentences make no sense at all (reports becoming upset and whatnot) and the
human test-taker's job is to rule them out as ungrammatical. It's like a
history question showing up on a logic test.

I'm pretty sure that `animate_subject_passive` is not a real rule.

The test seems intended to get at whether the model understands passive voice.
Certain verbs require an animate subject, like "hate", "bring", "attack". Sofas
mostly don't hate, bring things, or attack (yet — Mattie note: opportunity to
disrupt). So you would typically vote for "Some majorettes attacked us." over
"Some envelopes attacked us." But these verbs are rarer in the passive, so:
does the model nonetheless realize that "We were attacked by some majorettes"
is more likely than "...by some envelopes"?

But they use verbs like "cure" that are questionable for this purpose:

    11.
      A - Kristen wasn't cured by the dress.
      B - Kristen wasn't cured by the men.

and some nouns like "phenomena" or "deer" that are questionably inanimate. And
like I said, it's weird that in most cases the good sentences are grammatical
but nonsensical, and here the bad sentence is grammatical but nonsensical.

Most examples in COCA are of people being "cured by" nouns that could hardly be
considered animate, like "science", "hypnosis", "surgery", "an unexpected
encounter with fragrant hyacinths", or "the burying of the egg near the sea in
the light of the full moon". Lamentable conditions are also occasionally
cured by people, particularly popes. But still.


### (podcast) Gradient Dissent: Scaling LLMs and Accelerating Adoption with Aidan Gomez at Cohere

https://podcast.wandb.com/episode/scaling-llms-and-accelerating-adoption-with-aidan-gomez-at-cohere

I wonder what techniques he's talking about in response to the question at
10:28. There doesn't seem to be anything called "ALIBI", except some sort of
differential privacy thing.

Did eventually find ALiBI, <https://arxiv.org/pdf/2108.12409.pdf>. I definitely
want to read it, but there is a lot more to read first.


### Peters et al. Deep contextualized word representations 2018. (ELMo)

The ELMo paper. Stands for "Embeddings from Language Models".

https://arxiv.org/pdf/1802.05365.pdf

I have so many questions.

After reading the whole paper, it seems a bit questionable even referring to
these as "embeddings". Yes, it's pretrained without reference to the task; but
"in most cases we used a fine-tuned biLM in the downstream task". It seems
possible the paper has really discovered that "model airplanes can fly at Mach
2 if you just strap an F-16 to them". That is, put a giant state-of-the-art
language model at the start of your net, go out of your way to expose its
internal activations, and your net can learn to do the last, most effortless
step, picking out the answer from among the many answers provided.

To me, ELMo reads more like transfer learning (from C4W2L09) -- chopping off
the last layers from one model to get a pre-trained half network to use for a
new task, a huge head start. But with ELMo, they let the task model decide how
many layers to truncate (by training these *s* parameters). That is cool! But
not at all how it's presented.

One way these fail to be "word embeddings" as I think of them is that the
vectors it computes must be quite different per task.

Why no diagrams? Does the author hate me? Searches for "ELMo: The Missing
Figures" turned up nothing. Similarly I want to know the exact layers here and
the number of units (= vector size) in each layer.

I'm surprised the gains were not more impressive!


### Nanda et al. Progress measures for grokking via mechanistic interpretability. Jan 2023.

https://arxiv.org/abs/2301.05217

I'm unable to tell what the network is. Where is the code?

But they cite another paper
https://transformer-circuits.pub/2021/framework/index.html that explains many
unusual conventions. The paper also has the prettiest picture I've seen of the
transformer architecture:
https://transformer-circuits.pub/2021/framework/index.html#high-level-architecture
And it is full of insights about what transformers are doing. I need to read
this. But first I think I will watch the Karpathy nanoGPT video. I want a
little hands-on experience with the things, and I think he uses PyTorch which I
also want to learn.


### Bengio et al. A neural probabilistic language model. 2003.

https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

The basic idea being explored is simultaneously training (a) word-vectors and
(b) a language model.

The training set is about 14 million words -- tiny by today's standards.

They include a bunch of practical details, like how they distributed the work
using a cluster of 32 x 2 CPUs on a low-latency LAN (...it was CPUs back then,
I guess, no GPU compute, much less cloud TPU).

A lot of ink is dedicated to showing their model is better than statistical
models based on n-grams (the state of the art at the time).

-   Q: This talks of the model in terms of a function on n words that
    approximates the probability of the last of those words, given that the
    others preceded it. I assume this in turn is approximated by a continuous
    function on word-vectors. Is that what modern LLMs do? Do they have to, for
    each word they generate, search representation-space by doing gradient
    ascent on this function to find a maximum, then do k-nn on that point in
    representation space to find a word?

    A: I don't know, but I doubt that is the algorithm. For one thing, the
    global maximum may not be near any word (I've had the experience of groping
    for a word that simply doesn't exist). And of course gradient ascent isn't
    guaranteed to find the global maximum anyway. Instead you evaluate the
    function for every possible next word, and choose the most likely. I wonder
    now if there are ways to prune this search that are effective in practice
    and don't compromise on accuracy. (Morin and Bengio 2005 gives a way that
    compromises on accuracy.)

-   Q: word2vec didn't come along until 2013, according to Wikipedia. So what is this
    paper?

    A: word2vec was special because you could train it on a billion tokens in a
    day. This was the result of optimization work that made it computationally
    cheaper. Start with Morin and Bengio 2005, "Hierarchical Probabilistic
    Neural Network Language Model", below.


### Elhage et al. A mathematical framework for transformer circuits. Dec 2021.

by many authors, for Anthropic

https://transformer-circuits.pub/2021/framework/index.html#high-level-architecture

#### What is the self-attention part of a transformer really?

I hesitated to look for a paper introducing attention or self-attention,
because it was invented with reference to RNNs, but that was silly. Still,
here's what I gather from the "Attention is all you need" paper, this
"circuits" paper, and [this blog
post](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a):

The input is a vector per token. First we multiply every vector by a
parameter-matrix to produce "values" to be queried. And I guess by a different
matrix to produce "keys", and yet another matrix to produce "queries". The one
for values needs to be square -- because there's a residual link around this
self-attention layer, we don't change the size. Keys and values have to be the
same size; I imagine those will be square too.

Next, dot each query vector with each key vector to determine how well the
query matches that key. Transformers use a scaled dot product. Then do a
softmax for each query, across all keys, to normalize. Lastly, for each token,
compute a weighted average of all values, weighted by the normalized matching
results. That's the output.


Much more to read here.


### Frederic Morin and Yoshua Bengio. Hierarchical Probabilistic Neural Network Language Model. 2005.

This paper explains the technique that made word2vec computationally feasible.
It's slow at getting to the point but does a good job explaining itself.
In short:

Consider a language model that predicts the next token. Vocabulary size is V.
The network can produce an output for each possible token (V outputs) and then
do softmax to produce probability estimates. Producing these V numbers can be
the most expensive part (and for word2vec it overwhelmingly is). But you need
them all to do softmax.

The paper proposes instead classifying all the tokens in a binary hierarchy
(like a follower of Petrus Ramus). Instead of computing probabilities for every
token, you start at the root of the hierarchy and simply pick the likelier
branch at each step. This reduces the cost from `V` to `log V`.

Since the computation does not use most parameters, backpropagation and updates
are smaller too.

A surprise is that during training, you never actually fully predict the next
token. You follow the path to the *actual* next token, rather than actually follow
any wrong turns your model would have predicted.

Another big surprise is section 4. The predictor for deciding which branch to take
is

-   linear combination of (the word-embeddings of the context tokens, plus
    an embedding representing the current node), using the same weights and
    biases across all nodes; only the node-embedding is node-specific.

-   apply tanh to that

-   scale with a per-node weight and bias (two extra parameters per node)

-   sigmoid (in place of softmax) to decide: logistic regression.

That is, the bulk of the parameters are shared across all nodes.

I think in word2vec they might use a simpler predictor.

Section 5 says how they build the binary tree, using WordNet. The first part is
straightforward enough: WordNet contains a graph of IS-A relations on
word-senses that is "almost a tree". They begin by hacking it into a tree.
Then, associate each word with a node in this tree. (But we need every word to
be at a leaf node. They don't say what is supposed to happen when a word like
"animal" is assigned to a node that has many children: "cat", "squid", etc.
Maybe you can arbitrarily sink it into one of the children.) Since the IS-A
tree isn't a binary tree, they then use k-means clustering to split the many
children of a node into two lumps. But I don't understand how that works. The
end of the paragraph is about TF/IDF scores and I don't understand it. It all
sounds rather messy. I'm sure word2vec does this differently.

I wonder:

-   Seems like this would make predictions worse sometimes. For example, if the
    next word in the text is a past participle being used after the word "the",
    probably the noun branch of the tree will dominate and the verb branch will
    be nowhere. You'll miss. Do they quantify that?

-   But I guess the model would also train to compensate. If that's possible.

-   Is something like this still used today in LLMs? Or are they so
    spectacularly expensive already that it doesn't matter? (They do have many
    dense layers and a width of like 1000.)


### Mikolov et al. (the word2vec paper) Efficient estimation of word representations in vector space. 2013.

Paper focuses on computational complexity of training to produce an exciting
new system that can "learn high quality word vectors" from a 1.6-billion token
training set in less than a day.

This mentions latent semantic analysis (LSA) and latent Dirichlet alloacation
(LDA).

Not clear on how the binary tree (binary hierarchical classification of all
tokens) is constructed. They say that it's a Huffman tree and then claim that
"frequency of words works well for obtaining classes in neural net language
models", citing a 2011 paper by Mikolov and some associates
<https://sci-hub.st/https://doi.org/10.1109/ICASSP.2011.5947611>.

The paper uses the phrase "log-linear classifier" which is a terrible way of
saying: one dense layer with biases (the linear part) followed by something
exponential, like softmax. <https://en.wikipedia.org/wiki/Log-linear_model>

> Note that the weight matrix between the input and the projection layer is
> shared for all word positions in the same way as in the NNLM.

Also awkward phrasing -- notionally, the input comes in as one-hot vectors and
gets "multiplied" by this "matrix" which is just the array of word embeddings
we're training. They mean the embeddings are the same for words regardless of
their position in the context. But of course in reality this is not used as a
"weight matrix"; no matrix multiplication happens, just an array lookup. The
one-hot vectors are not actually constructed.

It is wild to me that the skip-gram model works, because the task is super
hard, essentially impossible. Perplexity is going to be high. The distribution
to be learned, the ideal distribution, is a symmetric V×V matrix; the model has
about 2VD weights.

The stunner is Table 4, where the performance is compared to a bunch of other
NNLMs. Skip-gram gets 53% on the analogy test set. No prior art breaks 25%.
That model trained on 783M words, using a single CPU (in 2013, which seems like
a weird flex), taking about three days.


## Random musings

My new favorite ML paper is

*   Rajpurkar, Jia, & Liang. "Know what you don't know: unanswerable questions
    for SQuAD". 2018. https://arxiv.org/abs/1806.03822

For two reasons.

1.  The introduction contains a fantastic run of 6 citations of papers
    introducing new data sets and 5 citations of papers introducing innovative
    model architectures, underlining the power of contributions on the data
    side. There is a dual nature to the field that's very appealing.

2.  The paper is about artificial humility, the property I miss so much in
    ChatGPT. Someone noticed the problem and did something about it!

NLP as a field is hitting problems that have been obvious in standardized
testing (like the SAT) for a century, but NLP doesn't have the various forces
holding them back from solving them -- and the researchers working on these
problems happen to be some of the most brilliant people in the world.
