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
