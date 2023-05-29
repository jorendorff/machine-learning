# Playing with the np.dot() builtin

import numpy as np

a = np.array([[0, 1, 2, 3, 4]], dtype=float)
b = a.T
c = np.array([[0, 1, 2]], dtype=float)
d = np.array([0, 1, 2, 3, 4], dtype=float)

print("a = ", a) # row
print("b = ", b) # col
print("a.shape = ", a.shape)
print("b.shape = ", b.shape)
print("c.shape = ", c.shape)
print("a . b = ", np.dot(a, b)) # dot product as i know it
print("b . a = ", np.dot(b, a)) # cartesian product by *
print("b . c = ", np.dot(b, c)) # cartesian product by *

# np.dot(A, B) matches the *last* dimension of A against the *next-to-last*
# dimension of B. The result has the shape produced by dropping the matched
# dimensions from A's shape and B's shape, then stringing them together:
e = np.zeros((4, 3, 2, 1))
f = np.zeros((6, 2, 1, 5))
assert np.dot(e, f).shape == (4, 3, 2, 6, 2, 5)


# From the docs:
#
# ```
# dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
# ```
#
# According to ChatGPT, "`numpy.dot` can be described mathematically as the
# operation that computes the composition of two linear transformations
# represented by the input arrays."
#
# How rad would it be to have that in the docs?




# np.dot(A, B) must match the *last* dimension of A against the *first*
# dimension of B, and check that they match. Then, for each i, it computes
# np.sum(A[...,i] * B[i,...]). But I don't know what that `*` means.
#
# Is this matrix multiplication?
#
# For 2-D arrays, it is. The documentation for numpy.dot describes its behavior
# as a list of five cases, and I bet it is actually implemented that way. If
# there is an underlying mathematical principle, it remains undocumented.


#print("a . a = ", np.dot(a, a)) # error, shapes not aligned
#print("b . b = ", np.dot(a, a)) # error, shapes not aligned




