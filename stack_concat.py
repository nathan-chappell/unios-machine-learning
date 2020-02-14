# stack_concat.py
#
# python refresher and introduction to doing linear algebra with numpy

# begin standard imports
import numpy as np

hr = lambda: print('\n'+40*'-'+'\n')
def _print(*args,**kwargs):
  print(*args,**kwargs)
  hr()

# end standard imports

## Exercise 1: creating matrices: SHAPE {{{

# CONCEPT: axis
# numpy arrays have shapes, such as (3,2,4).  Mathematically, if a in
# a numpy array and a.shape == (3,2,4), then
#
#     a \in R^3 x R^2 x R^4
#
# Each "dimension of the shape" is called an "axis."  Many numpy
# operations take an axis as a keyword argument, and modify their
# operations to proceed along that axis.  We'll see this later when we
# construct arrays from existing arrays.
#
# In reality, every numpy array is just a C-style array, a one
# dimensional array in memory somewhere, and its shape is a logical
# construct that effects operations.  Therefore, the shape can be
# changed at will

# The following matrix is conceptually 3 2x4 matrices...
a = np.arange(24).reshape(3,2,4)
_print('a.shape: {}, a:'.format(a.shape),a,sep='\n')

# While now it is 2 3x4 matrices...
a = a.reshape(2,3,4)
_print('a.shape: {}, a:'.format(a.shape),a,sep='\n')

## }}}

## {{{

# operations like reshape swapaxes, moveaxis, and transpose all
# operate on the shape of the arrays...

a = np.arange(2*3*4*5*6)
b = a.reshape(2,3,4,5,6)
c = np.moveaxis(b,0,2)
d = np.transpose(b)

_print('a',a.shape)
_print('b',b.shape)
_print('c',c.shape)
_print('d',d.shape)

# for linear algebra, usually transpose is enough.  Notice that what
# transpose "actually does" is reverse the order of the shapes.  There
# is a shorthand for np.transpose(a) - a.T
# Therefore, if we suppose that len(a.shape) == 2, then the following
# are all equivalent:
#
# a.T
# np.transpose(a)
# np.swapaxes(a,0,1)
# np.moveaxis(a,1,0)
#

a = np.arange(9).reshape(3,3)

truth = [
  a.T == np.transpose(a),
  a.T == np.swapaxes(a,0,1),
  a.T == np.moveaxis(a,1,0),
  np.transpose(a) == np.swapaxes(a,0,1),
  np.transpose(a) == np.moveaxis(a,1,0),
  np.swapaxes(a,0,1) == np.moveaxis(a,1,0),
]

_print('all transpose ops the same?', np.all(truth))

## }}}

## PRACTICE:  {{{
# use create a matrix m of shape (m,n) such that
#  m[i,j] == -1  |  i+j is odd
#  m[i,j] ==  1  |  i+j is even

# possible solution

def makeM(m,n):
  M = np.arange(m*n).reshape((m,n))
  if not n % 2: M += M // n
  return (-1)**M

_print('(3,4)', makeM(3,4), sep='\n')
_print('(2,3)', makeM(2,3), sep='\n')

## }}}

## Exercise 2: np.stack {{{
# if we have some arrays, we may like to build other arrays or
# matrices from them.  Two useful operations to accomplish this are
# stack and concatenate.
#
# stack takes some arrays and "stacks them up" along a NEW axis. i.e.
#     stack : (R^n1 x R^n2 x ...)^m -> R^m x R^n1 x R^n2 ...
#
# By default, it will stack arrays along a new "outermost" axis, but
# this can be changed with the keyword argument axis

a = np.arange(4)
_print(np.stack((a,a,a)))
_print(np.stack((a,a,a), axis=1))

a = np.arange(3*4).reshape((3,4))
b = np.arange(3*4).reshape((3,4))

_print('axis=0',np.stack((a,b)),sep='\n')
_print('axis=1',np.stack((a,b),axis=1),sep='\n')
_print('axis=2',np.stack((a,b),axis=2),sep='\n')

## }}}

## PRACTICE: {{{
# use create a matrix m of shape (m,n) such that
#  m[i,j] == -1  |  i+j is odd
#  m[i,j] ==  1  |  i+j is even

# possible solution:

makeM = lambda m,n: np.stack([np.arange(n)+i for i in range(m)])

## }}}

## PRACTICE:{{{
# implement the stack operation for two input arrays

# possible solution:

class DoneInc(Exception): pass

def incPos(curPos,endPos,d):
  if d >= len(endPos): raise DoneInc('done')
  _d = -1-d
  if curPos[_d] == endPos[_d]-1:
    curPos[_d] = 0
    incPos(curPos,endPos,d+1)
  else:
    curPos[_d] += 1

def stack(arrays,axis=0):
  if len(arrays) == 0: return np.array([])
  n = len(arrays)
  if not np.all([arrays[0].shape == a.shape for a in arrays]):
    raise ValueError('all shapes must be the same')
  outShape = arrays[0].shape[:axis] + (n,) + arrays[0].shape[axis:]
  out = np.empty(outShape)
  curPos = [0]*(len(outShape))
  endPos = list(outShape)
  try:
    while True:
      pos = tuple(curPos[:axis] + curPos[axis+1:])
      out[tuple(curPos)] = arrays[curPos[axis]][pos]
      incPos(curPos,endPos,0)
  except DoneInc:
    return out
  raise RuntimeError('unreachable')

a = np.arange(2*5).reshape(2,5)
b = np.arange(2*3*4).reshape(2,3,4)

_print('\n-------- a,a,a,a, 1 --------\n')
_print(stack((a,a,a,a),1))
_print(np.stack((a,a,a,a),1))
_print('\n-------- b,b 1 --------\n')
_print(stack((b,b),1))
_print(np.stack((b,b),1))
_print('\n-------- b,b 2 --------\n')
_print(stack((b,b),2))
_print(np.stack((b,b),2))

# the numpy solution is a more elegant, but more esoteric:
# https://github.com/numpy/numpy/blob/v1.13.3/numpy/core/shape_base.py#L296-L361

## }}}

## Exercise 3: concatenate {{{

# concatenate is almost the same as stack, but the difference is that
# no new axis is created, rather an existing axis is extended

a = np.arange(2*3).reshape((2,3))
_print(np.concatenate((a,a,a)))
_print(np.concatenate((a,a,a),axis=1))

## }}}

## PRACTICE:{{{
#
# relate concatenation and the concept of column/row space of a matrix
#
# compare stack and concatenate, how are they similar?  How are they
# different?  How do the underlying arrays look compared to one
# another?  Try the following:

_print("\n-------- STACK VS CONCAT --------\n")

a = np.arange(6).reshape((2,3))
b = np.arange(6,12).reshape((2,3))

_print(np.concatenate((a,a,b)).reshape((3,2,3)))
_print(np.stack((a,a,b)))

_print("\n-------- STACK VS CONCAT VS AXIS--------\n")

#
# why don't the following two commands produce the same result?
_print(np.concatenate((a,a,b),1).reshape((3,2,3)))
_print(np.stack((a,a,b),1))

## }}}

## PRACTICE: {{{
# reimplement (more simply) stack
# hint: use concatenate and reshape

# possible solutions:

def stack(arrays,axis=0):
  if len(arrays) == 0: 
    raise ValueError('need at least one array')
  if not np.all([arrays[0].shape == a.shape for a in arrays]):
    raise ValueError('all arrays must be same shape')
  shape = arrays[0].shape
  new_shape = shape[:axis] + (len(arrays),) + shape[axis:]
  return np.concatenate(arrays,axis=axis).reshape(new_shape)

a = np.arange(6).reshape((2,3))
b = np.arange(6,12).reshape((2,3))

_print(stack((a,a,b)))
_print(np.stack((a,a,b)))
_print(stack((a,a,b),1))
_print(np.stack((a,a,b),1))

## }}}
