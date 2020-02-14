# numpy_arrays.py
#
# python refresher and introduction to doing linear algebra with numpy

# begin standard imports
import numpy as np

# end standard imports

## Exercise 1: creating numpy arrays from python objects {{{

# arrays created from python array
array = np.array([1,2,3])
print(array)

# array comprehensions work well
array = np.array([x/2 for x in range(1,5)])
print(array)

# use with generator function:
def gen(n):
  i = 0;
  while i < n:
    yield i
    i += 1

array = np.array([*gen(5)])
print(array)

# PRACTICE
# print out an np array that contains the square of all odd numbers
# between 4 and 16

# possible solutions

array = np.array([5**2,7**2,9**2,11**2,13**2,15**2])
print("solution 1:", array)

array = np.array([x**2 for x in range(4,16) if x % 2])
print("solution 2:", array)

def gen(l,r):
  while l < r:
    yield l**2
    l += 2

array = np.array([*gen(5,16)])
print("solution 3:", array)

## }}}

## Exercise 2: numpy matrices from python arrays {{{

# Similar operations with multi-dimensional arrays (matrices)

# brute force
array = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(array)

# "double comprehension"
array = np.array([[x for x in range(i,j+1)] 
                    for (i,j) in [(1,3),(4,6),(7,9)]])
print(array)

# higher-level arrays

array = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(array)

# REMARK: a 3-d array is useful for image data, which contains a grid
# of rgb values

# indexing in an array starts from 0, first row then column

array = np.array([[0,1,2],[3,4,5]])
print('slice:   [0,1]:',array[0,1])
print('access: [0][1]:',array[0][1])

# more on slices later

# PRACTICE
# create the identity matrix for a given n
# HINT: make a "helper function" if necessary

# possible solutions:
def identity(n):
  array = np.array([range(n) for _ in range(n)])
  for i in range(n):
    for j in range(n):
      array[i,j] = int(i==j)
  return array

print("solution 1:\n",identity(5))

# get a one in the right place
eye = lambda i,n: np.array([0 if j != i else 1 for j in range(n)])
identity = lambda n: np.array([eye(k,n) for k in range(n)])
print("solution 2:\n",identity(5))

##  }}}

## Exercise 3: numpy array creation routines {{{

# IMPORTANT CONCEPT
# numpy arrays have a "shape," whose type is a python tuple

# an array of length n has the shape (n,)
# a matrix with R rows and C columns has shape (R,C)

print(np.array([1,2]).shape)
print(np.array([[1,2],[3,4]]).shape)
print(np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]).shape)

# when creating arrays, you usually provide a shape

array = np.empty(4) 
print("empty\n",array)

array = np.zeros((2,3))
print("zeroes\n",array)

array = np.identity(4)
print("identity\n",array)

array = np.eye(5,4)
print("eye(5,4)\n",array)

array = np.eye(4,5)
print("eye(4,5)\n",array)

# PRACTICE
#
# review the numpy documentation at:
# http://numpy.org/doc/1.18/reference/routines.array-creation.html

# How do you create an array of all zeroes with the same shape as a
# given array?

array = np.eye(4,5)
array = np.zeros_like(array)
print("solution 1:\n", array)

## }}}

## Exercise 4: basic operations {{{

# multiplication of arrays by a scalar
array = 2*np.array([1,2,3])
print(array)

array = 2*np.array(np.identity(4))
print(array)

# if two numpy arrays have a "similar shape", they can add and
# multiply each other

array1 = np.array([1,2,3])
array2 = np.array([-1,2,-3])

array = array1 + array2
print(array)

array = array1 * array2
print(array)

# CONCEPT: Broadcasting
# if you were designing numpy, what should the following operation do?

array = np.array([1,2,3]) + 1
print(array)

# what about...

array = np.array([1,2,3]) * np.array([[1],[2],[3]])
print(array)

# and what about!!!

try:
  array = np.array([1,2,3]) + np.array([1,2])
except ValueError as e:
  print('caught ValueError:',e)

# basic idea of broadcasting:
# if two axis are the same length, hadamard-style operations
# if one axis is one element, then extend it to match the other

# PRACTICE:
# what is the result of the following operation?
array = np.identity(3) + np.array([1,2,3]) + np.array([[-1],[-2],[-3]]) + 3
print(array)

# PRACTICE:
# create an array of arbitrary shape with all ones

# possible solutions:

ones = lambda i,j: np.ones((i,j))
array = ones(2,3)
print('solution 1:\n',array)

# That's cheating!
# For more practice, debug the following routine

def ones(i,j):
  cols = np.array([1 for _ in range(i)])
  rows = np.array([[1] for _ in range(j)])
  return rows * cols

array = ones(2,3)
print('solution 2:\n',array)

# solution 2:
# [[1 1]
#  [1 1]
#  [1 1]]
#
# Why is it transposed?

## }}}

## Exercise 5: numerical ranges {{{

# basic ranges:
# arange([start],stop,[step])
# linspace(start,stop,n=50,endpoint=True)
#
# advanced ranges:
# logspace
# geomspace

# arange
# NOTE: interval is of the form [start,end)

print("\n--------arange--------\n")

array = np.arange(3,8)
print(array)

array = np.arange(3,11,2)
print(array)

array = np.arange(10,2,-2)
print(array)

# PRACTICE:
# get an array with the squares of all numbers k in an arbitrary range
# whose value mod 3 is 2

# possible solutions:
def sqm3(l,r):
  l += 2 - l % 3
  array = np.arange(l,r,3)
  return array*array

array = sqm3(4,14)
print('solution 1:\n',array)

sqm3 = lambda l,r: np.array([x**2 for x in range(l,r) if x % 3 == 2])
array = sqm3(4,14)
print('solution 2:\n',array)

# linspace
# gives uniformly space points on an interval,
# i.e. an arithmetic sequence given endpoints and number of
# intermediate points

print("\n--------linspace--------\n")

array = np.linspace(0,4,5)
print(array)

array = np.linspace(0,4,10)
print(array)

array = np.linspace(0,4,10,endpoint=False)
print(array)

# if you want the "step" chosen by np.linspace, include the keyword
# argument retstep=True

array,step = np.linspace(0,1,10,retstep=True)
print(step,array)

array,step = np.linspace(0,1,10,endpoint=False,retstep=True)
print(step,array)

# PRACTICE
# implement the mid-point rule to estimate the integral of 
# x(1-x)(x-.5) = -x**3 + .5x**2 + .5x
#
# hint: sum is a python builtin function...

# possible solution

def integrate(a,b,n):
  # get left and right side of rectangles
  larray,step = np.linspace(a,b,n,retstep=True,endpoint=False)
  rarray = np.linspace(a+step,b,n)
  # midpoints
  m = (larray + rarray)/2
  # heights
  h = -1*m*m*m + .5*m*m + .5*m
  return sum(h*step)

print('[0,1] - 10:', integrate(0,1,10))
print('[0,2] - 10:', integrate(0,2,10))
print('[0,2] - 100:', integrate(0,2,100))

# logspace, geomspace
# both ranges create a geometric sequence, but...
#   logspace:  powers are given by linspace of interval
#   geomspace: enpoints are specified directly

print(np.logspace(-3,3,6,endpoint=False,base=2))
print(np.geomspace(1/8,8,6,endpoint=False))

## }}}

## Exercise 6: universal functions {{{

# consider the following:
array1 = np.array([1,2,3])
array2 = np.array([2,1,3])

# if you were designing numpy, what would the following operations do?
array = array1 / array2
print(array)

array1 = np.array([[1,2,3],[4,5,6]])
array = array1 / array2
print(array)

array = array1**2
print(array)

array = np.sin(np.linspace(0,np.pi/2,10))
print(array)

# The generalization of broadcasting straightforward operations such
# as multiplication and addition is to broadcast arbitrary functions
# These "broadcastable" functions are called "universal functions," or
# ufuncs.  All typical mathematical libarary functions are implemented
# as ufuncs in numpy, see:
# https://numpy.org/doc/1.18/reference/ufuncs.html#available-ufuncs

# PRACTICE:
# implement np.logspace using np.linspace

# possible solution:

def logspace(start,stop,num,endpoint=True,base=10.0):
  return base**np.linspace(start,stop,num,endpoint=endpoint)

print(logspace(-3,3,6,endpoint=False,base=2))
print(logspace(-3,3,6,endpoint=False,base=3))

# sometimes you will have a nice function that you would like to be
# applied like a universal function.  Say you have a function:
#     f : R^n -> R^m
# to get a ufunc for f call
#
# np.frompyfunc(f,n,m)
#
# Take, for example, the
# polynomial: x(1-x)(x+.5) = -x**3 + .5x**2 + .5x

ufunc = np.frompyfunc(lambda x:-x**3 + .5*x + .5*x, 1, 1)
array = np.linspace(0,1,5)
print(array)
print(ufunc(array))

# for multidimensional functions it's a little less obvious what to 
# do.
#
# say f(x,y) = (x*y,x+y)

ufunc = np.frompyfunc(lambda x,y: (x*y,x+y),2,2)
num = 5
yarray = xarray = np.linspace(0,2,num)
xres,yres = ufunc(xarray,yarray)

print('(x,y) -> (x*y,x+y)')
for i in range(num):
  print("(%.2f,%.2f) -> (%.2f,%.2f)" % 
            (xarray[i],yarray[i],xres[i],yres[i]))


# PRACTICE:
# generalize the midpoint rule for an arbitrary f : R -> R

# possible solution:

def integrate(f,a,b,n):
  # get left and right side of rectangles
  larray,step = np.linspace(a,b,n,retstep=True,endpoint=False)
  rarray = np.linspace(a+step,b,n)
  # midpoints
  m = (larray + rarray)/2
  # heights
  ufunc = np.frompyfunc(f,1,1)
  h = f(m)
  return sum(h*step)

# polynomial: x(1-x)(x+.5) = -x**3 + .5x**2 + .5x
polynomial = lambda x: x*(1-x)*(x+.5)
print('[0,1] - 10:', integrate(polynomial,0,1,10))

print('[0,5] - 100, exp:', integrate(np.exp,0,5,100))
print('exp(5) - 1:', np.exp(5) - 1)

print('[0,pi] - 10', integrate(lambda x: 2*np.sin(x)**2,0,np.pi,10))
print(np.pi)

## }}}
