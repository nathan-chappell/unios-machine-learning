# linalg.py
#
# python refresher and introduction to doing linear algebra with numpy

# begin standard imports
import numpy as np
import numpy.linalg as linalg

hr = lambda: print('\n'+40*'-'+'\n')
def _print(*args,**kwargs):
  print(*args,**kwargs)
  hr()

# end standard imports

# linalg operations
#   multiplication
#   inverse
# algorithms
#

## Exercise 1: basic operations of numpy: {{{

# the @ operator represents matrix multiplication

a = np.ones(4)
A = np.stack([np.ones(4)*i for i in range(4)])
_print(a)
_print(A)
_print('a@A\n',a@A)
_print('A@a\n',A@a)

# numpy offers the following functions, which are all similar but different:
#
#                vectors     matrices      tensors
# __________________________________________________
# @            | <a,b>    | AA          | ???
# numpy.inner  | <a,b>    | A^T A       | ???
# numpy.dot    | <a,b>    | AA          | ???
# numpy.vdot   | <a*,b>   | A:A         | ???
# numpy.outer  | <a,b^T>  | a11*A a12*A | ???
#                         | a21*A a22*A | ???

a = np.ones(4)
A = np.stack([np.ones(4)*i for i in range(4)])
c = np.array([1 + 1j, 1])

_print('a',a,'A',A,'c',c,sep='\n')
_print('a@a',a@a,'a@A',a@A,'A@a',A@a,'A@A',A@A,'c@c',c@c,sep='\n')

_print('a',a,'A',A,'c',c,sep='\n')
_print('np.inner(a,a)',np.inner(a,a),'np.inner(a,A)',np.inner(a,A),'np.inner(A,a)',np.inner(A,a),'np.inner(A,A)',np.inner(A,A),'np.inner(c,c)',np.inner(c,c),sep='\n')

_print('a',a,'A',A,'c',c,sep='\n')
_print('np.dot(a,a)',np.dot(a,a),'np.dot(a,A)',np.dot(a,A),'np.dot(A,a)',np.dot(A,a),'np.dot(A,A)',np.dot(A,A),'np.dot(c,c)',np.dot(c,c),sep='\n')

_print('a',a,'A',A,'c',c,sep='\n')
_print('np.vdot(a,a)',np.vdot(a,a),'np.vdot(a,A) - fail','np.vdot(A,a) - fail','np.vdot(A,A)',np.vdot(A,A),'np.vdot(c,c)',np.vdot(c,c),sep='\n')

_print('a',a,'A',A,'c',c,sep='\n')
_print('np.outer(a,a)',np.outer(a,a),'np.outer(a,A)',np.outer(a,A),'np.outer(A,a)',np.outer(A,a),'np.outer(A,A)',np.outer(A,A),'np.outer(c,c)',np.outer(c,c),sep='\n')
## }}}

