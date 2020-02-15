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

## Exercise 2.1: basic linalg algorithms {{{

# Projection
# A vector v \in R^n defines a hyperplane through the origin: 
#           {x : <v,x> = 0}.  
# Given
# another vector u, what point on the hyperplane is closest to u?
#
# We minimize f(x) = ||x - u||^2 : <v,x> = 0.
#
# The lagrangian is:
# <x-u,x-u> + l(<v,x>)
#
# Setting the gradient wrt x to zero:
#
# 2(x - u) + lv = 0 => x = u - lv/2
# 
# Since <v,x> = 0, we have:
#
# <v,u-lv/2> = <v,u> - l<v,v>/2 = 0 => l = 2<v,u>/<v,v>
#
# finally, we get:
#
# x = u - v(<v,u>/<v,v>)

leastSq = lambda v: lambda u: (np.array(u) -
            np.array(v)*(np.inner(v,u)/np.inner(v,v)))

xy_proj = leastSq([0,0,1])
_print(xy_proj([1,1,1]))
_print(xy_proj([-1,1,0]))

## }}}

## Exercise 2.2: {{{

# Every basis can be orthogonalized.  For this task we use
# Grahm-Schmidt.

proj = lambda a: lambda b: np.zeros(len(b)) if b@b == 0 else b*(a@b)/(b@b)

def gram_schmidt(arrays):
    print(arrays)
    basis = []
    for a in arrays:
        #_a = a - sum([b*np.inner(a,b)/np.inner(b,b)
                         #for b in basis if np.inner(b,b) != 0])
        _a = a - sum(map(proj(a),basis))
        print(list(map(proj(a),basis)))
        if not np.all(_a == 0): basis.append(_a)
    return np.stack(basis)

A = np.array([[1,0,1],[1,1,0],[1,1,1]])
print(A)
gs = gram_schmidt(A)
_print(gs)
_print(np.round(np.inner(gs,gs)*100)/100)

## }}}

## Exercise 2.3: {{{

# Every square matrix A can be decomposed into the form
#       A = QR
# where Q is orthogonal and R is upper triangular

import pdb

def gs_eliminate(Q,v):
    _v = np.copy(v)
    c = []
    for q in Q:
        c.append(np.dot(q,_v))
        _v -= c[-1]*q
    c.append(np.linalg.norm(_v))
    return _v, c

def QR(A):
    q_len,n = 0,len(A)
    if n != len(A[0]): raise ValueError('QR: A not square')
    Q,R,I = [],np.zeros((n,n)),np.identity(n)
    for a_col in range(n):
        q,c = gs_eliminate(Q,A[:,a_col])
        if c[-1] > 1e-8: Q.append(q/c[-1])
        for i,_c in enumerate(c): 
            R[i,a_col] = _c
    for i in range(n):
        if len(Q) == n: break
        q,c = gs_eliminate(Q,I[:,i])
        if c[-1] > 1e-8: Q.append(q/c[-1])
    return np.stack(Q).T,R

def QR_verify(A):
    if not len(A.shape) == 2 and A.shape[0] == A.shape[1]:
        raise ValueError('QR requires a square matrix')
    Q,R = QR(A)
    I = np.identity(A.shape[0])
    success = True
    if not np.all((Q.T@Q - I) < 1e-5) or not np.all((Q@Q.T - I) < 1e-5):
        print('Q not orthogonal')
        return None,None
    if not np.all((Q@R - A) < 1e-5):
        print('QR bad decomposition')
        return None,None
    return Q,R

def testQR(A,*msg):
    Q,R = QR_verify(A)
    if not Q is None and not R is None: print('success:',*msg)
    else: print('fail:',*msg)

for i in range(1,10):
    testQR(np.arange(i*i,dtype=float).reshape((i,i)),'i:%d'%i)

## }}}
