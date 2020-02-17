# linalg.py
#
# python refresher and introduction to doing linear algebra with numpy

# begin standard imports
import pdb
import numpy as np
import numpy.linalg as linalg

hr = lambda: print('\n'+40*'-'+'\n')
def _print(*args,**kwargs):
  print(*args,**kwargs)
  hr()

# end standard imports

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

## Exercise 2.2: {{{ Gram-Schmidt

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

## Exercise 2.3: {{{ QR

# Every square matrix A can be decomposed into the form
#       A = QR
# where Q is orthogonal and R is upper triangular

# Q should consist of orthonormal vectors
def gs_eliminate(Q,v):
    _v = np.copy(v)
    c = []
    for q in Q:
        c.append(np.dot(q,_v))
        _v -= c[-1]*q
    c.append(np.linalg.norm(_v))
    return _v, c

def QR(A):
    q_len,n = 0,A.shape[1]
    if n != A.shape[0] or A.ndim != 2: 
        raise ValueError('QR: A not square')
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
    if not A.ndim == 2 and A.shape[0] == A.shape[1]:
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
    testQR(np.arange(i*i,dtype=float).reshape((i,i))**2,'i:%d'%i)
    testQR(np.sin(np.arange(i*i,dtype=float).reshape((i,i))),'i:%d'%i)

## }}}

## Exercise {{{ Solving Ax = b for A upper triangular

# the technique is back substitution:
# given:  |a b|[x1] = b1
#         |0 c|[x2] = b2
#
# the last equation can be solved trivially.  Then the next to last
# can be solved by using this value, etc.

# A should be upper triangular
def backsub(A,b):
    m,n = A.shape
    r = np.zeros(m)
    m-=1
    for i in range(m,-1,-1):
        r[i] = b[i] - np.dot(r,A[i,n-m-1:])
        if A[i,i] != 0: r[i] /= A[i,i]
        elif r[i] == 0: continue
        else: raise ValueError('backsub not solvable')
    return r

A = np.array([[1,2,3],
              [0,2,1],
              [0,0,4]])
_print(A)
print("A@backsub(A,np.array([1,1,1]))")
print(A@backsub(A,np.array([1,1,1])))
print("A@backsub(A,np.array([0,1,2]))")
print(A@backsub(A,np.array([0,1,2])))
print("A@backsub(A,np.array([3,-1,1]))")
print(A@backsub(A,np.array([3,-1,1])))

# Q should consist of orthonormal vectors
def gs_eliminate(Q,v):
    _v = np.copy(v)
    c = []
    for q in Q:
        c.append(np.dot(q,_v))
        _v -= c[-1]*q
    c.append(np.linalg.norm(_v))
    return _v, c

def QR(A):
    q_len,n = 0,A.shape[1]
    if n != A.shape[0] or A.ndim != 2: 
        raise ValueError('QR: A not square')
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

def QR_Solve(A,b):
    Q,R = QR(A)
    return backsub(R,Q.T@b)

A = np.array([[1,2,3],
              [1,2,1],
              [2,7,4]],dtype=float)
_print(A)
print("A@QR_Solve([1,0,0])")
print(np.around(A@QR_Solve(A,[1,0,0]),5))
print("A@QR_Solve([1,1,0])")
print(np.around(A@QR_Solve(A,[1,1,0]),5))
print("A@QR_Solve([0,1,1])")
print(np.around(A@QR_Solve(A,[0,1,1]),5))

# PRACTICE:
# implement inversion using QR

def inv(A):
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('inv takes only square matrix')
    n = A.shape[0]
    return np.stack([QR_Solve(A,e) for e in np.identity(n)],1)

def test_inv(A):
    I_ = np.around(inv(A)@A,10)
    I = np.identity(A.shape[0])
    return np.all(np.abs(I_ - I) < 1e-5)

for i in range(2,10):
    A = np.vander(np.arange(1,i,dtype=float),i-1)
    print('i:',i,test_inv(A))

## }}}

## Exercise {{{ Solve Ax = b using LDU decomposition

def get_pivot(U,i,p):
  j = np.argmin(np.abs(np.log(U[i:,i])))
  print('get_pivot',i,j,p,U,sep='\n')
  if U[i,j] == 0: return False
  p[[i,j]] = p[[j,i]]
  U[[i,j]] = U[[j,i]]
  return True

def LU(A):
  n = A.shape[0]
  p,L,U = np.arange(n),np.zeros(A.shape),np.copy(A)
  for i in range(n-1):
    if not get_pivot(U,i,p): continue
    print('i',i)
    L[i:,i] = U[i:,i]/U[i,i]
    U[i+1:] -= U[i]*L[i+1:,i].reshape(n-i-1,1)
  return p,L,U

## }}}
