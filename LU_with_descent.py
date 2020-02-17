# test.py

# begin global
import numpy as np
import pdb
# end global

## {{{

def pivot(U,i,p):
  # want the element of the ith row "closest to 1"
  # col = np.argmin(np.abs(np.log(np.abs(U[i,i:])))) + i
  # different heruistic
  col = np.argmax(U[i,i:]) + i
  p[[i,col]] = p[[col,i]]
  U[:,[i,col]] = U[:,[col,i]]

def LU(A):
  m,n = A.shape
  L,U,p = np.zeros_like(A),np.copy(A),np.arange(m)
  for i in range(n):
    pivot(U,i,p)
    if (U[i,i] == 0): continue
    L[i:,i] = U[i:,i]/U[i,i]
    U[i+1:] -= U[i]*L[i+1:,i].reshape((n-i-1,1))
  return L,U,p

def verify_LU(A,msg=''):
  L,U,p = LU(A)
  if not np.all(np.abs(L@(U[:,p]) - A)) < 1e-8: print('LU failure %s'%msg)
  return L,U,p

#for i in range(1,15): 
#  verify_LU(np.vander(np.arange(i)))
#  verify_LU(np.arange(i**2).reshape((i,i)))

# solve Ax = b where A[index[i],i+1:] == 0
def index_sub(A,b,index):
  #print('index sub',index)
  x = np.zeros_like(b)
  for i in index:
    #print('index',i)
    diff = b[i] - A[i]@x
    if A[i,i] != 0: x[i] = diff/A[i,i]
    elif diff != 0: 
      #pdb.set_trace()
      raise Exception("index_sub failed")
  return x

# solve Lx = b where L is lower triangular
def fsub(A,b): 
  i = np.arange(A.shape[0])
  #print('fsub',A)
  return index_sub(A,b,np.arange(A.shape[0]))

# solve Ux = b where U is upper triangular
def bsub(A,b): 
  i = np.arange(A.shape[0]-1,-1,-1)
  #print('bsub',A)
  return index_sub(A,b,np.arange(A.shape[0]-1,-1,-1))

def inverse_p(p):
  q = np.empty_like(p)
  q[p] = np.arange(q.shape[0])
  return q

def solve(A,b,msg=''):
  L,U,p = verify_LU(A,msg='')
  # have A = L@(U[p])
  y = fsub(L,b)
  x_ = bsub(U,y)
  return x_[inverse_p(p)]

def verify_solve(A,b,msg=''):
  x = solve(A,b)
  residue = A@x - b
  if not np.all(residue < 1e-5): 
    print('solve fail: %s',msg,b)
    print(residue)
    print('np residue:')
    print(A@np.linalg.inv(A)@b - b)

def gradient_descent(grad,f,x,target,iterations):
    #print('beginning descent, residue %e'%(np.abs(f(x)-target)))
    step = .1
    i = 0
    while i < iterations:
        x_ = x - step*grad(x)
        if np.abs(f(x_) - target) <= np.abs(f(x) - target):
          x = x_
          step *= 1.3**2
          i += 1
        else:
          step /= 1.3
          i += 1
        if step == 0.:
          #print('step has vanished')
          break
    return x

def solve_with_descent(A,b,iters):
    x = solve(A,b)
    f = lambda y: .5*np.linalg.norm(A@y - b)**2
    grad = lambda y: A.T@(A@y - b)
    #print('residue %e' % (f(x)))
    return gradient_descent(grad,f,x,0,iters)

def descent_solver(iters):
  solver = lambda A,b: solve_with_descent(A,b,iters)
  solver.__name__ = 'descent_%d'%iters
  return solver

def verify_solve_with_descent(A,b,msg=''):
  x,iters = solve_with_descent(A,b)
  residue = .5*np.linalg.norm(A@x - b)**2
  np_residue = .5*np.linalg.norm(A@np.linalg.inv(A)@b - b)**2
  fmt = 'iters: {:3d}, res: {:2.4e}, np_res: {:2.4e}'
  #print(fmt.format(iters,residue,np_residue))

def test_solvers(solvers):
  res = lambda A,b,x: .5*np.linalg.norm(A@x-b)**2
  fmt = '{:20s}:{:3.3e}'
  for i in range(2,15): 
    try: 
      A = 1.*np.vander(np.arange(i))
      b = np.random.random(i)
      print('\n----vander %d----\n'%(i))
      for solver in solvers:
        print(fmt.format(solver.__name__,res(A,b,solver(A,b))))
    except Exception as e:
      print("caught:",e)

def np_inv_solve(A,b):
  return np.linalg.inv(A)@b

def recursive_solve(A,b,iters):
  n = b.shape[0]
  x = solve(A,b)
  i = 0
  while i < iters:
    row = i % n
    j = np.random.randint(0,n)
    step = .1
    if np.abs(A[row,j]) > 1e-3:
      _x = np.copy(x)
      _x[j] = x[j] + step*(b[row] - A[row]@x)/A[row,j]
      if np.linalg.norm(A@_x - b) < np.linalg.norm(A@x - b):
        x = _x
    i += 1
  return x

def recursive_solver(iters):
  l = lambda A,b: recursive_solve(A,b,iters)
  l.__name__ = 'recursive_%d'%iters
  return l

solvers = [
    np_inv_solve,
    solve,
    descent_solver(500),
    descent_solver(5000),
    recursive_solver(500),
    recursive_solver(5000),
]

test_solvers(solvers)

## }}}
