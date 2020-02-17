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

for i in range(1,15): 
  try : 
    verify_solve(1.*np.vander(np.arange(i)),np.random.random(i),'vander %d'%i)
    #verify_solve(np.arange(i**2).reshape((i,i)),np.random.random(i),'arange %d'%i)
  except Exception as e:
    print("caught:",e)

## }}}
