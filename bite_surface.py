import numpy as np

# define Pauli matrices
s0 = np.array([[1,0],[0,1]])
s1 = np.array([[0,1],[1,0]])
s2 = np.array([[0,-1j],[1j,0]])
s3 = np.array([[1,0],[0,-1]])

# position operator
def Position(wave,size):
    """
    position of the state in y
    Equipped to handle array where W[:,i] is ith wave
    works VVV
    todo: adapt for doubly-open system (already have code)
    """
    # make wave into what it was Born to be: probability
    prob = np.abs(wave)**2
    prob_norm = prob / np.sum(prob, axis=0)

    fac = int(wave.shape[0] / size) 

    ys = np.repeat(np.arange(size),int(fac))

    ypos = ys@prob_norm

    return np.asarray(ypos.T)

# effective Hamiltonian found from simplifying 8x8 Hamiltonian
def Heff_bulk(kx,ky,m,a,v,b):
    # diags
    H = (-m + b*(kx**2+ky**2))*np.kron(s3,s0) + a*(kx**2+ky**2)*np.kron(s0,s0) + v*kx*np.kron(s2,s1) - v*ky*np.kron(s1,s1)
    H = H.astype(complex)
    
    return H

def spectrum_eff_bulk(ky,m,a,v,b):
	res = 1000
	kx = 0
	s = 4
	ks = np.linspace(-1,1,res)
	Es = np.zeros(s*res)
	pos = np.zeros(s*res)
    
	for i,k in enumerate(ks):
	    H = Heff_bulk(kx,k,m,a,v,b)
	    E, W = np.linalg.eigvalsh(H)
	    Es[i*s:(i+1)*s] = E
	
	ks_ret = np.repeat(ks,s)

	return ks_ret,Es

def Heff_bulk(kx,ky,m,a,v,b):
	# diags
	H = (-m + b*(kx**2+ky**2))*np.kron(s3,s0) + a*(kx**2+ky**2)*np.kron(s0,s0) + v*kx*np.kron(s2,s1) - v*ky*np.kron(s1,s1)
	H = H.astype(complex)

	return H

# open the system in y, say
def Heff_open(size,kx,m,a,v,b):
	# diags
	diags = np.kron(np.eye(size),-m*np.kron(s3,s0) + a*(4-2*np.cos(kx))*np.kron(s0,s0) + v*np.sin(kx)*np.kron(s1,s2) - b*(3/4*np.sin(kx)-1/4*np.sin(3*kx))*np.kron(s2,s1))
	
	# off diags
	# nn hopping
	off1 = np.kron(np.eye(size,k=-1), -a*np.kron(s0,s0) +1j/2*v*np.kron(s1,s1) + 1j*3*b/8*np.kron(s2,s1))
	
	# nnnn hopping
	off3 = np.kron(np.eye(size,k=-3), -1j*b/8*np.kron(s2,s1))
	
	
	H = diags + off1 + off1.conj().T + off3 + off3.conj().T
	
	return H

def spectrum_eff_open(size,m,a,v,b):
	res = 1000
	s = 4
	ks = np.linspace(-1,1,res)
	Es = np.zeros(s*res*size)
	pos = np.zeros(s*res*size)
	
	for i in range(res):
		k = ks[i]
		H = Heff_open(size,k,m,a,v,b)
		E, W = np.linalg.eigh(H)
		ps = Position(W,size)
		Es[i*s*size:(i+1)*s*size] = E
		pos[i*s*size:(i+1)*s*size] = ps

	ks_ret = np.repeat(ks,s*size)
	
	return ks_ret,Es, pos

