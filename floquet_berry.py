import numpy as np

hbar = 0.6582119569 # ev * fs

# define paulis
s0 = np.array([[1,0],[0,1]])
s1 = np.array([[0,1],[1,0]])
s2 = np.array([[0,-1j],[1j,0]])
s3 = np.array([[1,0],[0,-1]])

def BulkModel(ms,kx,ky,A2,D2,mu,eE0,Omega):
    """
    Floquet Hamiltonian
    for ms = 1, include 0 and 1
    for ms = 2, include -1, 0, 1, and 2
    for ms = 3, include -2, -1, 0, 1, 2, and 3
    etc
    """
    # set a0
    a0 = eE0/(hbar*Omega)
    # size is ms x 2
    size = int(2*ms)

    kx = np.sin(kx)
    kx2 = 2-2*np.cos(kx)
    ky = np.sin(ky)
    ky2 = 2-2*np.cos(ky)
    
    # make diagonals
    diags = np.kron(np.eye(size), D2*(kx2+ky2+a0**2)*s0 + A2*(kx*s2-ky*s1) - mu*s0)
    diags_floquet = np.zeros((2*size,2*size),dtype=float)
    for i in range(size):
        m = ms - i
        diags_floquet[2*i:2*(i+1),2*i:2*(i+1)] = m*hbar*Omega * s0
        
    diag = diags + diags_floquet
    
    # make off-diagonals
    off_diag = np.kron(np.eye(size,k=+1), -D2*a0*(kx-1j*ky)*s0 - 1j*a0/2*A2*(s1-1j*s2))
    
    # matrix
    H = diag + off_diag + off_diag.conj().T
    
    return H

def States(res,band,ms,A2,D2,mu,eE0,Omega):
    """
    Returns a grid of states
    3 dimensions: [kx,ky, 4 band]
    band is band index
    band index runs from 0 to 3, with 0 being the lowest energy band and 3 the highest
    """
    bands = int(4*ms)
    states = np.zeros((res,res,bands),dtype=complex)

    for i in range(res):
        kx = -np.pi + i * 2 * np.pi / res 
        for j in range(res):
            ky = -np.pi + j * 2 * np.pi / res 
            _, waves = np.linalg.eigh(BulkModel(ms=ms,kx=kx,ky=ky,A2=A2,D2=D2,mu=mu,eE0=eE0,Omega=Omega))
            states[i,j,:] = waves[:,band]

    return states

def uij(u,v):
    """
    Computes overlap of wavefunctions u, v
    """
    return np.dot(np.conjugate(u),v)

def BerryFlux(n,m,states,res):
    """
    Computes product
    <u_{n,m}|u_{n+1,m}><u_{n+1,m}|u_{n+1,m+1}><u_{n+1,m+1}|u_{n,m+1}><u_{n,m+1}|u_{n,m}>
    Returns the Wilson loop for a given kz
    """
    # for a given kz
    # product over neighbouring sites
    # imposing pbc by virtue of remainder division %
    W = uij(states[n,m,:],states[(n+1)%res,m,:]) 
    W *= uij(states[(n+1)%res,m,:],states[(n+1)%res,(m+1)%res,:])
    W *= uij(states[(n+1)%res,(m+1)%res,:],states[n,(m+1)%res,:])
    W *= uij(states[n,(m+1)%res,:],states[n,m,:])

    return np.arctan2(W.imag,W.real) # might be a minus sign in front

def ChernNumber(res,band,ms,A2,D2,mu,eE0,Omega):
    """
    Discrete sum over all plaquettes (n,m)
    """
    # Chern numbers
    Q = 0

    # states
    states = States(res=res,band=band,ms=ms,A2=A2,D2=D2,mu=mu,eE0=eE0,Omega=Omega)

    # Sum over all plaquettes
    for n in range(res):
        for m in range(res):
            Fnm = BerryFlux(n,m,states,res)
            Q += Fnm
    
    Q /= 2 * np.pi

    return Q

# def ChernNumber(res,band,A2,D2,mu,eE0,Omega):
#     """
#     Chern number of band
#     """
#     nc = np.zeros(res+1,dtype=float)

#     for i in range(res+1):
#         kz = -np.pi + i * 2 * np.pi / res 
#         st = States(res=res,band=band,A2=A2,D2=D2,mu=mu,eE0=eE0,Omega=Omega)
#         nc_kz = ChernNumberKz(st,res)
#         nc[i] = nc_kz
    
#     return nc

# def PhaseDiagram(res,occ=True,tx=1,ty=1,tz=1):
#     """
#     Chern number as a function of gamma and kz
#     Returns 2D array of Chern numbers
#     """
#     res += 1 # include midpoint
#     gs = np.linspace(-6,2,num=res,endpoint=True)
#     kzs = np.linspace(-np.pi,np.pi,num=res,endpoint=True)

#     CNs = np.zeros((res,res),dtype=int)

#     for i in range(res):
#         g = gs[i]
#         for j in range(res):
#             kz = kzs[j]
#             st = States(kz,res,occ=occ,tx=tx,ty=ty,tz=tz,g=g)
#             C = ChernNumberKz(st,res)
#             CNs[i,j] = C

#     return CNs

# finite system -- compute Chern number of surface BZ

# import weyl_green as wg




# def StatesFinite(size,res,index,t=1,g=0,tm=0,mu=-4,r=0):
#     """
#     Returns a grid of states
#     4 dimensions: [kx,ky,kz, 2 band]
#     occ is True (occupied band) or False (valence band)
#     """
#     # 0 if filled, 1 if valence
#     # if occ:
#     #     index = 0
#     # else:
#     #     index = 1

#     Hdim = int(2 * size) # <- 2 for spin, 2*size for size of sys

#     states = np.zeros((res,res,Hdim),dtype=complex)

#     for i in range(res):
#         kx = -np.pi + i * 2 * np.pi / res 
#         for j in range(res):
#             kz = -np.pi + j * 2 * np.pi / res 
#             Es, waves = np.linalg.eigh(wg.FullHamiltonian(size=size,kx=kx,kz=kz,t=t,g=g,tm=tm,mu=mu,r=r))
#             states[i,j,:] = waves[:,index]

#     return states

# def BerryFluxFinite(n,m,states,res):
#     """
#     Computes product
#     <u_{n,m}|u_{n+1,m}><u_{n+1,m}|u_{n+1,m+1}><u_{n+1,m+1}|u_{n,m+1}><u_{n,m+1}|u_{n,m}>
#     Returns the Wilson loop for a given kz
#     """
#     # for a given kz
#     # product over neighbouring sites
#     # imposing pbc by virtue of remainder division %
#     W = uij(states[n,m,:],states[(n+1)%res,m,:]) 
#     W *= uij(states[(n+1)%res,m,:],states[(n+1)%res,(m+1)%res,:])
#     W *= uij(states[(n+1)%res,(m+1)%res,:],states[n,(m+1)%res,:])
#     W *= uij(states[n,(m+1)%res,:],states[n,m,:])

#     return np.arctan2(W.imag,W.real) # might be a minus sign in front

# def ChernNumberFinite(size=10,res=10,index=0,t=1,g=0,tm=0,mu=-4,r=0):
#     """
#     Discrete sum over all plaquettes (n,m)
#     """
#     # Chern numbers
#     Q = 0

#     states = StatesFinite(size=size,res=res,index=index,t=t,g=g,tm=tm,mu=mu,r=r)

#     # Sum over all plaquettes
#     for n in range(res):
#         for m in range(res):
#             Fnm = BerryFluxFinite(n,m,states,res)
#             Q += Fnm
    
#     Q /= 2 * np.pi

#     return Q

