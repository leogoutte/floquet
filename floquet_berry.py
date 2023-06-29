import numpy as np
from scipy.linalg import expm, logm

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

def dvector(H,norm=True):
    """
    Returns the d-vector for a 2x2 Hermitian Hamiltonian
    """
    top = H[0,1]
    dx = np.real(top)
    dy = -np.imag(top)
    dz = (H[0,0]-H[1,1])/2
    d0 = (H[0,0]+H[1,1])/2
    dx = np.real(dx)
    dy = np.real(dy)
    dz = np.real(dz)
    if norm:
        norm = np.sqrt(dx**2+dy**2+dz**2)
        return np.array([dx/norm,dy/norm,dz/norm],dtype=float)
    else:
        return np.array([dx,dy,dz],dtype=float)

def HamiltonianPeriodic(t,kx,ky,A,B,D,M,mu,eE0,Omega,Tpump,hbar):
    """
    Hamiltonian for the periodic system
    """
    # kx = kx - eE0/(hbar*Omega) * np.exp(-t**2/(2*Tpump**2)) * np.cos(Omega*t)
    # ky = ky + eE0/(hbar*Omega) * np.exp(-t**2/(2*Tpump**2)) * np.sin(Omega*t)
    # H = A*(np.sin(kx)*s2 - np.sin(ky)*s1) + 2*D*(2-np.cos(kx)-np.cos(ky))*s0 + 2*B*(2 - M/(2*B) - np.cos(kx) - np.cos(ky))*s3 - mu*s0
    H = 2*D*(2-np.cos(kx)-np.cos(ky))*s0 + (np.sin(kx)*s2 - np.sin(ky)*s1) + (1-np.cos(kx)-np.cos(ky))*s3

    return H

def HFPeriodic(N,kx,ky,A,B,D,M,mu,eE0,Omega,Tpump,hbar):
    """
    define floquet unitary
    """
    T = 2*np.pi/Omega
    dt = T/N # time step
    U = np.array([[1,0],[0,1]],dtype=complex)

    for i in range(N):
        t_star = dt*i # for better results, presumably
        H = HamiltonianPeriodic(t=t_star,kx=kx,ky=ky,A=A,B=B,D=D,M=M,mu=mu,eE0=eE0,Omega=Omega,Tpump=Tpump,hbar=hbar)
        U_ = expm(-1j * H * dt / hbar)
        U = U_ @ U

    HF = 1j * hbar / T * logm(U)

    return HF

def States(res,band,N,A,B,D,M,mu,eE0,Omega,Tpump,hbar):
    """
    Returns a grid of states
    4 dimensions: [kx,ky, 2 band]
    occ is True (occupied band) or False (valence band)
    """
    bands = 2
    ks = np.linspace(-np.pi,np.pi,res)
    states = np.zeros((res,res,bands),dtype=complex)
    T = 2*np.pi/Omega

    for i in range(res):
        kx = ks[i]
        for j in range(res):
            ky = ks[j]
            # H = HFPeriodic(N=N,kx=kx,ky=ky,A=A,B=B,D=D,M=M,mu=mu,eE0=eE0,Omega=Omega,Tpump=Tpump,hbar=hbar)
            H_ = np.sin(kx)*s2 - np.sin(ky)*s1 + (1-np.cos(kx)-np.cos(ky))*s3
            H = 1j * hbar * logm(expm(-1j/hbar*(H_)))
            # H = 2*D*(2-np.cos(kx)-np.cos(ky))*s0 + (np.sin(kx)*s2 - np.sin(ky)*s1) + (1-np.cos(kx)-np.cos(ky))*s3
            # H = (np.sin(kx)*s2 - np.sin(ky)*s1) + (1-np.cos(kx)-np.cos(ky))*s3
            # _, waves = np.linalg.eig(H_)
            # states[i,j,:] = waves[:,band]
            d1,d2,d3 = dvector(H,norm=False)
            d1_,d2_,d3_ = dvector(H_,norm=False)
            print(d1-d1_)
            print(d2-d2_)
            print(d3-d3_)
            # print(H-H_)
            d = np.sqrt(d1**2+d2**2+d3**2)
            states[i,j,:] = np.array([d3+d,d1+1j*d2])  / np.sqrt(2*d**2 + 2*d3*d)

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

def ChernNumber(states,res):
    """
    Discrete sum over all plaquettes (n,m)
    """
    # Chern numbers
    Q = 0

    # Sum over all plaquettes
    for n in range(res):
        for m in range(res):
            Fnm = BerryFlux(n,m,states,res)
            Q += Fnm
    
    Q /= 2 * np.pi

    return np.around(Q,2)

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

