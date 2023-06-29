import numpy as np
from scipy.linalg import expm, logm
from scipy.integrate import solve_ivp
from scipy.integrate import simpson
from scipy.special import jv # bessel function of the first kind

# define Pauli matrices
s0 = np.array([[1,0],[0,1]])
s1 = np.array([[0,1],[1,0]])
s2 = np.array([[0,-1j],[1j,0]])
s3 = np.array([[1,0],[0,-1]])

# hbar is a global variable
hbar = 0.6582119569 # ev * fs

# these are parameters
v = 3.6/hbar # Ang/fs
Omega = 0.12/hbar # fs^-1
Tpump = 106.16 # fs
Tprobe = 26 # fs
eE0 = 7.5e-3 # eV / Ang
mu = 0.3 # eV
wf = mu/hbar # fs^-1

# function that diagonalizes Ephi=H0phi for IC of Dirac ODE
def InitialConditions(kx,ky,v,mu):
    """
    Returns phi and E for the time-independent Hamiltonian
    phi[:,alpha] is the alpha'th eigenvector
    """
    # initial Hamiltonian
    hk = hbar*v*kx*s2 - hbar*v*ky*s1 - mu*s0
    
    # diagonalize it
    E, phi = np.linalg.eigh(hk)
    
    return E, phi

# function f in dt{psi} = f(t,psi)
def dirac(t,psi,kx,ky,v,mu,eE0,Omega,Tpump):
    # kinetic energy part
    K = hbar*v*kx*s2 - hbar*v*ky*s1 - mu*s0
    # floquet pump part
    A = -v * eE0/Omega * np.exp(-t**2/(2*Tpump**2)) * np.cos(Omega*t) * s2
    # hamiltonian
    H = K + A 
    # TDSE d psi / dt = (below)
    f = -1j/hbar * H @ psi
    return f

# enveloppe and fermi functions. 
# the latter is functionally irrelevant as we consider low T
def Enveloppe(t,t0,Tprobe):
    """
    Probe pulse enveloppe function
    """
    return np.exp(-(t-t0)**2/(2*Tprobe**2))

def Fermi(E,kT):
    """
    Fermi distribution
    """
    if kT==0:
        return np.heaviside(-E,0.5)
    else:
        f = 1/(np.exp(E/kT) + 1)
    return f

### workhorses

# integrates s(t,t0)*e^{iwt}*psi_{alpha,s}(t) and modulus squared
# this is the heart of the program
def PhotocurrentSA(psi,s,kx,ky,w,t0,v,mu,eE0,Omega,Tpump,Tprobe):
    # t-space
    t_space = psi.t
    # integrand: enveloppe * phase * psi
    integrand = Enveloppe(t_space,t0,Tprobe) * np.exp(1j*w*t_space) * psi.y[s,:]
    # integrate samples data using simpson's rule
    I = simpson(integrand,x=t_space)
    return np.abs(I)**2

# main function
def Photocurrent(psi_plus,psi_minus,kx,ky,Es,w,t0,v,mu,eE0,Omega,Tpump,Tprobe):
    """
    Main photocurrent function
    Es[i] corresponds to the energy of the minus (i=0) or plus (i=1) psi
    """
    # the phis are solved for in the main looping function (30/01/23 change)
    
    # integrate and add it up
    P = 0
    for s in range(2):
        # add up both spins
        P += Fermi(Es[0],kT=0) * PhotocurrentSA(psi_minus,s=s,kx=kx,ky=ky,w=w,t0=t0,v=v,mu=mu,eE0=eE0,Omega=Omega,Tpump=Tpump,Tprobe=Tprobe) 
        P += Fermi(Es[1],kT=0) * PhotocurrentSA(psi_plus,s=s,kx=kx,ky=ky,w=w,t0=t0,v=v,mu=mu,eE0=eE0,Omega=Omega,Tpump=Tpump,Tprobe=Tprobe)
    return P

### circular

# function f in dt{psi} = f(t,psi)
def dirac_circular(t,psi,kx,ky,v,mu,eE0,Omega,Tpump):
    # kinetic energy part
    K = hbar*v*kx*s2 - hbar*v*ky*s1 - mu*s0
    # floquet pump part
    A = -v * eE0/Omega * np.exp(-t**2/(2*Tpump**2)) * (np.cos(Omega*t) * s2 - np.sin(Omega*t) * s1) ###### SHOULD BE MINUS
    # hamiltonian
    H = K + A 
    # TDSE d psi / dt = (below)
    f = -1j/hbar * H @ psi
    return f

# function to help plotting and making arrays to visualize
def PhotocurrentArrayEnergyCircular(res_w,wp_bounds,kx,ky,t0,v,mu,eE0,Omega,Tpump,Tprobe):
    """
    Makes an array in w 
    """
    # set initial parameters
    ws = np.linspace(wp_bounds[0],wp_bounds[1],res_w)*Omega - mu/hbar # from inverting (w+wF)/Omega
    # note that the input bounds are on the normalized frequency wp = (w+wf)/Omega
    
    # diagonalize the time-independent system (at t-> -\infty)
    Es, phis = InitialConditions(kx,ky,v,mu)

    # solve the dirac equation
    psi_plus = solve_ivp(fun=dirac_circular, t_span=[-600,500], y0=phis[:,1], args=(kx,ky,v,mu,eE0,Omega,Tpump), method='RK45')
    psi_minus = solve_ivp(fun=dirac_circular, t_span=[-600,500], y0=phis[:,0], args=(kx,ky,v,mu,eE0,Omega,Tpump), method='RK45')

    P = np.zeros(res_w, dtype=float)
    for i,w in enumerate(ws):
        P[i] = Photocurrent(psi_plus=psi_plus,psi_minus=psi_minus,Es=Es,kx=kx,ky=ky,w=w,t0=t0,v=v,mu=mu,eE0=eE0,Omega=Omega,Tpump=Tpump,Tprobe=Tprobe)
            
    return ws, P

def PhotocurrentArrayWKCircular(k_dir,k_other,k_bounds,wp_bounds,t0,v,mu,eE0,Omega,Tpump,Tprobe):
    """
    Makes an array in k-w plane 
    k_dir==1 for x and 2 for y
    k_other is the value of the remaining momentum
    """
    # set initial parameters
    res = 100 # takes ~30 seconds
    res_k = res
    res_w = res
    
    # make arrays
    ks = np.linspace(k_bounds[0],k_bounds[1],res_k)
    P = np.zeros((res_k,res_w), dtype=float)
    
    # loop over k
    if k_dir == 1:
        for i,kx in enumerate(ks):
            P[i,:] = PhotocurrentArrayEnergyCircular(res_w=res_w,wp_bounds=wp_bounds,kx=kx,ky=k_other,t0=t0,v=v,mu=mu,eE0=eE0,Omega=Omega,Tpump=Tpump,Tprobe=Tprobe)[1] # second component is ps
            
    elif k_dir == 2:
        for i,ky in enumerate(ks):
            P[i,:] = PhotocurrentArrayEnergyCircular(res_w=res_w,wp_bounds=wp_bounds,kx=k_other,ky=ky,t0=t0,v=v,mu=mu,eE0=eE0,Omega=Omega,Tpump=Tpump,Tprobe=Tprobe)[1] # second component is ps
        
    return P

### polarized photocurrent

def PhotocurrentSA_polarized(psi,sigma,kx,ky,w,t0,v,mu,eE0,Omega,Tpump,Tprobe):
    # t-space
    t_space = psi.t
    # psi polarized
    psi_pol = np.dot(sigma,psi.y)
    # integrand: enveloppe * phase * psi
    integrand = Enveloppe(t_space,t0,Tprobe) * np.exp(1j*w*t_space) * psi_pol # take inner product of s and psi
    # integrate samples data using simpson's rule
    I = simpson(integrand,x=t_space)
    return np.abs(I)**2

# main function
def Photocurrent_polarized(sigma,psi_plus,psi_minus,kx,ky,Es,w,t0,v,mu,eE0,Omega,Tpump,Tprobe):
    """
    Main photocurrent function
    Es[i] corresponds to the energy of the minus (i=0) or plus (i=1) psi
    """
    # the phis are solved for in the main looping function (30/01/23 change)
    # set spin value
#     s=1 # or 1 for down and up, resp. do sigma_z first
#     sigma = np.array([1,-1]/np.sqrt(2))
    
    # integrate and add it up
    P = 0
    P += Fermi(Es[0],kT=0) * PhotocurrentSA_polarized(psi_minus,sigma=sigma,kx=kx,ky=ky,w=w,t0=t0,v=v,mu=mu,eE0=eE0,Omega=Omega,Tpump=Tpump,Tprobe=Tprobe) 
    P += Fermi(Es[1],kT=0) * PhotocurrentSA_polarized(psi_plus,sigma=sigma,kx=kx,ky=ky,w=w,t0=t0,v=v,mu=mu,eE0=eE0,Omega=Omega,Tpump=Tpump,Tprobe=Tprobe)
    return P

def PhotocurrentArrayEnergyCircular_polarized(sigma,res_w,wp_bounds,kx,ky,t0,v,mu,eE0,Omega,Tpump,Tprobe):
    """
    Makes an array in w 
    """
    # set initial parameters
    ws = np.linspace(wp_bounds[0],wp_bounds[1],res_w)*Omega - mu/hbar # from inverting (w+wF)/Omega
    # note that the input bounds are on the normalized frequency wp = (w+wf)/Omega
    
    # diagonalize the time-independent system (at t-> -\infty)
    Es, phis = InitialConditions(kx,ky,v,mu)

    # solve the dirac equation
    psi_plus = solve_ivp(fun=dirac_circular, t_span=[-600,500], y0=phis[:,1], args=(kx,ky,v,mu,eE0,Omega,Tpump), method='RK45')
    psi_minus = solve_ivp(fun=dirac_circular, t_span=[-600,500], y0=phis[:,0], args=(kx,ky,v,mu,eE0,Omega,Tpump), method='RK45')

    P = np.zeros(res_w, dtype=float)
    for i,w in enumerate(ws):
        P[i] = Photocurrent_polarized(sigma=sigma,psi_plus=psi_plus,psi_minus=psi_minus,Es=Es,kx=kx,ky=ky,w=w,t0=t0,v=v,mu=mu,eE0=eE0,Omega=Omega,Tpump=Tpump,Tprobe=Tprobe)
            
    return ws, P

def PhotocurrentArrayWKCircular_polarized(sigma,k_dir,k_other,k_bounds,wp_bounds,t0,v,mu,eE0,Omega,Tpump,Tprobe):
    """
    Makes an array in k-w plane 
    k_dir==1 for x and 2 for y
    k_other is the value of the remaining momentum
    """
    # set initial parameters
    res = 100 # takes ~30 seconds
    res_k = res
    res_w = res
    
    # make arrays
    ks = np.linspace(k_bounds[0],k_bounds[1],res_k)
    P = np.zeros((res_k,res_w), dtype=float)
    
    # loop over k
    if k_dir == 1:
        for i,kx in enumerate(ks):
            P[i,:] = PhotocurrentArrayEnergyCircular_polarized(sigma=sigma,res_w=res_w,wp_bounds=wp_bounds,kx=kx,ky=k_other,t0=t0,v=v,mu=mu,eE0=eE0,Omega=Omega,Tpump=Tpump,Tprobe=Tprobe)[1] # second component is ps
            
    elif k_dir == 2:
        for i,ky in enumerate(ks):
            P[i,:] = PhotocurrentArrayEnergyCircular_polarized(sigma=sigma,res_w=res_w,wp_bounds=wp_bounds,kx=k_other,ky=ky,t0=t0,v=v,mu=mu,eE0=eE0,Omega=Omega,Tpump=Tpump,Tprobe=Tprobe)[1] # second component is ps
        
    return P



### FLOQUET UNITARY
def HamiltonianT(t,kx,ky,A,D,mu,eE0,Omega,Tpump,hbar=1):
    """
    time-dependent Hamiltonian
    """
    kx = kx - eE0/(hbar*Omega) * np.exp(-t**2/(2*Tpump**2)) * np.cos(Omega*t)
    ky = ky + eE0/(hbar*Omega) * np.exp(-t**2/(2*Tpump**2)) * np.sin(Omega*t)
    H = D * (kx**2 + ky**2) * s0 + A * (kx * s2 - ky * s1) - mu * s0 + 0.1*s3

    return H

def HamiltonianTm(t,kx,ky,A,D,mu,m,T,Tpump,hbar=1):
    """
    time-dependent Hamiltonian
    """
    # get t reduced to set the "mass"
    t_dim = t%T
    if t_dim < T/2:
        H = D * (kx**2 + ky**2) * s0 + A * (kx * s2 - ky * s1) - mu * s0
    else:
        m = m * s3
        H_bump = m #* np.exp(-t**2/(2*Tpump**2))
        A = 0
        H = D * (kx**2 + ky**2) * s0 + A * (kx * s2 - ky * s1) - mu * s0 + H_bump # note the *0 next to A

    return H

def FloquetUnitary(N,kx,ky,A,D,mu,eE0,Omega,Tpump,hbar):
    """
    define floquet unitary
    """
    T = 2*np.pi/Omega
    dt = T/N # time step
    U = np.array([[1,0],[0,1]],dtype=complex)

    for i in range(N):
        t_star = dt*i 
        # H = HamiltonianT(t=t_star,kx=kx,ky=ky,A=A,D=D,mu=mu,eE0=eE0,Omega=Omega,Tpump=Tpump,hbar=hbar)
        H = HamiltonianTm(t=t_star,kx=kx,ky=ky,A=A,D=D,mu=mu,m=0.1,T=T,Tpump=Tpump,hbar=hbar)
        U_ = expm(-1j * H * dt / hbar)
        U = U @ U_

    return U

def FloquetEnergies(N,kx,ky,A,D,mu,eE0,Omega,Tpump,hbar):
    """
    compute floquet energies
    """
    T = 2*np.pi/Omega
    U = FloquetUnitary(N,kx,ky,A,D,mu,eE0,Omega,Tpump,hbar)
    HF = 1j * hbar / T * logm(U)
    Es =  np.linalg.eigvalsh(HF)

    return Es

def FloquetEnergiesArray(res,N,ky,A,D,mu,eE0,Omega,Tpump,hbar):
    """
    energies as a function of kx
    """
    ks = np.linspace(-0.1,0.1,res)
    Es = np.zeros((res,2), dtype=float)

    for i,k in enumerate(ks):
        E = FloquetEnergies(N=N,kx=k,ky=ky,A=A,D=D,mu=mu,eE0=eE0,Omega=Omega,Tpump=Tpump,hbar=hbar)
        Es[i,:] = np.real(E)

    return np.repeat(ks,2), Es


### BERRY CURVATURE PROGRAM

def HF_array(res,N,A,D,mu,eE0,Omega,Tpump,hbar):
    """
    Makes array of HF (kxs, kys).
    kxs are rows, kys are columns.
    """
    T = 2*np.pi/Omega
    ks = np.linspace(-0.3,0.3,num=res)
    dk = ks[1]-ks[0]

    HFs = np.zeros((res,res,2,2),dtype=complex)

    for i,kx in enumerate(ks):
        for j,ky in enumerate(ks):
            HFs[i,j,:,:] = 1j * hbar / T * logm(FloquetUnitary(N,kx,ky,A,D,mu,eE0,Omega,Tpump,hbar))

    return HFs

def HF_array_derivatives(HFs,dk):
    """
    Derivatives of HF array
    might have to do by hand (potential source of error)
    """
    all_derivatives = np.gradient(HFs,dk) # the last two elements of all_derivatives are not important (D along Hamiltonian elements)
    HFs_dkx = all_derivatives[0]
    HFs_dky = all_derivatives[1]

    return HFs_dkx, HFs_dky

# doesn't work too well (possibly because of gauge problems)
def Berry_curvature(HFs,HFs_dkx,HFs_dky):
    """
    Compute Berry curvature as a sum over the eigenstates
    """
    res = HFs.shape[0]
    ks = np.linspace(-0.3,0.3,num=res)
    dk = ks[1]-ks[0]

    BCs = np.zeros((res,res),dtype=float)

    for i,kx in enumerate(ks):
        for j,ky in enumerate(ks):
            HF = HFs[i,j,:,:]
            HF_dkx = HFs_dkx[i,j,:,:]
            HF_dky = HFs_dky[i,j,:,:]

            es, kets = np.linalg.eig(HF) 
            ket_prime, ket = kets # choose ket to be state of lowest energy band (n and not n')

            BC = 1j / (es[1]-es[0])**2 * ((ket.conj().T @ HF_dkx @ ket_prime) * (ket_prime.conj().T @ HF_dky @ ket) - (ket.conj().T @ HF_dky @ ket_prime) * (ket_prime.conj().T @ HF_dkx @ ket))
            BCs[i,j] = np.real(BC)

    return BCs

def dvector(H):
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
    norm = np.sqrt(dx**2+dy**2+dz**2)

    return np.array([dx/norm,dy/norm,dz/norm],dtype=float)

def plot_dvector(res,N,A,D,mu,eE0,Omega,Tpump,hbar):
    """
    Plots d-vector for HF in the brillouin zone
    """
    T = 2*np.pi/Omega
    ks = np.linspace(-0.1,0.1,num=res)
    dk = ks[1]-ks[0]

    dvecs = np.zeros((res,res,3),dtype=float)

    for i,ky in enumerate(ks): # ky are the rows
        for j,kx in enumerate(ks): # kx are the columns
            HF = 1j * hbar / T * logm(FloquetUnitary(N,kx,ky,A,D,mu,eE0,Omega,Tpump,hbar))
            # HF = A * (kx * s2 - ky * s1) - mu * s0 + 0.1*s3
            # HF = kx*s1+ky*s2
            dvec = dvector(HF)
            dvecs[i,j,:] = dvec

    return dvecs

# periodic function
def HamiltonianPeriodic(t,kx,ky,A,B,D,M,mu,eE0,Omega,Tpump,hbar):
    """
    Hamiltonian for the periodic system
    """
    kx = kx - eE0/(hbar*Omega) * np.exp(-t**2/(2*Tpump**2)) * np.cos(Omega*t)
    ky = ky + eE0/(hbar*Omega) * np.exp(-t**2/(2*Tpump**2)) * np.sin(Omega*t)
    H = A*(np.sin(kx)*s2 - np.sin(ky)*s1) + 2*D*(2-np.cos(kx)-np.cos(ky))*s0 + 2*B*(-M/(2*B) - np.cos(kx) - np.cos(ky))*s3 - mu*s0

    return H

def FloquetUnitaryPeriodic(N,kx,ky,A,B,D,M,mu,eE0,Omega,Tpump,hbar):
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

    return U

def SpectrumPeriodic(res,krange,N,ky,A,B,D,M,mu,eE0,Omega,Tpump,hbar):
    """
    Spectrum along kx for the periodic Hamiltonia
    """
    ks = np.linspace(-krange,krange,res)
    ks_ret = np.zeros(2*res,dtype=float)
    Es = np.zeros(2*res,dtype=float)

    for i in range(res):
        kx = ks[i]
        # H = 1j*hbar / (2*np.pi/Omega) * logm(FloquetUnitaryPeriodic(N=N,kx=kx,ky=ky,A=A,B=B,D=D,M=M,mu=mu,eE0=eE0,Omega=Omega,Tpump=Tpump,hbar=hbar))
        H = A*(np.sin(kx)*s2 - np.sin(ky)*s1) + A*(1 - np.cos(kx) - np.cos(ky))*s3
        E, Ws = np.linalg.eigh(H)
        Es[i*2:(i+1)*2] = E
        ks_ret[i*2:(i+1)*2] = np.repeat(kx,2)

    return ks_ret, Es





