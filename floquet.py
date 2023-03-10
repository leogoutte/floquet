import numpy as np
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



