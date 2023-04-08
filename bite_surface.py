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

# rotation matrix
def rotate_matrix_spin(matrix,tau_sigma,direction):
    """
    Matrix for unitary transformation sigma_z -> sigma_x
    0 for tau or 1 for sigma
    direction = x (0) or y (1)
    """
    size = int(matrix.shape[0] / 4)
    U_pos = np.eye(size)
    U_other = np.eye(2) 

    # x direction
    if direction==0:
        U_spin = 1 / np.sqrt(2) * (s0 - 1j * s2)
    elif direction==1:
        U_spin = 1 / np.sqrt(2) * (s0 + 1j * s3)

    if tau_sigma==0:
        # rotate sigmas
        U = np.kron(U_pos, np.kron(U_other,U_spin))
    elif tau_sigma==1:
        # roate taus
        U = np.kron(U_pos, np.kron(U_spin,U_other)) 

    rot_matrix = U @ matrix @ U.conj().T    
    
    return rot_matrix

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


### 3D TI model from Qi and Zhang

def hamiltonian_3DTI(kx,ky,kz,A1,B1,A2,B2,C,D1,D2,M,R1=45.02,R2=-89.37):
    """
    Hamiltonian for 3DTI BiTe model
    """
    kp2 = 4-2*(np.cos(kx)+np.cos(ky))
    kz2 = 2-2*np.cos(kz)
    kplus = np.sin(kx)+1j*np.sin(ky)
    kminus = np.sin(kx)-1j*np.sin(ky)

    H0 = (C + D1*kz2)*np.kron(s0,s0) + (M+B1*kz2)*np.kron(s0,s3) + A1*np.sin(kz)*np.kron(s0,s2)
    H1 = D2*kp2*np.kron(s0,s0) + B2*kp2*np.kron(s0,s3) + A2*np.kron(np.sin(ky)*s1-np.sin(kx)*s2,s1)
    H3 = R1/2*(kplus**3+kminus**3)*np.kron(s0,-s2) + R2/2*(kplus**3-kminus**3)*np.kron(-s3,1j*s1)

    return H0 + H1 + H3

def spectrum_3DTI(ky,kz,A1,B1,A2,B2,C,D1,D2,M,R1,R2):
    res = 1000
    s = 4
    ks = np.linspace(-np.pi,np.pi,res)
    Es = np.zeros(s*res)
    
    for i in range(res):
        k = ks[i]
        H = hamiltonian_3DTI(kx=k,ky=ky,kz=kz,A1=A1,B1=B2,A2=A2,B2=B2,C=C,D1=D1,D2=D2,M=M,R1=R1,R2=R2)
        E = np.linalg.eigvalsh(H)
        Es[i*s:(i+1)*s] = E

    ks_ret = np.repeat(ks,s)
    
    return ks_ret,Es

def spectral_function_bulk(w,kx,ky,kz,A1,B1,A2,B2,C,D1,D2,M,R1,R2,side=0):
    # set number of dofs
    s = 4

    # define the hamiltonian
    H = hamiltonian_3DTI(kx=kx,ky=ky,kz=kz,A1=A1,B1=B1,A2=A2,B2=B2,C=C,D1=D1,D2=D2,M=M,R1=R1,R2=R2)

    # green function
    epsilon = 0.03
    G = np.linalg.inv((w + 1j*epsilon)*np.eye(s) - H)

    # sides

    if side==1:
        # take only the points on the right surface
        G_right = G[-s:,-s:] 
        # G_right = rotate_matrix_spin(G_right,tau_sigma=1,direction=0)[-2:,-2:]
        A = -1/np.pi * np.imag(np.trace(G_right))

    elif side==-1:
        # take only the points on the left surface
        G_left = G[:s,:s] 
        A = -1/np.pi * np.imag(np.trace(G_left))

    else:
        # full spectral function
        A = -1/np.pi * np.trace(np.imag(G))

    return A

def spectral_function_plot_bulk(ky,kz,A1,B1,A2,B2,C,D1,D2,M,R1,R2,side=0):
    res = 100
    ws = np.linspace(-1,1.,res)
    ks = np.linspace(-0.5,0.5,res)
    As = np.zeros((res,res),dtype=float)

    for i in range(res):
        w = ws[i]
        for j in range(res):
            k = ks[j]
            A = spectral_function_bulk(w=w,kx=k,ky=ky,kz=kz,A1=A1,B1=B1,A2=A2,B2=B2,C=C,D1=D1,D2=D2,M=M,R1=R1,R2=R2,side=side)
            As[i,j] = A
    return As

def hamiltonian_3DTI_open(size,kx,ky,A1,B1,A2,B2,C,D1,D2,M,R1,R2):
    """
    Hamiltonian for 3DTI BiTe model
    """
    # t0s0 = np.kron(s0,s0)
    # t0s3 = np.kron(s0,s3)
    # t3s2 = np.kron(s3,s2)
    # t1s2 = np.kron(s1,s2)
    # t2s2 = np.kron(s2,s2)

    g0 = np.kron(s0,s0) # constant
    g1 = np.kron(s1,s1) # linear kx/ky
    g2 = np.kron(s2,s1) # linear ky/kx
    g4 = np.kron(s0,s2) # linear kz (s3,s1)
    g5 = np.kron(s0,s3) # mass

    kp2 = 4-2*np.cos(kx)-2*np.cos(ky)

    # diagonals
    diags = np.kron(np.eye(size),(C+2*D1 + D2*(kp2))*g0 + (M+2*B1+B2*(kp2))*g5 + A2*np.sin(ky)*g1 - A2*np.sin(kx)*g2)
    kplus = kx + 1j*ky
    kminus = kx - 1j*ky
    diags_3 = np.kron(np.eye(size), R1/2*(kplus**3+kminus**3)*np.kron(s0,-s2) + R2/2*(kplus**3-kminus**3)*np.kron(-s3,1j*s1))

    # off-diags
    upper_diag = np.kron(np.eye(size,k=+1), -D1*g0 - B1*g5 + 1j/2*A1*g4)

    # H
    H = diags + upper_diag + upper_diag.conj().T #+ diags_3

    return H

def spectrum_3DTI_open(size,ky,A1,B1,A2,B2,C,D1,D2,M,R1,R2):
    res = 100
    s = 4*size
    ks = np.linspace(-0.5,0.5,res)
    Es = np.zeros(s*res)
    pos = np.zeros(s*res)

    for i in range(res):
        k = ks[i]
        H = hamiltonian_3DTI_open(size=size,kx=k,ky=ky,A1=A1,B1=B2,A2=A2,B2=B2,C=C,D1=D1,D2=D2,M=M,R1=R1,R2=R2)
        E, W = np.linalg.eigh(H)
        ps = Position(W,size)
        Es[i*s:(i+1)*s] = E
        pos[i*s:(i+1)*s] = ps

    ks_ret = np.repeat(ks,s)
    
    return ks_ret,Es,pos

# spectral function
def spectral_function_open(size,w,kx,ky,A1,B1,A2,B2,C,D1,D2,M,R1,R2,side=0):
    # set number of dofs
    s = 4

    # define the hamiltonian
    H = hamiltonian_3DTI_open(size=size,kx=kx,ky=ky,A1=A1,B1=B1,A2=A2,B2=B2,C=C,D1=D1,D2=D2,M=M,R1=R1,R2=R2)

    # green function
    epsilon = 0.1
    G = np.linalg.inv((w + 1j*epsilon)*np.eye(s*size) - H)

    # sides

    if side==1:
        # take only the points on the right surface
        G_right = G[-s:,-s:] 
        G_right = rotate_matrix_spin(G_right,tau_sigma=1,direction=0)[-2:,-2:]
        A = -1/np.pi * np.imag(np.trace(G_right))

    elif side==-1:
        # take only the points on the left surface
        G_left = G[:s,:s] 
        A = -1/np.pi * np.imag(np.trace(G_left))

    else:
        # full spectral function
        A = -1/np.pi * np.imag(np.trace(G))

    return A

def spectral_function_plot_open(size,res,w_range,ky,A1,B1,A2,B2,C,D1,D2,M,R1,R2,side=0):
    ws = np.linspace(-w_range,w_range,res)
    ks = np.linspace(-0.5,0.5,res)
    As = np.zeros((res,res),dtype=float)

    for i in range(res):
        w = ws[i]
        for j in range(res):
            k = ks[j]
            A = spectral_function_open(size=size,w=w,kx=k,ky=ky,A1=A1,B1=B1,A2=A2,B2=B2,C=C,D1=D1,D2=D2,M=M,R1=R1,R2=R2,side=side)
            As[i,j] = A
    return As

# surface theory results
def hamiltonian_3DTI_surface(kx,ky,A1,B1,A2,B2,C,D1,D2,M,R1,R2):
    """
    hamiltonian for the surface of a 3DTI
    """
    kplus = np.sin(kx)+1j*np.sin(ky)
    kminus = np.sin(kx)-1j*np.sin(ky)
    kp2 = 4-2*(np.cos(kx)+np.cos(ky))
    H = (C + D2*(kp2))*s0 + A2*(np.sin(kx)*s1 + np.sin(ky)*s2) + R1/2*(kplus**3+kminus**3)*s3

    return H

def spectrum_3DTI_surface(ky,A1,B1,A2,B2,C,D1,D2,M,R1,R2):
    res = 1000
    s = 2
    ks = np.linspace(-1,1,res)
    Es = np.zeros(s*res)
    
    for i in range(res):
        k = ks[i]
        H = hamiltonian_3DTI_surface(kx=k,ky=ky,A1=A1,B1=B2,A2=A2,B2=B2,C=C,D1=D1,D2=D2,M=M,R1=R1,R2=R2)
        E = np.linalg.eigvalsh(H)
        Es[i*s:(i+1)*s] = E

    ks_ret = np.repeat(ks,s)
    
    return ks_ret,Es









# Script to run on cluster

if __name__ == "__main__":
    import sys

    # get k from argv
    # args = sys.argv
    # k_idx = int(args[1])
    # k = ks[k_idx]
    
    As = spectral_function_plot_open(size=100,ky=0,A1=2.26,B1=6.86,A2=3.33,B2=44.5,C=0,D1=5.74,D2=30.4,M=-0.50,R1=50.6,R2=-113.3,side=0)

    np.savetxt("spectral_function_open.csv", As, delimiter = ",")


### LPBG