import numpy as np
import numpy.linalg as la
import petsc_io as io
#import scipy as sp
import matplotlib as plt
import pyDOE
import modred as mr


class VecHandlePetsc(mr.VecHandle):
    """Gets and puts array vector objects from/in text files."""
    def __init__(self, vec_path, base_vec_handle=None, scale=None):
        mr.VecHandle.__init__(self, base_vec_handle, scale)
        self.vec_path = vec_path

    def _get(self):
        """Loads vector from path."""
        return io.read_PETSc_vec(self.vec_path)

    def _put(self, vec):
        """Saves vector to path."""
        io.write_PETSc_vec(vec, self.vec_path)

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return self.vec_path == other.vec_path


def constructSnapshotMatrix(path,pattern,nspinup,ntimestep,starttimestep = 1):
    ''' returns an matix of Snapshots,
    Params: nspinup: number of spinups
            ntimestep: number of timesteps
            path: loding directory
            distribution: string code of distributions
    '''
    ny = 52749
    Y_N = np.empty([ny,np.int_(nspinup *(ntimestep-starttimestep+1))],dtype='float_')
    #Y_DOP = np.empty([ny,np.int_(ndistribution * nspinup *(ntimestep-starttimestep))],dtype='float_')

    counter = 0
    for s in range(0,nspinup):
            #time step
        for i in range(starttimestep,ntimestep+1):

            Y_N[:,counter] = io.read_PETSc_vec(path+pattern % (s,240*i-1))
            counter+=1
    return Y_N

def computeSVD(Y):
    U, s, V = la.svd(Y,full_matrices=False)
    return U, s, V

def plotSingular(s,eps):
    ''' computes the index dependent on eps
        and plot the singular values
    '''
    plt.pyplot.plot(s)
    plt.pyplot.yscale('log')
    #plt.pyplot.xscale('log')
    plt.pyplot.xlabel('Sigma i')
    plt.pyplot.ylabel('Value')
    sigmaSum_r = 0
    sigmaSum = sum(s)
    for i,j in enumerate(s):
        sigmaSum_r += j
        if  sigmaSum_r/sigmaSum >= 1- eps*eps:
            index = i
            break

    plt.pyplot.axvline(ls= ':',x=index,color='r')
    plt.pyplot.text(index, 1, 'eps='+str(eps))
    plt.pyplot.title('Singularvalues of SVD')
    plt.pyplot.show()
    return index

def generate_prameter_samples(n,p):
        ''' generates Latin-Hypercubesamples with pyDOE and
        transforms bounds.
        n: number of samples
        return: 2d-array of parameter samples (n,7)
        '''
        if p==7:
            b=np.array([0.75,200,0.95,1.5,50,0.05,1.5])
            a=np.array([0.25,1.5,0.05,0.25,10,0.01,0.7])
        elif p==5 :
            b=np.array([0.05,200,1.5,50,1.5])
            a=np.array([0.01,1.5,0.25,10,0.7])
        else:
            return 1
        lh = pyDOE.lhs(p, samples=n)
        #upper and lower bound of parameters un MIT-gcm-PO4-DOP
 
        lh = lh*(b-a)+a
        return lh

def parse_config(filename):
    COMMENT_CHAR = '#'
    OPTION_CHAR =  '-'
    options = {}
    f = open(filename)
    for line in f:
        # First, remove comments:
        if COMMENT_CHAR in line:
            # split on comment char, keep only the part before
            line, comment = line.split(COMMENT_CHAR, 1)
        # Second, find lines with an option=value:
        if OPTION_CHAR in line:
            # split on option char:
            option, value = line.split(' ', 1)

            # strip spaces:
            option = option.strip()
            value = value.strip().replace("$","%")
            
            value = value.split(",")
            #print(value)
            # store in dictionary:
            options[option] = value
    f.close()
    return options

def plotVecOnWorldmap(path,vec=None):
    # longitude
    dx = 2.8125
    xx = np.concatenate([np.arange(-180 + 0.5 * dx, 0, dx), np.arange(0 + 0.5 * dx, 180, dx)])
    # latitude
    dy = 2.8125
    yy = np.arange(-90 + 0.5 * dy, 90, dy)
    # color range
    cmin, cmax = [0.0, 4.5]
    aspect = 1.0

    slices = [73, 117, 32]
    if vec == None:
        v3d = io.read_data(path)
    else:
        v3d = io.vectogeometry(vec)


    # levels
    levels = np.arange(0,3, 0.1)

    io.create_figure_surface(4, aspect, xx, yy, cmin, cmax, levels, slices, v3d)

def createReducedMatrices(U_POD,U_DEIM,PT,p,path):
    ''' creates 12 matrices for the reduced system with V as basis and saves them
    '''
    np.save(path + "U_POD_truncated.npy",U_POD)
    np.save(path + "U_DEIM_truncated.npy",U_DEIM)
    np.save(path + "PT.npy",PT)
    np.save(path + "p.npy",p)
    P = np.dot(U_DEIM,la.inv(np.dot(PT.T,U_DEIM)))

    for i in range(0,12):
        Ai = io.read_PETSc_mat('data/TMM/2.8/Transport/Matrix5_4/1dt/Ai_'+str(i).zfill(2)+'.petsc')
        Ae = io.read_PETSc_mat('data/TMM/2.8/Transport/Matrix5_4/1dt/Ae_'+str(i).zfill(2)+'.petsc')
        #Ai = sp.sparse.block_diag((Ai,Ai))
        #Ae = sp.sparse.block_diag((Ae,Ae))
        Ar = U_POD.T.dot(Ai.dot(Ae.dot(U_POD)))
        Pr = U_POD.T.dot(Ai.dot(P))
        np.save(path + 'reduced_A' +str(i).zfill(2), Ar)
        np.save(path + 'reduced_P' +str(i).zfill(2), Pr)

def linearinterpolation(nstep,ndata,dt):
    """used for lienar interpolation between transportation matrices.
    returns weights alpha, beta and indices of matrices.
    Parameters:
    nstep: Number of timesteps
    ndata: Number of matrices
    """

    t = np.zeros(nstep,dtype=np.float_)
    for i in range(nstep):
        t[i] = np.fmod(0 + i*dt, 1.0)


    beta    = np.array(nstep,dtype=np.float_)
    alpha   = np.array(nstep,dtype=np.float_)

    w       = t * ndata+0.5
    beta    = np.float_(np.fmod(w, 1.0))
    alpha   = np.float_(1.0-beta)
    jalpha  = np.fmod(np.floor(w)+ndata-1.0,ndata).astype(int)
    jbeta   = np.fmod(np.floor(w),ndata).astype(int)

    return alpha,beta,jalpha,jbeta

def deimInterpolationIndices(U_DEIM):
    #indices
    p = np.zeros(U_DEIM.shape[1],dtype=np.int)
    zero_vec = np.zeros(U_DEIM.shape[0])
    
    #full index matrix with zeros
    PT = np.zeros((U_DEIM.shape[0],U_DEIM.shape[1]))

    #first index
    p[0] = np.abs(U_DEIM[:,0]).argmax()
    #print("p: ",p[0], np.max(U_DEIM[:,0]),U_DEIM[np.abs(U_DEIM[:,0]).argmax(),0],U_DEIM[:,0].shape)
    zero_vec[p[0]]  = 1
    PT[:,0] = zero_vec

    for l in range(1,U_DEIM.shape[1]):
        #solve (P.T U)c = P.T u_l
        c = la.solve(np.dot(PT[:,:l].T,U_DEIM[:,:l]),np.dot(PT[:,:l].T,U_DEIM[:,l]))
        #r = u_l -Uc
        r = U_DEIM[:,l] - np.dot(U_DEIM[:,:l],c)
        #print(np.count_nonzero(r))
        #next index
        p[l] = np.abs(r).argmax()
        #print("p: ",p[l],l)
        zero_vec = np.zeros(U_DEIM.shape[0])
        zero_vec[p[l]]  = 1
        PT[:,l] = zero_vec
    return PT,p

def deimInterpolationIndicesTriag(U_DEIM):
    #indices
    p = np.zeros(U_DEIM.shape[1],dtype=np.int)
    zero_vec = np.zeros(U_DEIM.shape[0])
    
    #full index matrix with zeros
    PT = np.zeros((U_DEIM.shape[0],U_DEIM.shape[1]))
    U = np.zeros((U_DEIM.shape[0],U_DEIM.shape[1]))

    #first index
    p[0] = U_DEIM[:,0].argmax()
    zero_vec[p[0]]  = 1
    PT[:,0] = zero_vec
    U[:,0] = U_DEIM[:,0]

    for l in range(1,U_DEIM.shape[1]):
        #solve (P.T U)c = P.T u_l
        c = la.solve(np.dot(PT[:,:l].T,U_DEIM[:,:l]),np.dot(PT[:,:l].T,U_DEIM[:,l]))
        #r = u_l -Uc
        r = U_DEIM[:,l] - np.dot(U_DEIM[:,:l],c)
        #next index
        p[l] = r.argmax()
        zero_vec = np.zeros(U_DEIM.shape[0])
        zero_vec[p[l]]  = 1
        PT[:,l] = zero_vec
        U[:,l]  = r
    return PT,p

def generateIndicesForNonlinearFunction(lsm,profiles,ny):
    offset = 0
    J = [] 
    for ix in range(lsm.shape[0]):
        for iy in range(lsm.shape[1]):
            length = lsm[ix, iy]
            if not length == 0:
                #chage lenght of J to 52749
                #for i in range(length.astype(int)):
                #J.append(np.arange(offset,offset+length))
                J.append(offset)
                offset = offset + length
    J.append(52749)
    #convert to array
    J = np.array(J,dtype=(int))


    #
    PJ = np.zeros((ny,2),dtype=np.int_)
    for j in range(profiles +1):
        for i in range(J[j-1],J[j]):
            #print(i-J[j-1])
            PJ[i,0] = j-1;
            PJ[i,1] = i-J[j-1]
    return J,PJ