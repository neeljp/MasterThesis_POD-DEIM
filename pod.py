import numpy as np
import numpy.linalg as la
import petsc_io as io
import scipy as sp
import matplotlib as plt



def constructSnapshotMatrix(path,distribution,startdistribution,ndistribution,nspinup,ntimestep,starttimestep = 0):
    ''' returns an matix of Snapshots,
    Params: ndistribution:number of diffrent initial distributions
            nspinup: number of spinups
            ntimestep: number of timesteps
            path: loding directory
            distribution: string code of distributions
    '''
    ny = 52749
    Y = np.empty([ny,((ndistribution-startdistribution+1)/2) * nspinup *(ntimestep-1-starttimestep)],dtype='float_')

    counter = 0
    #start distribution
    for j in range(startdistribution,ndistribution,2):
        #spinup
        for x in range(0,nspinup):
            #time step
             for i in range(starttimestep+1,ntimestep):

                 Y[:,counter] = io.read_PETSc_vec(path+'/sp'+str(x).zfill(4)+'ts' + str(i*240-1).zfill(4) + 'N_'+str(j).zfill(2)+'_'+distribution+'.petsc')
                 #print(path + '/sp' + str(x).zfill(4) +'ts' + str(i*240-1).zfill(4) + 'N_' + str(j).zfill(2)+ '_'+pattern+ '.petsc')
                 counter+=1

    print('Data loaded:'+path+'/sp000$dts000$dN_'+str(startdistribution).zfill(2)+'-'+str(ndistribution).zfill(2)+'_'+distribution+'.petsc')
    return Y

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
    return index

def reducedModel(initVec,reducedBasis,distribution,spinups):
        ''' computes solution of the redued modell
                Params: initVec: path to start vector
                        reducedBasis: path to the reduced basis
                        distribution: string code of distribution
                        spinup: number of spinups
        '''
        #start vector
        v0 = io.read_PETSc_vec(initVec)
        #reduced Basis_matrix
        V = np.load(reducedBasis + '/Basis_matrix.npy')
        #reduced start vector
        c = V.T.dot(v0)

        index = V.shape[1]
        #interpolation indices
        [a,b,j,k] = linearinterpolation(2880,12)
        #load monthly reduced matrices
        A = np.ndarray(shape=(12,index,index), dtype=float, order='C')
        for i in range(0,12):
            A[i] = np.load( reducedBasis + '/reduced_A' +str(i).zfill(2)+'.npy')

        #best solution
        n = np.ones(v0.shape[0])
        n = n*2.17


        #spinups
        for l in range(spinups):
            e = np.linalg.norm(n-V.dot(c))
            print('spingup:', l, 'error:',e)
            #one year
            for m in range(1,13):
                #one month
                for i in range(240):
                    #one timestep
                    Aint = a[i]*A[j[i]] + b[i]*A[k[i]]
                    c = Aint.dot(c)

                #save solution vector at the end of one month
                vstep = V.dot(c)
                io.write_PETSc_vec(vstep, reducedBasis + '/sp'+str(l).zfill(4)+'ts'+str(i+(m-1)*240).zfill(4)+'N_'+distribution+'.petsc')


def plotErrorNorm(distribution,reducedBasis,ndistribution,nspinup,nspinup_reduced,timesteps):
    ''' loads trajecotries from the highdim. and the reduced system, computes error norm and plots it.
        Params: distribution: string code of distribution
                reducedBasis: string code of the reduced basis
                ndistribution: number of distribution
                nspinup: number of spinups
                nspinup_reduced: number of spinups of the reduced data
                timesteps: number of timesteps
    '''
    Y_reduced = constructSnapshotMatrix('work/Reduced/'+reducedBasis,distribution,ndistribution,ndistribution+1,nspinup_reduced,13,timesteps)
    Y_hd = constructSnapshotMatrix('work/UD',distribution,ndistribution,ndistribution+1,nspinup,13,timesteps)
    n = np.ones(Y_reduced.shape[0]) * 2.17

    plt.pyplot.title('Error norm of '+ distribution + str(ndistribution).zfill(2) + ' created with '+ reducedBasis )
    #plt.pyplot.title('Error norm of '+ distribution + str(ndistribution).zfill(2) + ' created with metos3d')
    plt.rc('text', usetex=True)


    plt.pyplot.plot(np.linalg.norm(Y_hd - Y_reduced[:,:Y_hd.shape[1]],axis=0) ,'b--')
    plt.pyplot.plot(np.linalg.norm(Y_reduced[:,1:] - Y_reduced[:, 0:nspinup_reduced*(13-timesteps-1)-1] ,axis=0),'r:')
    plt.pyplot.plot(np.linalg.norm(Y_reduced - n[:, np.newaxis] ,axis=0),'g-.')
    plt.pyplot.plot(np.linalg.norm(Y_hd[:,1:] - Y_hd[:, 0:nspinup*(13-timesteps-1)-1] ,axis=0),'c-')
    plt.pyplot.plot(np.linalg.norm(Y_hd - n[:, np.newaxis] ,axis=0))

    labels = [r"$\parallel y_{hd} -y_r \parallel_2 $",r"$\parallel y_{r} -y_{op} \parallel_2 $",r"$\parallel y_{rl} -y_{rl-1} \parallel_2 $",
            r"$\parallel y_{hdl} -y_{hdl-1} \parallel_2 $",r"$\parallel y_{hd} -y_{op} \parallel_2 $"]
    #labels = [r"$\parallel y_{hdl} -y_{hdl-1} \parallel_2 $",r"$\parallel y_{hd} -y_{op} \parallel_2 $"]
    #labels = [r"$\parallel y_{rl} -y_{r-1} \parallel_2 $",r"$\parallel y_{r} -y_{op} \parallel_2 $"]

    plt.pyplot.legend(labels,loc='best')
    plt.pyplot.yscale('log')
    #plt.pyplot.xscale('log')
    plt.pyplot.xlabel('Model Month')
    plt.pyplot.ylabel(r"\textbf{norm}  $\frac{mmol P}{m^3} $")


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
    levels = np.arange(0,3, 0.01)

    io.create_figure_surface(4, aspect, xx, yy, cmin, cmax, levels, slices, v3d)

def createReducedMatrices(V):
    ''' creates 12 matrices for the reduced system with V as basis and saves them
    '''
    for i in range(0,12):
        Ai = io.read_PETSc_mat('data/TMM/2.8/Transport/Matrix5_4/1dt/Ai_'+str(i).zfill(2)+'.petsc')
        Ae = io.read_PETSc_mat('data/TMM/2.8/Transport/Matrix5_4/1dt/Ae_'+str(i).zfill(2)+'.petsc')
        Ar = V.T.dot(Ai.dot(Ae.dot(V)))
        np.save('reduced_A' +str(i).zfill(2), Ar)

    np.save('Basis_matrix',V)

def createReducedMatricesExp(V):
    ''' TEST creates 12 explicid matrices for the reduced system with V as basis
    '''
    for i in range(0,12):
        Ae = io.read_PETSc_mat('data/TMM/2.8/Transport/Matrix5_4/1dt/Ae_'+str(i).zfill(2)+'.petsc')
        Ar = Ae.dot(V)
        #print(Ar)
        Ar =  V.T.dot(Ar)
        #print(Ar)
        np.save('reduced_Ae' +str(i).zfill(2), Ar)

def linearinterpolation(nstep,ndata):
    """used for lienar interpolation between transportation matrices.
    returns weights alpha, beta and indices of matrices.
    Parameters:
    nstep: Number of timesteps
    ndata: Number of matrices
    """
    import numpy
    tstep   = 1/nstep
    t       = numpy.linspace(0,1,nstep)

    w       = t * ndata+0.5
    beta    = numpy.mod(w, 1)
    alpha   = 1-beta
    jalpha  = numpy.mod(numpy.floor(w)+ndata-1,ndata).astype(int)
    jbeta   = numpy.mod(numpy.floor(w),ndata).astype(int)

    return alpha,beta,jalpha,jbeta
    
def nolh(conf, remove=None):
    """Constructs a Nearly Orthogonal Latin Hypercube (NOLH) of order *m* from
    a configuration vector *conf*. The configuration vector may contain either
    the numbers in $[0 q-1]$ or $[1 q]$ where $q = 2^{m-1}$. The columns to be
    *removed* are also in $[0 d-1]$ or $[1 d]$ where $d = m + \binom{m-1}{2}$
    is the NOLH dimensionality.
    """
    I = numpy.identity(2, dtype=int)
    R = numpy.array(((0, 1),
                     (1, 0)), dtype=int)

    if 0 in conf:
        conf = numpy.array(conf) + 1

        if remove is not None:
            remove = numpy.array(remove) + 1


    q = len(conf)
    m = math.log(q, 2) + 1
    s = m + (math.factorial(m - 1) / (2 * math.factorial(m - 3)))
    # Factorial checks if m is an integer
    m = int(m)
    #print(q,m)
    A = numpy.zeros((q, q, m - 1), dtype=int)
    for i in range(1, m):
        Ai = 1
        for j in range(1, m):
            if j < m - i:
                Ai = numpy.kron(Ai, I)
            else:
                Ai = numpy.kron(Ai, R)

        A[:, :, i-1] = Ai

    M = numpy.zeros((q, s), dtype=int)
    M[:, 0] = conf

    col = 1
    for i in range(0, m - 1):
        for j in range(i + 1, m):
            if i == 0:
                M[:, col] = numpy.dot(A[:, :, j-1], conf)
            else:
                M[:, col] = numpy.dot(A[:, :, i-1], numpy.dot(A[:, :, j-1], conf))
            col += 1

    S = numpy.ones((q, s), dtype=int)

    v = 1
    for i in range(1, m):
        for j in range(0, q):
            if j % 2**(i-1) == 0:
                v *= -1
            S[j, i] = v

    col = m
    for i in range(1, m - 1):
        for j in range(i + 1, m):
            S[:, col] = S[:, i] * S[:, j]
            col += 1

    T = M * S
    #print(s,S.shape,M.shape,T.shape)

    keep = numpy.ones(s, dtype=bool)
    if remove is not None:
        keep[numpy.array(remove) - 1] = [False] * len(remove)

    concat =numpy.concatenate((T, numpy.zeros((1, s)), -T),axis=0)
    #print(T,-T,concat,concat.shape)

    return (numpy.concatenate((T, numpy.zeros((1, s)), -T), axis=0)[:, keep] + 8) / (2.0 * q)


def bounderyIntervall(nolh):
    for i in range(nolh.shape[0]):
        nolh[i,0] = nolh[i,0] * (0.75-0.25) +0.25
        nolh[i,1] = nolh[i,1] * (200-1.5) +1.5
        nolh[i,2] = nolh[i,2] * (0.95-0.05) +0.05
        nolh[i,3] = nolh[i,3] * (1.5-0.25) +0.25
        nolh[i,4] = nolh[i,4] * (50-10) +10
        nolh[i,5] = nolh[i,5] * (0.05-0.01) +0.01
        nolh[i,6] = nolh[i,6] * (1.5-0.7) +0.7
    return nolh

def lhs(dist, parms, siz=100, noCorrRestr=False, corrmat=None):
    '''
    Latin Hypercube sampling of any distribution.
    dist is is a scipy.stats random number generator
    such as stats.norm, stats.beta, etc
    parms is a tuple with the parameters needed for
    the specified distribution.

    :Parameters:
        - `dist`: random number generator from scipy.stats module or a list of them.
        - `parms`: tuple of parameters as required for dist, or a list of them.
        - `siz` :number or shape tuple for the output sample
        - `noCorrRestr`: if true, does not enforce correlation structure on the sample.
        - `corrmat`: Correlation matrix
    '''
    if not isinstance(dist,(list,tuple)):
        dists = [dist]
        parms = [parms]
    else:
        assert len(dist) == len(parms)
        dists = dist
    indices=rank_restr(nvars=len(dists), smp=siz, noCorrRestr=noCorrRestr, Corrmat=corrmat)
    smplist = []
    for j,d in enumerate(dists):
        if not isinstance(d, (stats.rv_discrete,stats.rv_continuous)):
            raise TypeError('dist is not a scipy.stats distribution object')
        n=siz
        if isinstance(siz,(tuple,list)):
            n=numpy.product(siz)
        #force type to float for sage compatibility
        pars = tuple([float(k) for k in parms[j]])
        #perc = numpy.arange(1.,n+1)/(n+1)
        step = 1./(n)
        perc = numpy.arange(0, 1, step) #class boundaries
        s_pos = [uniform(i, i+ step) for i in perc[:]]#[i+ step/2. for i in perc[:]]
        v = d(*pars).ppf(s_pos)
        #print len(v), step, perc
        index=map(int,indices[j]-1)
        print(index)
        v = v[list(index)]
        if isinstance(siz,(tuple,list)):
            v.shape = siz
        smplist.append(v)
    if len(dists) == 1:
        return smplist[0]
    return smplist

def rank_restr(nvars=4, smp=100, noCorrRestr=False, Corrmat=None):
    """
    Returns the indices for sampling variables with
    the desired correlation structure.

    :Parameters:
        - `nvars`: number of variables
        - `smp`: number of samples
        - `noCorrRestr`: No correlation restriction if True
        - `Corrmat`: Correlation matrix. If None, assure uncorrelated samples.
    """
    if isinstance(smp,(tuple,list)):
            smp=numpy.product(smp)
    def shuf(s):
        s1=[]
        for i in range(nvars):
            numpy.random.shuffle(s)
            s1.append(s.copy())
        return s1
    if noCorrRestr or nvars ==1:
        x = [stats.randint.rvs(0,smp+0,size=smp) for i in range(nvars)]
    else:
        if Corrmat == None:
            C=numpy.core.numeric.identity(nvars)
        else:
            if Corrmat.shape[0] != nvars:
                raise TypeError('Correlation matrix must be of rank %s'%nvars)
            C=numpy.matrix(Corrmat)
        s0=numpy.arange(1.,smp+1)/(smp+1.)
        s=stats.norm().ppf(s0)
        s1 = shuf(s)
        S=numpy.matrix(s1)
        P=cholesky(C)
        Q=cholesky(numpy.corrcoef(S))

        Final=S.transpose()*inv(Q).transpose()*P.transpose()
        x = [stats.stats.rankdata(Final.transpose()[i,]) for i in range(nvars)]
    return x
