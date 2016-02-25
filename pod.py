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
