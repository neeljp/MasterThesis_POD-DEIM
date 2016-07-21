import numpy as np
import numpy.linalg as la
import petsc_io as io
import scipy as sp
import matplotlib as plt
import model
import util as util
#import NDOPmodel
np.set_printoptions(threshold=np.nan)



def constructSnapshotMatrix(path,ndistribution,nspinup,ntimestep,starttimestep = 1):
    ''' returns an matix of Snapshots,
    Params: ndistribution:number of diffrent initial distributions
            nspinup: number of spinups
            ntimestep: number of timesteps
            path: loding directory
            distribution: string code of distributions
    '''
    ny = 52749
    Y_N = np.empty([ny,np.int_(ndistribution * nspinup *(ntimestep-starttimestep+1))],dtype='float_')
    #Y_DOP = np.empty([ny,np.int_(ndistribution * nspinup *(ntimestep-starttimestep))],dtype='float_')

    counter = 0
    #start distribution
    #for j in range(startdistribution,ndistribution,2):
        #spinup
    for x in range(0,nspinup):
            #time step
        for i in range(starttimestep,ntimestep+1):

            Y_N[:,counter] = io.read_PETSc_vec(path+'/sp'+str(x).zfill(4)+'ts' + str(i*240-1).zfill(4) + 'N.petsc')
            #Y_DOP[:,counter] = io.read_PETSc_vec(path+'/sp'+str(x).zfill(4)+'ts' + str(i*240-1).zfill(4) + 'DOP.petsc')
            #print(path + '/sp' + str(x).zfill(4) +'ts' + str(i*240-1).zfill(4) + 'N.petsc')
            counter+=1

    print('Data loaded')
    return Y_N#,Y_DOP

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

def reducedModel_pod(initVec,reducedBasis,distribution,spinups):
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
            for m in range(12):
                #one month
                for i in range(240):
                    #one timestep
                    Aint = a[i]*A[j[i]] + b[i]*A[k[i]]
                    c = Aint.dot(c)

                #save solution vector at the end of one month
                vstep = V.dot(c)
                io.write_PETSc_vec(vstep, reducedBasis + '/sp'+str(l).zfill(4)+'ts'+str(i+m*240).zfill(4)+'N_'+distribution+'.petsc')


def reducedModelDeimNDOP(nspinup,ntimestep,PJ,J,p,U_pod,U_deim):
    N = np.ones(52749,dtype=np.float_) * 2.17
    DOP = np.ones(52749,dtype=np.float_) * 1e-4
    y = np.zeros((2*52749),dtype=np.float_)
    y[:52749] = N
    y[52749:] = DOP
    y_red = np.dot(U_pod.T,y) 
    y_long = U_pod.dot(y_red)
    offset = 52749

    print("projection error: ", (y_long[:offset]),(y_long[offset:]),np.nonzero(y_long < 0))
    #print(y_long[52749],y_long[-1],y_long.shape)

    fice = np.zeros((4448,12),dtype=np.float_)
    for i in range(12):
        fice[:,i] = io.read_PETSc_vec('data/TMM/2.8/Forcing/BoundaryCondition/fice_' + str(i).zfill(2)+ '.petsc')
        

    lat = io.read_PETSc_vec('data/TMM/2.8/Forcing/BoundaryCondition/latitude.petsc')
    dz = io.read_PETSc_vec('data/TMM/2.8/Forcing/DomainCondition/dz.petsc')
    z = io.read_PETSc_vec('data/TMM/2.8/Forcing/DomainCondition/z.petsc')
    lsm = io.read_PETSc_mat("data/TMM/2.8/Geometry/landSeaMask.petsc")

     #load monthly reduced matrices
    A = np.ndarray(shape=(12,U_pod.shape[1],U_pod.shape[1]), dtype=np.float_, order='C')
    for i in range(0,12):
        A[i] = np.load('reduced_A' +str(i).zfill(2)+'.npy')
        
    P = np.ndarray(shape=(12,U_pod.shape[1],U_deim.shape[1]), dtype=np.float_, order='C')
    for i in range(0,12):
        P[i] = np.load('reduced_P' +str(i).zfill(2)+'.npy')

    [a,b,j,k] = linearinterpolation(2880,12)
    bc = np.zeros(2,dtype=np.float_)
    dc = np.zeros((52749,2),dtype=np.float_)
    dc[:,0] = z
    dc[:,1] = dz

    #check if q is zero in fortran routine 
    q = np.zeros((52749,2),dtype=np.float64)
    u = np.array([0.02,2.0,0.5,30.0,0.67,0.5,0.858],dtype=np.float64)
    dt = 0.0003472222222222
    t = 0
    q_select = np.zeros(p.shape[0],dtype=np.float_)
    for spin in range(nspinup):
            for s in range(ntimestep):
                t = np.fmod(0 + s*dt, 1.0);
                counter = 0
                for i,l in PJ[p]:
                        index = np.array(np.vstack((np.arange(J[i],J[i+1]),np.arange(J[i]+offset,J[i+1]+offset))))
                        y = np.dot(U_pod[index,:],y_red).T
                        
                        
                        bc[0] = lat[i]
                        bc[1] = a[s]*fice[i,j[s]] + b[s]*fice[i,k[s]]
                        q_select[counter] = model.metos3dbgc(dt,t,y,u,bc,dc[J[i]:J[i+1],:])[l,np.int(offset >= p[counter])]
                        counter += 1
                #print("i: ", i,l)  
                #print("PJ: ", PJ[p])      
                Pint = a[s]*P[j[s]] + b[s]*P[k[s]]
                v_biostep = np.dot(Pint,q_select)
                
                Aint = a[s]*A[j[s]] + b[s]*A[k[s]]
                y_red = Aint.dot(y_red) + v_biostep
                #print("y", y_red,"\n","biostep",v_biostep,"\n")#,U.dot(y_red))
                
                
                #if(s % 240 == 239):
                y_long = U_pod.dot(y_red)
                v1 = io.read_PETSc_vec('simulation/POD/sp' + str(spin).zfill(4) + 'ts'+  str(s).zfill(4) + 'N.petsc')
                    #io.write_PETSc_vec(y[:,0], 'simulation/reduced/' + '/sp'+str(spin).zfill(4)+'ts'+str(s).zfill(4)+'N.petsc')
                v2 = io.read_PETSc_vec('simulation/POD/sp' + str(spin).zfill(4) + 'ts' + str(s).zfill(4)+ 'DOP.petsc')
                v_long = np.zeros_like(y_long)
                v_long[:offset] = v1
                v_long[offset:] = v2

                    #io.write_PETSc_vec(y[:,1], 'simulation/reduced/' + '/sp'+str(spin).zfill(4)+'ts'+str(s).zfill(4)+'DOP.petsc')
                print("spin: ", spin,"step: ", s,"t:", t,"norm: ", str(s).zfill(4) ,np.linalg.norm(y_long[:offset]-v1), np.linalg.norm(y_long[offset:]-v2),
                        np.linalg.norm(y_long[np.nonzero(y_long>0)]-v_long[np.nonzero(y_long>0)]))
                    #print(y_long[:52749]-v1,y_long[52749:]-v2)


def reducedModelDeimN(config,nspinup,ntimestep,PJ,J,p,U_pod,U_deim):
    options = util.parse_config(config)
    
    y = np.ones(52749,dtype=np.float_) * np.float_(options["-Metos3DTracerInitValue"])

    y_red = np.dot(U_pod.T,y) 
    y_long = U_pod.dot(y_red)

    print("projection error: ", y_long,np.nonzero(y_long < 0))
    #print(y_long[52749],y_long[-1],y_long.shape)

    fice = np.zeros((4448,12),dtype=np.float_)
    for i in range(12):
        fice[:,i] = io.read_PETSc_vec('data/TMM/2.8/Forcing/BoundaryCondition/fice_' + str(i).zfill(2)+ '.petsc')
        

    lat = io.read_PETSc_vec('data/TMM/2.8/Forcing/BoundaryCondition/latitude.petsc')
    dz = io.read_PETSc_vec('data/TMM/2.8/Forcing/DomainCondition/dz.petsc')
    z = io.read_PETSc_vec('data/TMM/2.8/Forcing/DomainCondition/z.petsc')
    lsm = io.read_PETSc_mat("data/TMM/2.8/Geometry/landSeaMask.petsc")

     #load monthly reduced matrices
    A = np.ndarray(shape=(12,U_pod.shape[1],U_pod.shape[1]), dtype=np.float_, order='C')
    for i in range(0,12):
        A[i] = np.load('reduced_A' +str(i).zfill(2)+'.npy')
        
    P = np.ndarray(shape=(12,U_pod.shape[1],U_deim.shape[1]), dtype=np.float_, order='C')
    for i in range(0,12):
        P[i] = np.load('reduced_P' +str(i).zfill(2)+'.npy')

    [a,b,j,k] = linearinterpolation(2880,12)
    bc = np.zeros(2,dtype=np.float_)
    dc = np.zeros((52749,2),dtype=np.float_)
    dc[:,0] = z
    dc[:,1] = dz

    #check if q is zero in fortran routine 
    q = np.zeros((52749),dtype=np.float64)
    u = np.array([0.02,2.0,0.5,30.0,0.858],dtype=np.float64)
    dt = 0.0003472222222222
    t = 0
    q_select = np.zeros(p.shape[0],dtype=np.float_)
    for spin in range(nspinup):
            for s in range(ntimestep):
                t = np.fmod(0 + s*dt, 1.0);
                counter = 0
                for i,l in PJ[p]:
                        index = np.arange(J[i],J[i+1])
                        #transpose for input in biomodel
                        y = np.dot(U_pod[index,:],y_red).T
                        
                        
                        bc[0] = lat[i]
                        bc[1] = a[s]*fice[i,j[s]] + b[s]*fice[i,k[s]]
                        #print("bio: ", y, bc ,dc[J[i]:J[i+1],:],u)
                        q_select[counter] = model.metos3dbgc(dt,t,y,u,bc,dc[J[i]:J[i+1],:])[l]
                        counter += 1
                #print("i: ", i,l)  
                #print("PJ: ", PJ[p])      
                Pint = a[s]*P[j[s]] + b[s]*P[k[s]]
                v_biostep = np.dot(Pint,q_select)
                
                Aint = a[s]*A[j[s]] + b[s]*A[k[s]]
                y_red = Aint.dot(y_red) + v_biostep
                #print("y", y_red,"\n","biostep",v_biostep,"\n")#,U.dot(y_red))
                
                
                if(s % 240 == 239):
                    y_long = U_pod.dot(y_red)
                    v1 = io.read_PETSc_vec('simulation/POD/N-Model/sp' + str(spin).zfill(4) + 'ts'+  str(s).zfill(4) + 'N.petsc')
                    io.write_PETSc_vec(y_long, 'simulation/reduced/N-Model/' + '/sp'+str(spin).zfill(4)+'ts'+str(s).zfill(4)+'N.petsc')
                    print("spin: ", spin,"step: ", s,"t:", t,"norm: ", str(s).zfill(4) ,np.linalg.norm(y_long-v1))
                    #print(y_long[:52749]-v1,y_long[52749:]-v2)


def reducedModelDeim_test(nspinup,ntimestep,J):
    N = np.ones(52749,dtype=np.float_) * 2.17
    DOP = np.ones(52749,dtype=np.float_) * 1e-4
    y = np.zeros((2*52749),dtype=np.float_)
    y[:52749] = N
    y[52749:] = DOP
    
    #print(y_long[52749],y_long[-1],y_long.shape)
    offset = 52749

    fice = np.zeros((4448,12),dtype=np.float_)
    for i in range(12):
        fice[:,i] = io.read_PETSc_vec('data/TMM/2.8/Forcing/BoundaryCondition/fice_' + str(i).zfill(2)+ '.petsc')
        

    lat = io.read_PETSc_vec('data/TMM/2.8/Forcing/BoundaryCondition/latitude.petsc')
    dz = io.read_PETSc_vec('data/TMM/2.8/Forcing/DomainCondition/dz.petsc')
    z = io.read_PETSc_vec('data/TMM/2.8/Forcing/DomainCondition/z.petsc')
    lsm = io.read_PETSc_mat("data/TMM/2.8/Geometry/landSeaMask.petsc")

     #load monthly reduced matrices
    Ae = []
    Ai = []
    for i in range(0,1):
        Ai00 = io.read_PETSc_mat('data/TMM/2.8/Transport/Matrix5_4/1dt/Ai_'+str(i).zfill(2)+'.petsc')
        Ae00 = io.read_PETSc_mat('data/TMM/2.8/Transport/Matrix5_4/1dt/Ae_'+str(i).zfill(2)+'.petsc')
        Ai01 = io.read_PETSc_mat('data/TMM/2.8/Transport/Matrix5_4/1dt/Ai_'+str(i+11).zfill(2)+'.petsc')
        Ae01 = io.read_PETSc_mat('data/TMM/2.8/Transport/Matrix5_4/1dt/Ae_'+str(i+11).zfill(2)+'.petsc')
    #Aea = np.asarray(Ae,dtype=np.float_)
    #Aia = np.asarray(Ai,dtype=np.float_)

    [a,b,j,k] = linearinterpolation(2880,12)
    bc = np.zeros(2,dtype=np.float_)
    dc = np.zeros((52749,2),dtype=np.float_)
    dc[:,0] = z
    dc[:,1] = dz

    #check if q is zero in fortran routine 
    q = np.zeros((52749,2),dtype=np.float64)
    q_com = np.zeros(2*52749,dtype=np.float_)
    u = np.array([0.02,2.0,0.5,30.0,0.67,0.5,0.858],dtype=np.float64)
    dt = 0.0003472222222222
    t = 0
    #q_select = np.zeros(p.shape[0],dtype=np.float_)
    for spin in range(nspinup):
            for s in range(ntimestep):
                t = np.fmod(0 + s*dt, 1.0);
                counter = 0
                for i in range(4448):
                   
                    bc[0] = lat[i]
                    bc[1] =a[s]*fice[i,j[s]] + b[s]*fice[i,k[s]]
                    
                    y_sep =  np.zeros((J[i+1] - J[i],2),dtype=np.float_)
                    y_sep[:,0] = y[J[i]:J[i+1]]
                    y_sep[:,1] = y[J[i]+offset:J[i+1]+offset]

                    q[J[i]:J[i+1],:] = model.metos3dbgc(dt,t,y_sep,u,bc,dc[J[i]:J[i+1],:])


                
                Aiint = b[s]*Ai00 + a[s]*Ai01
                Aeint = b[s]*Ae00 + a[s]*Ae01
                Aiint = sp.sparse.block_diag((Aiint,Aiint))
                Aeint = sp.sparse.block_diag((Aeint,Aeint))
                
                q_com[:offset] = q[:,0]
                q_com[offset:] = q[:,1]
                
                
                #print("y", y_red,"\n","biostep",v_biostep,"\n")#,U.dot(y_red))
                
                
                #if(s % 240 == 239):
                #y_long = U_pod.dot(y_red)
                v1 = io.read_PETSc_vec('simulation/POD/sp' + str(spin).zfill(4) + 'ts'+  str(s).zfill(4) + 'N.petsc')
                    #io.write_PETSc_vec(y[:,0], 'simulation/reduced/' + '/sp'+str(spin).zfill(4)+'ts'+str(s).zfill(4)+'N.petsc')
                v2 = io.read_PETSc_vec('simulation/POD/sp' + str(spin).zfill(4) + 'ts' + str(s).zfill(4)+ 'DOP.petsc')
                    #io.write_PETSc_vec(y[:,1], 'simulation/reduced/' + '/sp'+str(spin).zfill(4)+'ts'+str(s).zfill(4)+'DOP.petsc')
                print("spin: ", spin,"step: ", s,"t:", t,"norm: ", str(s).zfill(4) ,np.linalg.norm(y[:offset]-v1), np.linalg.norm(y[offset:]-v2), np.min(y[:offset]-v1),np.min(y[offset:]-v2))
                    #print(y_long[:52749]-v1,y_long[52749:]-v2)
                y = Aiint.dot((Aeint.dot(y) + q_com))

def plotErrorNorm(ndistribution,nspinup,nspinup_reduced,timesteps):
    ''' loads trajecotries from the highdim. and the reduced system, computes error norm and plots it.
        Params: distribution: string code of distribution
                reducedBasis: string code of the reduced basis
                ndistribution: number of distribution
                nspinup: number of spinups
                nspinup_reduced: number of spinups of the reduced data
                timesteps: number of timesteps
    '''
    Y_reduced = constructSnapshotMatrix('simulation/reduced/N-Model/',ndistribution,nspinup_reduced,12,timesteps)
    Y_hd = constructSnapshotMatrix('simulation/POD/N-Model/',ndistribution,nspinup,12,timesteps)
    n = np.ones(Y_reduced.shape[0]) * 2.17

    #plt.pyplot.title('Error norm of '+ distribution + str(ndistribution).zfill(2) + ' created with '+ reducedBasis )
    #plt.pyplot.title('Error norm of '+ distribution + str(ndistribution).zfill(2) + ' created with metos3d')
    plt.pyplot.title('Error norm of metos3d and implementation in python')
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = [r'\boldmath']
    plt.rc('font', family='serif', size=20, weight='bold')

    plt.pyplot.plot(np.linalg.norm(Y_hd - Y_reduced[:,:Y_hd.shape[1]],axis=0) ,color='darkgray',linestyle='--',linewidth=2)
    #plt.pyplot.plot(np.linalg.norm(Y_reduced[:,1:] - Y_reduced[:, 0:nspinup_reduced*(13-timesteps-1)-1] ,axis=0),color='black',linestyle=':',linewidth=2)
    #plt.pyplot.plot(np.linalg.norm(Y_reduced - n[:, np.newaxis] ,axis=0),color='slategray',linestyle='-.',linewidth=2)
    #plt.pyplot.plot(np.linalg.norm(Y_hd[:,1:] - Y_hd[:, 0:nspinup*(13-timesteps-1)-1] ,axis=0),color='dimgray',linestyle='-',linewidth=2)
    #plt.pyplot.plot(np.linalg.norm(Y_hd - n[:, np.newaxis] ,axis=0),color='black',linestyle='-',linewidth=2)

    #labels = [r"$\parallel y_{hd} -y_r \parallel_2 $",r"$\parallel y_{r} -y_{op} \parallel_2 $",r"$\parallel y_{rl} -y_{rl-1} \parallel_2 $",
            #r"$\parallel y_{hdl} -y_{hdl-1} \parallel_2 $",r"$\parallel y_{hd} -y_{op} \parallel_2 $"]
    #labels = [r"$\parallel y_{hdl} -y_{hdl-1} \parallel_2 $",r"$\parallel y_{hd} -y_{op} \parallel_2 $"]
    #labels = [r"$\parallel y_{rl} -y_{r-1} \parallel_2 $",r"$\parallel y_{r} -y_{op} \parallel_2 $"]

    #plt.pyplot.legend(labels,loc='best')
    plt.pyplot.yscale('log')
    #plt.pyplot.xscale('log')
    plt.pyplot.xlabel('Model Year')
    plt.pyplot.ylabel(r"\textbf{norm}  $\frac{mmol P}{m^3} $")
    plt.pyplot.show()


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

def createReducedMatrices(V,P):
    ''' creates 12 matrices for the reduced system with V as basis and saves them
    '''
    for i in range(0,12):
        Ai = io.read_PETSc_mat('data/TMM/2.8/Transport/Matrix5_4/1dt/Ai_'+str(i).zfill(2)+'.petsc')
        Ae = io.read_PETSc_mat('data/TMM/2.8/Transport/Matrix5_4/1dt/Ae_'+str(i).zfill(2)+'.petsc')
        #Ai = sp.sparse.block_diag((Ai,Ai))
        #Ae = sp.sparse.block_diag((Ae,Ae))
        Ar = V.T.dot(Ai.dot(Ae.dot(V)))
        Pr = V.T.dot(Ai.dot(P))
        np.save('reduced_A' +str(i).zfill(2), Ar)
        np.save('reduced_P' +str(i).zfill(2), Pr)

    #np.save('Basis_matrix',V)

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
    dt = 0.0003472222222222
    t = numpy.zeros(nstep,dtype=numpy.float_)
    for i in range(nstep):
        t[i] = numpy.fmod(0 + i*dt, 1.0)

    #tstep   = numpy.longdouble(1.0)/numpy.longdouble(nstep)
    #t       = numpy.linspace(0,1,nstep,dtype=numpy.longdouble,endpoint=False)
    beta    = numpy.array(nstep,dtype=numpy.float_)
    alpha   = numpy.array(nstep,dtype=numpy.float_)

    w       = t * ndata+0.5
    #print("%0.20f" % w[1])
    beta    = numpy.float_(numpy.fmod(w, 1.0))
    alpha   = numpy.float_(1.0-beta)
    jalpha  = numpy.fmod(numpy.floor(w)+ndata-1.0,ndata).astype(int)
    jbeta   = numpy.fmod(numpy.floor(w),ndata).astype(int)

    return alpha,beta,jalpha,jbeta

def deimInterpolationIndices(U_deim):
    #indices
    p = np.zeros(U_deim.shape[1],dtype=np.int)
    zero_vec = np.zeros(U_deim.shape[0])
    
    #full index matrix with zeros
    PT = np.zeros((U_deim.shape[0],U_deim.shape[1]))

    #first index
    p[0] = np.abs(U_deim[:,0]).argmax()
    #print("p: ",p[0], np.max(U_deim[:,0]),U_deim[np.abs(U_deim[:,0]).argmax(),0],U_deim[:,0].shape)
    zero_vec[p[0]]  = 1
    PT[:,0] = zero_vec

    for l in range(1,U_deim.shape[1]):
        #solve (P.T U)c = P.T u_l
        c = la.solve(np.dot(PT[:,:l].T,U_deim[:,:l]),np.dot(PT[:,:l].T,U_deim[:,l]))
        #r = u_l -Uc
        r = U_deim[:,l] - np.dot(U_deim[:,:l],c)
        #print(np.count_nonzero(r))
        #next index
        p[l] = np.abs(r).argmax()
        #print("p: ",p[l],l)
        zero_vec = np.zeros(U_deim.shape[0])
        zero_vec[p[l]]  = 1
        PT[:,l] = zero_vec
    return PT,p

def deimInterpolationIndicesTriag(U_deim):
    #indices
    p = np.zeros(U_deim.shape[1],dtype=np.int)
    zero_vec = np.zeros(U_deim.shape[0])
    
    #full index matrix with zeros
    PT = np.zeros((U_deim.shape[0],U_deim.shape[1]))
    U = np.zeros((U_deim.shape[0],U_deim.shape[1]))

    #first index
    p[0] = U_deim[:,0].argmax()
    zero_vec[p[0]]  = 1
    PT[:,0] = zero_vec
    U[:,0] = U_deim[:,0]

    for l in range(1,U_deim.shape[1]):
        #solve (P.T U)c = P.T u_l
        c = la.solve(np.dot(PT[:,:l].T,U_deim[:,:l]),np.dot(PT[:,:l].T,U_deim[:,l]))
        #r = u_l -Uc
        r = U_deim[:,l] - np.dot(U_deim[:,:l],c)
        #next index
        p[l] = r.argmax()
        zero_vec = np.zeros(U_deim.shape[0])
        zero_vec[p[l]]  = 1
        PT[:,l] = zero_vec
        U[:,l]  = r
    return PT,p

def generateIndicesForNonlinearFunction():
    lsm = io.read_PETSc_mat("data/TMM/2.8/Geometry/landSeaMask.petsc")
    offset = 0
    J = [] 
    for ix in range(64):
        for iy in range(128):
            length = lsm[ix, iy]
            if not length == 0:
                #chage lenght of J to 52749
                #for i in range(length.astype(int)):
                #J.append(np.arange(offset,offset+length))
                J.append(offset)
                offset = offset + length
    J.append(52749)
    J = np.array(J,dtype=(int))



    PJ = np.zeros((2*52749,2),dtype=np.int_)
    for j in range(4449):
        for i in range(J[j-1],J[j]):
            #print(i-J[j-1])
            PJ[i,0] = j-1;
            PJ[i,1] = i-J[j-1]
            PJ[i+52749,0] = j-1;
            PJ[i+52749,1] = i-J[j-1]
    return J,PJ