import numpy as np
import numpy.linalg as la
import petsc_io as io
import scipy as sp
import matplotlib as plt
<<<<<<< HEAD
import model
import util as util
#import NDOPmodel
np.set_printoptions(threshold=np.nan)



def constructSnapshotMatrix(path,ndistribution,nspinup,ntimestep,starttimestep = 1):
=======
import pynolh
import argparse
import math
import sys


def constructSnapshotMatrix(path,distribution,startdistribution,ndistribution,nspinup,ntimestep,starttimestep = 0):
>>>>>>> 8e942201d16d7b74c2228a0549727df827f2d601
    ''' returns an matix of Snapshots,
    Params: ndistribution:number of diffrent initial distributions
            nspinup: number of spinups
            ntimestep: number of timesteps
            path: loding directory
            distribution: string code of distributions
    '''
    ny = 52749
<<<<<<< HEAD
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
=======
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
>>>>>>> 8e942201d16d7b74c2228a0549727df827f2d601

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
<<<<<<< HEAD
    plt.pyplot.show()
    return index

def reducedModel_pod(initVec,reducedBasis,distribution,spinups):
=======
    return index

def reducedModel(initVec,reducedBasis,distribution,spinups):
>>>>>>> 8e942201d16d7b74c2228a0549727df827f2d601
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
<<<<<<< HEAD
            for m in range(12):
=======
            for m in range(1,13):
>>>>>>> 8e942201d16d7b74c2228a0549727df827f2d601
                #one month
                for i in range(240):
                    #one timestep
                    Aint = a[i]*A[j[i]] + b[i]*A[k[i]]
                    c = Aint.dot(c)

                #save solution vector at the end of one month
                vstep = V.dot(c)
<<<<<<< HEAD
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
=======
                io.write_PETSc_vec(vstep, reducedBasis + '/sp'+str(l).zfill(4)+'ts'+str(i+(m-1)*240).zfill(4)+'N_'+distribution+'.petsc')


def plotErrorNorm(distribution,reducedBasis,ndistribution,nspinup,nspinup_reduced,timesteps):
>>>>>>> 8e942201d16d7b74c2228a0549727df827f2d601
    ''' loads trajecotries from the highdim. and the reduced system, computes error norm and plots it.
        Params: distribution: string code of distribution
                reducedBasis: string code of the reduced basis
                ndistribution: number of distribution
                nspinup: number of spinups
                nspinup_reduced: number of spinups of the reduced data
                timesteps: number of timesteps
    '''
<<<<<<< HEAD
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
=======
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
>>>>>>> 8e942201d16d7b74c2228a0549727df827f2d601


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

<<<<<<< HEAD
def createReducedMatrices(V,P):
=======
def createReducedMatrices(V):
>>>>>>> 8e942201d16d7b74c2228a0549727df827f2d601
    ''' creates 12 matrices for the reduced system with V as basis and saves them
    '''
    for i in range(0,12):
        Ai = io.read_PETSc_mat('data/TMM/2.8/Transport/Matrix5_4/1dt/Ai_'+str(i).zfill(2)+'.petsc')
        Ae = io.read_PETSc_mat('data/TMM/2.8/Transport/Matrix5_4/1dt/Ae_'+str(i).zfill(2)+'.petsc')
<<<<<<< HEAD
        #Ai = sp.sparse.block_diag((Ai,Ai))
        #Ae = sp.sparse.block_diag((Ae,Ae))
        Ar = V.T.dot(Ai.dot(Ae.dot(V)))
        Pr = V.T.dot(Ai.dot(P))
        np.save('reduced_A' +str(i).zfill(2), Ar)
        np.save('reduced_P' +str(i).zfill(2), Pr)

    #np.save('Basis_matrix',V)
=======
        Ar = V.T.dot(Ai.dot(Ae.dot(V)))
        np.save('reduced_A' +str(i).zfill(2), Ar)

    np.save('Basis_matrix',V)
>>>>>>> 8e942201d16d7b74c2228a0549727df827f2d601

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
<<<<<<< HEAD
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
=======
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
>>>>>>> 8e942201d16d7b74c2228a0549727df827f2d601
