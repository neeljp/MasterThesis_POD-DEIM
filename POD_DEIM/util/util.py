import numpy as np
import numpy.linalg as la
from util import petsc_io as io
import scipy as sp
import matplotlib as plt
import pyDOE


def constructMatrix(path,outpath,pattern,nstep):
    ''' returns an matix constructed out of the vectos in the given directory than macht the pattern
   Parameters
    -------
    path: str 
            path to directory
    outpath: str 
                path ot output directory
    pattern: str
                pattern to match files
    nstep: int
                number of vectors 
            '''
    ny = 52749
    Y  = np.empty([ny,np.int_(nstep)],dtype='float_')

    counter = 0
    for i in range(nstep):
        Y[:,i] = io.read_PETSc_vec(path+pattern % i)
    
    np.save(outpath,Y)

def computeSVD(Y):
    """computes trucated svd of a matrix
    Parameters
    -------
    Y  : 2-D array
    Returns
    -------
    U : 2-D array
        left singular vectors
    s : array
            singular values
    V : 2-D array
            right singular values
    """
    U, s, V = la.svd(Y,full_matrices=False)
    return U, s, V

def computePODError(s,eps):
    """computes POD error out of the singular values
    Parameters
    -------
    s  : array
            singular values
    eps : float
            desired error
    Returns
    -------
    index : int
                size of the pod base 
    """
    sigmaSum_r = 0
    sigmaSum = sum(s)
    for i,j in enumerate(s):
        sigmaSum_r += j
        if  sigmaSum_r/sigmaSum >= 1- eps*eps:
            index = i
            break
    return index

def plotSingular(s,qs,eps,bounds=True):
    ''' computes the index dependent on eps
        and plot the singular values
    Parameters
    -------
    s  : array
            singular values
    eps : float
            desired error
    bounds : bool
                true if error bounds should be added to the plot
    Returns
    -------
    plot of singular values
    '''
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = [r'\boldmath']
    plt.rc('font', family='Bitstream Vera Sans', size=22, weight='bold')
    plt.pyplot.plot(s,'--',color='black')#,marker='D',markersize=5)   
    plt.pyplot.plot(qs,'-',color='gray')#,marker='o',markersize=5)
    plt.pyplot.yscale('log')
    #plt.pyplot.xscale('log')
    if(bounds):
        for i in range(2,6):
            index = computePODError(s,np.power(10.0,-i*2))
            #qindex = computePODError(qs,np.power(10.0,-i*2))
            plt.pyplot.axvline(ls= ':',x=index,color='dimgray')
            #plt.pyplot.axvline(ls= ':',x=qindex,color='g')
            plt.pyplot.text(index, np.power(30,i), r"$\epsilon=$"+str(np.power(10.0,-i*2)))
            #plt.pyplot.text(qindex, np.power(10,i), 'qeps='+str(np.power(10.0,-i*2)))


def generate_prameter_samples(n,p):
        ''' generates Latin-Hypercubesamples with pyDOE and
        transforms bounds.
         Parameters
        -------
        n: int 
            number of samples
        p: 5,7 
            number of parameters
        Returns
        -------
        lh:  2D array
                 of parameter samples 
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
    ''' parses metos3d config and returns options
    Parameters
    -------
    filename: str 
                path to config 
    Returns
    -------
    options :  list
                 of options 
    '''
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
    ''' plots a the surface of a tracer on a worldmap 
    Parameters
    -------
    path: str 
            paht to vector
    vec : array
            tracer vector 
    Returns
    -------
    plot
    '''
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
    if vec is None:
        v3d = io.read_data(path)
    else:
        v3d = io.vectogeometry(vec)


    # levels
    levels = np.arange(0,2,0.1)

    create_figure_surface(4, aspect, xx, yy, cmin, cmax, levels, slices, v3d)

def createReducedMatrices(U_POD,U_DEIM,PT,p,path):
    ''' creates 12 matrices for the reduced system with V as basis and saves them
    Parameters
    -------
    U_POD    : 2-D array
                pod base vectors 
    U_DEIM   : 2-D array
                deim base vectors
    PT       : 2-D array
                DEIM index matrix
    p        : array
                DEIM indices 
    path     : str 
                path to directory to save the base matrices
    Returns
    -------
    saves the base matrices in the path directory
    '''
    np.save(path + "U_POD_truncated.npy",U_POD)
    np.save(path + "U_DEIM_truncated.npy",U_DEIM)
    np.save(path + "PT.npy",PT)
    np.save(path + "p.npy",p)
    P = np.dot(U_DEIM,la.inv(np.dot(PT.T,U_DEIM)))

    for i in range(0,12):
        Ai = io.read_PETSc_mat('data/TMM/2.8/Transport/Matrix5_4/1dt/Ai_'+str(i).zfill(2)+'.petsc')
        Ae = io.read_PETSc_mat('data/TMM/2.8/Transport/Matrix5_4/1dt/Ae_'+str(i).zfill(2)+'.petsc')
        #Ai = sp.sparse.block_diag((Ai,Ai))y
        #Ae = sp.sparse.block_diag((Ae,Ae))
        Ar = U_POD.T.dot(Ai.dot(Ae.dot(U_POD)))
        Pr = U_POD.T.dot(Ai.dot(P))
        np.save(path + 'reduced_A' +str(i).zfill(2), Ar)
        np.save(path + 'reduced_P' +str(i).zfill(2), Pr)




def linearinterpolation(nstep,ndata,dt):
    """used for lienar interpolation between transportation matrices.
    returns weights alpha, beta and indices of matrices.
    Parameters
    -------
    nstep   : int Number of timesteps
    ndata   : int Number of matrices
    Returns
    -------
    alpha,beta : array 
                    coefficients for interpolation
    jalpha,jbeta : array
                    indices for interpolation
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
    '''computes an DIEM indices
    Parameters
    -------
    U_DEIM    : 2-D array
                    POD Base vectors of the nonlinear term 

    Returns
    -------
    PT      : 2-D array
                  matrix with unit vectors of the deim indices 
    P       : array
              
    '''
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
    '''generates an DIEM indices with an diffrent algorithm that uses tridiagonal matrices
        NOT TESTED
    Parameters
    ----------
    U_DEIM    : 2-D array
                    POD Base vectors of the nonlinear term 

    Returns
    -------
    PT      : 2-D array
                  matrix with unit vectors of the deim indices 
    P       : array
                of deim indices
    '''
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
    '''generates an index vector for the nonlinear function to mark each profile
    Parameters
    ----------
    lsm      : 2-D array
                land sea mask
    profiles : int
                number of profiles
    ny       : int 
                length of tracer vector

    Returns
    -------
    J        : array
                 of start an end indices for each profile
    PJ       : array
                of indices of profiles for each vector entry
    '''
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


def create_figure_surface(figid, aspect, xx, yy, cmin, cmax, levels, slices, v3d):
    ''' creates a plot of the surface of a tracer data
    Parameters
    ----------
    figid : int  
            id of figure
    aspect: float
            aspect ratio of figure
    xx    : array 
            scale of x axis
    yy    : array 
            scale of y axis     
    cmin,cmax : array
                minimum and maxminm of the color range
    levels : array
            range of contourlines
    slices : array
            location of slices
    v3d    : vector data in geometry format
    Returns
    -------
    plot :  of surface data
    '''
    # prepare matplotlib
    import matplotlib
    matplotlib.rc("font",**{"family":"sans-serif"})
    matplotlib.rc("text", usetex=True)
    #matplotlib.use("PDF")
    import matplotlib.pyplot as plt
    # basemap
    from mpl_toolkits.basemap import Basemap
    # numpy
    import numpy as np

    # data
    vv = v3d[0,:,:,0]
    # shift
    vv = np.roll(vv, 64, axis=1)

    # plot surface
    plt.figure(figid)
    # colormap
    cmap = plt.cm.bone_r
    # contour fill
    p1 = plt.contourf(xx, yy, vv, cmap=cmap, levels=levels, origin="lower")#, hold="on")
    plt.clim(cmin, cmax)
    # contour lines
    p2 = plt.contour(xx, yy, vv, levels=levels, linewidths = (1,), colors="k")#, hold="on")
    plt.clabel(p2, fmt = "%2.1f", colors = "k", fontsize = 14)
    #plt.colorbar(p2,shrink=0.8, extend='both')
    # slices
    #s1 = xx[np.mod(slices[0]+64, 128)]
    #s2 = xx[np.mod(slices[1]+64, 128)]
    #s3 = xx[np.mod(slices[2]+64, 128)]
#    print s1, s2, s3
    #plt.vlines([s1, s2, s3], -90, 90, color='k', linestyles='--')
    # set aspect ratio of axes
    plt.gca().set_aspect(aspect)

    # basemap
    m = Basemap(projection="cyl")
    m.drawcoastlines(linewidth = 0.5)

    # xticks
    plt.xticks(range(-180, 181, 45), range(-180, 181, 45))
    plt.xlim([-180, 180])
    plt.xlabel("Longitude [degrees]", labelpad=8)
    # yticks
    plt.yticks(range(-90, 91, 30), range(-90, 91, 30))
    plt.ylim([-90, 90])
    plt.ylabel("Latitude [degrees]")


    # write to file
    plt.savefig("solution-surface", bbox_inches="tight")
    plt.show()

