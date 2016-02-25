import pynolh
import argparse
import math
import sys


def read_PETSc_vec(file):
    import numpy
    dsource = numpy.DataSource()
    # open file
    # omit header
    # read length
    # read values
    # close file

    try:
        f = open(file, "rb")
    except:
            print("Unexpected error:", sys.exc_info()[0],file)

    numpy.fromfile(f, dtype=">i4", count=1)
    nvec = numpy.fromfile(f, dtype=">i4", count=1)
    #load data and change it to little endian, importend for np.dot
    v = numpy.fromfile(f, dtype=">f8", count=nvec[0]).astype('<f8')
    f.close()

    return v

def write_PETSc_vec(v,file):
    import numpy

    f = open(file, "wb")
    header      = numpy.array([1211214])
    nx          = numpy.array(v.shape[0])
    header.astype('>i4').tofile(f)
    nx.astype('>i4').tofile(f)
    v.astype('>f8').tofile(f)
    f.close()

    return 0


def write_PETSc_mat(A,file):
    import struct
    import numpy
    from scipy.sparse import csc_matrix

    header      = numpy.array([1211216])
    dims        = A.shape
    nx          = numpy.array(dims[0])
    ny          = numpy.array(dims[1])
    nnz         = numpy.array([A.nnz])
    j,colidx    = A.nonzero()
    nrow,k      = numpy.histogram(j,range(0,dims[0]+1))

    # print('header')
    # print(header)
    # print("dims")
    # print(dims)
    # print("nnz")
    # print (nnz)
    # print ("nrow")
    # print (nrow,nrow.shape)
    # print ("colidx")
    # print (colidx,colidx.shape)
    # print('val')
    # print(A.data)
    f = open(file, "wb")
    header.astype('>i4').tofile(f)
    nx.astype('>i4').tofile(f)
    ny.astype('>i4').tofile(f)
    nnz.astype('>i4').tofile(f)
    nrow.astype('>i4').tofile(f)
    colidx.astype('>i4').tofile(f)
    A.data.astype('>f8').tofile(f)
    f.close()
    return 0


def read_PETSc_mat(file):
    import numpy
    from scipy.sparse import csc_matrix

    # open file
    try:
        f = open(file, "rb")
    except:
        print("Unexpected error:", sys.exc_info()[0],file)
    # omit header
    numpy.fromfile(f, dtype=">i4", count=1)
    # read dims
    nx     = numpy.fromfile(f, dtype=">i4", count=1)
    ny     = numpy.fromfile(f, dtype=">i4", count=1)
    nnz    = numpy.fromfile(f, dtype=">i4", count=1)
    nrow   = numpy.fromfile(f, dtype=">i4", count=nx[0])
    colidx = numpy.fromfile(f, dtype=">i4", count=nnz[0])
    val    = numpy.fromfile(f, dtype=">f8", count=nnz[0])

    # print("dims")
    # print( nx, ny)
    # print("nnz")
    # print (nnz)
    # print ("nrow")
    # print (nrow,nrow.shape)
    # print ("colidx")
    # print (colidx,colidx.shape)
    # print ("val")
    # print (val)

    # close file
    f.close()
    # create sparse matrix
    spdata = numpy.zeros((nnz[0],3))
    offset = 0
    for i in range(nx[0]):
        if not nrow[i] == 0.0:
            spdata[offset:offset+nrow[i],0]= i
            spdata[offset:offset+nrow[i],1]= colidx[offset:offset+nrow[i]]
            spdata[offset:offset+nrow[i],2]= val[offset:offset+nrow[i]]
            offset = offset+nrow[i]
    #print(spdata[:,0])
    return csc_matrix((spdata[:,2], (spdata[:,0],spdata[:,1])), shape=(nx, ny))

    # create full matrix
    # lsmfull = numpy.zeros(shape=(nx[0], ny[0]), dtype=int)
    # offset = 0
    # for i in range(nx[0]):
    #     if not nrow[i] == 0.0:
    #         for j in range(nrow[i]):
    #             lsmfull[i, colidx[offset]] = int(val[offset])
    #             offset = offset + 1

    #return lsmfull

def read_data(tracer):
    ''' reads vector and returns it wiht the right geometry
    '''
    import numpy as np
    # arrays

    # v1d, z, dz, lsm (land sea mask)
    v1d = read_PETSc_vec("work/" + tracer)
    z   = read_PETSc_vec("data/TMM/2.8/Forcing/DomainCondition/z.petsc")
    dz  = read_PETSc_vec("data/TMM/2.8/Forcing/DomainCondition/dz.petsc")
    lsm = read_PETSc_mat("data/TMM/2.8/Geometry/landSeaMask.petsc")

    # dims
    nx, ny = lsm.shape
    nz = 15
    # v3d
    v3d = np.zeros(shape=(3, nx, ny, nz), dtype=float)
    v3d[:,:,:,:] = np.nan

    # v1d -> (v3d, z, dz)
    offset = 0
    for ix in range(nx):
        for iy in range(ny):
            length = lsm[ix, iy]
            if not length == 0:
                v3d[0, ix, iy, 0:length] = v1d[offset:offset+length]
                v3d[1, ix, iy, 0:length] = z[offset:offset+length]
                v3d[2, ix, iy, 0:length] = dz[offset:offset+length]
                offset = offset + length

    return v3d

def vectogeometry(v1d):
    ''' creates right vectorgeometry for ploting on world map
    '''
    import numpy as np
    # arrays
    # v1d, z, dz, lsm (land sea mask)
    z   = read_PETSc_vec("data/TMM/2.8/Forcing/DomainCondition/z.petsc")
    dz  = read_PETSc_vec("data/TMM/2.8/Forcing/DomainCondition/dz.petsc")
    lsm = read_PETSc_mat("data/TMM/2.8/Geometry/landSeaMask.petsc")

    # dims
    nx, ny = lsm.shape
    nz = 15
    # v3d
    v3d = np.zeros(shape=(3, nx, ny, nz), dtype=float)
    v3d[:,:,:,:] = np.nan

    # v1d -> (v3d, z, dz)
    offset = 0
    for ix in range(nx):
        for iy in range(ny):
            length = lsm[ix, iy]
            if not length == 0:
                v3d[0, ix, iy, 0:length] = v1d[offset:offset+length]
                v3d[1, ix, iy, 0:length] = z[offset:offset+length]
                v3d[2, ix, iy, 0:length] = dz[offset:offset+length]
                offset = offset + length

    return v3d

def create_figure_surface(figid, aspect, xx, yy, cmin, cmax, levels, slices, v3d):
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
    plt.clabel(p2, fmt = "%2.3f", colors = "k", fontsize = 14)
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
    #plt.savefig("solution-surface", bbox_inches="tight")
    plt.show()


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
