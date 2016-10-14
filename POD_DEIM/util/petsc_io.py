import argparse
import math
import os
import sys
import numpy


def read_PETSc_vec(file):
    """
    Reads a petsc vector file.
    Parameters
    ----------
    file : str 
            Path to file
    Returns
    -------
    v :  file data as numpy array
    """
    # open file
    # omit header
    # read length
    # read values
    # close file
    if not os.path.exists(file):
        raise IOError("%s not found." % file)

    f = open(file, "rb")
    numpy.fromfile(f, dtype=">i4", count=1)
    nvec = numpy.fromfile(f, dtype=">i4", count=1)
    #load data and change it to little endian, importend for np.dot
    v = numpy.fromfile(f, dtype=">f8", count=nvec[0]).astype('<f8')
    f.close()

    return v

def write_PETSc_vec(v,file):
    """
    Writes a numpy array to petsc vector file.
    Parameters
    ----------
    v    : numpy array 
    file : str 
            Path to file
    """
    try:
        f = open(file, "wb")
    except:
            print("IO error:", sys.exc_info()[0],file)

    header      = numpy.array([1211214])
    nx          = numpy.array(v.shape[0])
    header.astype('>i4').tofile(f)
    nx.astype('>i4').tofile(f)
    v.astype('>f8').tofile(f)
    f.close()

    return 0


def write_PETSc_mat(A,file):
    """
    Writes a numpy array to petsc sparse matrix file.
    Parameters
    ----------
    v    : numpy array 
    file : str 
            Path to file
    """
    try:
        f = open(file, "wb")
    except:
            print("IO error:", sys.exc_info()[0],file)
    header          = numpy.array([1211216])
    dims            = A.shape
    nx              = numpy.array(dims[0])
    ny              = numpy.array(dims[1])
    nnz             = numpy.array([A.nnz])
    rowidx,colidx   = A.nonzero()
    nrow,k          = numpy.histogram(rowidx,range(0,dims[0]+1))

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

def write_PETSc_mat_dense(A,file):
    """
    Writes a numpy array to petsc dense matrix file.
    Parameters
    ----------
    v    : numpy array 
    file : str 
            Path to file
    """
    try:
        f = open(file, "wb")
    except:
            print("IO error:", sys.exc_info()[0],file)

    import struct
    import numpy
    header          = numpy.array([1211216])
    dims            = A.shape
    nx              = numpy.array(dims[0])
    ny              = numpy.array(dims[1])
    matrixFormat    = numpy.array([-1])


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
    matrixFormat.astype('>i4').tofile(f)
    A.astype('>f8').tofile(f)
    f.close()
    return 0


def read_PETSc_mat_dense(file):
    """
    Reads a petsc dense matrix file.
    Parameters
    ----------
    file : str 
            Path to file
    Returns
    -------
    v :  file data as numpy array
    """
    # open file
    # omit header
    # read length
    # read values
    # close file
    if not os.path.exists(file):
        raise IOError("%s not found." % file)
    f = open(file, "rb")
    # omit header
    numpy.fromfile(f, dtype=">i4", count=1)
    # read dims
    nx     = numpy.fromfile(f, dtype=">i4", count=1)
    ny     = numpy.fromfile(f, dtype=">i4", count=1)
    format = numpy.fromfile(f, dtype=">i4", count=1)
    val    = numpy.fromfile(f, dtype=">f8", count=(ny[0]*nx[0]))

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
    #create full matrix
    mat = numpy.zeros(shape=(nx[0], ny[0]), dtype=numpy.float_)
    offset = 0
    for i in range(nx[0]):
            for j in range(ny[0]):
                mat[i, j] = val[offset]
                offset = offset + 1
                #print (numpy.nonzero(lsmfull),i,j,offset,val[offset] )
    return mat

def read_PETSc_mat(file):
    """
    Reads a petsc sparse matrix file.
    Parameters
    ----------
    file : str 
            Path to file
    Returns
    -------
    v :  file data as numpy array
    """
    from scipy.sparse import csr_matrix

    # open file
    if not os.path.exists(file):
        raise IOError("%s not found." % file)
    f = open(file, "rb")
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

    return csr_matrix((spdata[:,2], (spdata[:,0],spdata[:,1])), shape=(nx, ny),dtype=numpy.float_)

def read_data(tracer):
    ''' Reads vector and returns it with the right world geometry
    Parameters
    ----------
    tracer : str 
            Path to file
    Returns
    -------
    v3d :  file data formated with right world geometry as numpy array
    '''
    import numpy as np
    # arrays

    # v1d, z, dz, lsm (land sea mask)
    v1d = read_PETSc_vec(tracer)
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
       Parameters
    ----------
    v1d : array 
            tracer vector
    Returns
    -------
    v3d :  file data formated with right world geometry as numpy array
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
            length = int(lsm[ix, iy])
            if not length == 0:
                v3d[0, ix, iy, 0:length] = v1d[offset:offset+length]
                v3d[1, ix, iy, 0:length] = z[offset:offset+length]
                v3d[2, ix, iy, 0:length] = dz[offset:offset+length]
                offset = offset + length
    return v3d
