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

    #does transpose the matrix samehow fixed with A.tocsr() because of row major/colum major

    import struct
    import numpy
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

    #does transpose the matrix samehow fixed with A.tocsr() because of row major/colum major

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


def read_PETSc_mat(file):
    import numpy
    from scipy.sparse import csr_matrix

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

    return csr_matrix((spdata[:,2], (spdata[:,0],spdata[:,1])), shape=(nx, ny),dtype=numpy.float_)
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
    v1d = read_PETSc_vec("simulation/" + tracer)
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

