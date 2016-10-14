import numpy as np
import numpy.linalg as la
import petsc_io as io
import model
import class_pod_ndop
import util
import cProfile
import time
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=20)


ndop = class_pod_ndop.ReducedModel("reduced_basis/N-DOP/1000/POD100DEIM50/option.txt")
ndop.Init()
ndop.simulate()
