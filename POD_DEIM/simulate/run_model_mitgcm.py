import numpy as np
import numpy.linalg as la
import petsc_io as io
import model
import class_pod_mitgcm
import util
import cProfile
import time
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=20)


mitgcm = class_pod_mitgcm.ReducedModel("reduced_basis/MITgcm/POD300DEIM300/option.txt")
mitgcm.Init()
mitgcm.simulate()
