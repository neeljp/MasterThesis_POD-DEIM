import util
import petsc_io as io
import numpy as np

A = util.constructSnapshotMatrix("../../../metos3d/work/simulation/POD_DEIM/9/","sp%.4dts%.4dN.petsc",3000,12,12)
B = util.constructSnapshotMatrix("../../../metos3d/work/simulation/POD_DEIM/8/","sp%.4dts%.4dN.petsc",3000,12,12)
C = util.constructSnapshotMatrix("../../../metos3d/work/simulation/POD_DEIM/7/","sp%.4dts%.4dN.petsc",3000,12,12)
D = util.constructSnapshotMatrix("../../../metos3d/work/simulation/POD_DEIM/6/","sp%.4dts%.4dN.petsc",3000,12,12)
E = np.column_stack((A,B,C,D))
U,s,V = util.computeSVD(E)
np.save("U_6_7_8_9_6000.npy",U)
np.save("S_6_7_8_9_6000.npy",s)
