import numpy as np
import util
import petsc_io as io
pod_vecs = util.constructSnapshotMatrix("/home/njp/metos3d/work/simulation/POD_DEIM/3/","sp%.4dts%.4dN.petsc"
                                    ,3000,12,12)
pod_vecs_2 = util.constructSnapshotMatrix("/home/njp/metos3d/work/simulation/POD_DEIM/4/","sp%.4dts%.4dN.petsc"
                                    ,3000,12,12)
pod_vecs_3 = np.column_stack((pod_vecs,pod_vecs_2))
print(pod_vecs_3.shape)
#p = csr_matrix(pod_vecs)
io.write_PETSc_mat_dense(pod_vecs_3,"POD_mat.petsc")
