import numpy as np
import util
import petsc_io as io
pod_vecs = util.constructSnapshotMatrix("/space/hydra/njp/work/0/","sp%.4dts%.4dN.petsc"
                                    ,3,12,12)
for i in range(1,10):
        pod_vecs = np.column_stack((pod_vecs,util.constructSnapshotMatrix("/space/hydra/njp/work/%.d/" % i,"sp%.4dts%.4dN.petsc"
                                    ,3,12,12)))


#pod_vecs_2 = util.constructSnapshotMatrix("/home/njp/metos3d/work/simulation/POD_DEIM/4/","sp%.4dts%.4dN.petsc"
                                    ,3000,12,12)
#pod_vecs_3 = np.column_stack((pod_vecs,pod_vecs_2))
print(pod_vecs.shape)
#p = csr_matrix(pod_vecs)
io.write_PETSc_mat_dense(pod_vecs,"POD_mat.petsc")
