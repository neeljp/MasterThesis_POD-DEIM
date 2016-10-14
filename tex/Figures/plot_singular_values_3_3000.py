import numpy as np
import util
import petsc_io as io
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams.update({'font.weight': 'bold'})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'figure.figsize': (8, 6)})
axis_font = {'fontname':'Bitstream Vera Sans', 'weight':'bold' ,'size':'22'}



s_deim = io.read_PETSc_vec("reduced_basis/full_trajectory/one_month/s_DEIM.petsc")


s_pod = io.read_PETSc_vec("reduced_basis/full_trajectory/one_month/s_POD.petsc")


pod_k  = util.computePODError(np.square(s_pod),1e-5)
deim_m = util.computePODError(np.square(s_deim),1e-5)
print(pod_k,deim_m)
util.plotSingular(np.square(s_pod),np.square(s_deim),1e-6,bounds=False)
plt.xlabel(r'\textbf{Sigma }  $i$',**axis_font)
plt.ylabel(r'\textbf{Value}',**axis_font)
plt.legend(["Singular values FS","Singular values of q"],loc='upper right')
plt.title(r'\textbf{Singularvalues of SVD 3\_3000}',**axis_font)
plt.savefig('singularvalues_3_3000.png', bbox_inches='tight')
plt.show()