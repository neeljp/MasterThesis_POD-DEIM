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
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='Bitstream Vera Sans', size=22, weight='bold')



s_deim = io.read_PETSc_vec("reduced_basis/last_year/s_DEIM.petsc")


s_pod = io.read_PETSc_vec("reduced_basis/last_year/s_POD.petsc")


pod_k  = util.computePODError(np.square(s_pod),1e-5)
deim_m = util.computePODError(np.square(s_deim),1e-5)
print(pod_k,deim_m)
util.plotSingular(np.square(s_pod),np.square(s_deim),1e-6,bounds=False)
plt.xlabel(r'\textbf{Sigma } $i$',**axis_font)
plt.ylabel(r'\textbf{Value}',**axis_font)
plt.legend(["Singular values FS","Singular values of q"],loc='best')
#plt.title(r'\textbf{Singularvalues of SVD 12\_3000}',**axis_font)
plt.savefig('singularvalues_last_year.png', bbox_inches='tight')
plt.show()