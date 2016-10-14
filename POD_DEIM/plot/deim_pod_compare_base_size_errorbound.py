import numpy as np
import util
import petsc_io as io
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams.update({'font.weight': 'bold'})
matplotlib.rcParams.update({'lines.linewidth': 5})
matplotlib.rcParams.update({'figure.figsize': (10, 8)})
axis_font = {'fontname':'Bitstream Vera Sans', 'weight':'bold' ,'size':'22'}

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='Bitstream Vera Sans', size=22, weight='bold')

nspinup = 100
timesteps = 12
bases = [5,20,50,80,100,120,140,160,180,200,250,300]
Y_hd = util.constructSnapshotMatrix('simulation/POD_DEIM/',"sp%.4dts%.4dN.petsc",nspinup,12,timesteps)
sPOD = io.read_PETSc_vec('reduced_basis/full_trajectory/s_POD.petsc')
sDEIM = io.read_PETSc_vec('reduced_basis/full_trajectory/s_DEIM.petsc')
U = np.load('reduced_basis/full_trajectory/fixed_deim/pod_5/U_DEIM_truncated.npy')
P = np.load('reduced_basis/full_trajectory/fixed_deim/pod_5/PT.npy')
PTU = np.linalg.inv(P.T.dot(U))
errorDEIM = np.linalg.norm(PTU,ord=2) * sDEIM[150]

err1 = np.zeros(len(bases),dtype=np.float_)
err2 = np.zeros(len(bases),dtype=np.float_)

j = 0
for i in bases:
    
    Y_reduced = util.constructSnapshotMatrix('simulation/reduced/full_trajectory/fixed_deim/pod_'+ str(i) +'/',"sp%.4dts%.4dN.petsc",nspinup,12,timesteps)
    err1[j] = np.sum(np.linalg.norm(Y_hd - Y_reduced)/np.linalg.norm(Y_hd)/nspinup)
    err2[j] = np.sum(np.square(sPOD[i:]))
    j += 1

p = plt.subplot(111)
plt.plot((5,20,50,80,100,120,140,160,180,200,250,300),err1, '--',color='black', marker='o', markersize=15)
plt.plot((5,20,50,80,100,120,140,160,180,200,250,300),(err2+errorDEIM)/2, ':',color='dimgray', marker='s', markersize=15)

bases = [5,10,20,30,40,60,100,120,140,180,200,300]
errorPOD = np.sum(np.square(sPOD[150:]))
err1 = np.zeros(len(bases),dtype=np.float_)
err2 = np.zeros(len(bases),dtype=np.float_)

j = 0
for i in bases:
    Y_reduced = util.constructSnapshotMatrix('simulation/reduced/full_trajectory/fixed_pod/deim_'+ str(i) +'/',"sp%.4dts%.4dN.petsc",nspinup,12,timesteps)
    err1[j] = np.sum(np.linalg.norm(Y_hd - Y_reduced)/np.linalg.norm(Y_hd)/nspinup)
    U = np.load('reduced_basis/full_trajectory/fixed_pod/deim_'+ str(i) +'/U_DEIM_truncated.npy')
    P = np.load('reduced_basis/full_trajectory/fixed_pod/deim_'+ str(i) +'/PT.npy')
    PTU = np.linalg.inv(P.T.dot(U))
    err2[j] = np.linalg.norm(PTU,ord=2) * sDEIM[i]
    j += 1
#p = plt.subplot(111)
plt.plot((5,10,20,30,40,60,100,120,140,180,200,300),err1, '--',color='gray', marker='D', markersize=15)
plt.plot((5,10,20,30,40,60,100,120,140,180,200,300),(err2+errorPOD)/2, ':',color='darkslategray', marker='^', markersize=15)

#p.set_yscale('log')



#plt.plot(deim_err1, '--b', marker='o', markersize=10)
#plt.show()

Y_reduced = util.constructSnapshotMatrix('simulation/reduced/full_trajectory/test01/',"sp%.4dts%.4dN.petsc",nspinup,12,timesteps)
err3 = np.sum(np.linalg.norm(Y_hd - Y_reduced)/np.linalg.norm(Y_hd)/nspinup)
U = np.load('reduced_basis/full_trajectory/U_DEIM_truncated.npy')
P = np.load('reduced_basis/full_trajectory/PT.npy')
PTU = np.linalg.inv(P.T.dot(U))
err4 = np.linalg.norm(PTU,ord=2) * sDEIM[300]


#plt.plot(300,err3,color="lightgray",marker='*',markersize=15)
#plt.plot(300,err4,color="lightgray",marker='+',markersize=20)
p.set_yscale('log')
p.set_xlim([0,320])
p.set_ylim([10e-7,1.2*10e3])
#p.set_ylim([10e-11,1.2*10e-6])
#p.set_title(r"\textbf{Average Relative Error compared to Error bound }",**axis_font)
plt.legend(["Fixed DEIM 150","Error Bound Fixed DEIM","Fixed POD 150","Error Bound Fixed POD"],loc='best')
plt.xlabel(r"\textbf{Size of other base}",**axis_font)
plt.ylabel(r"\textbf{Average Relative Error}  $\bar{\mathcal{E}}$",**axis_font)

plt.savefig('realative_error_bound.png', bbox_inches='tight')

plt.show()