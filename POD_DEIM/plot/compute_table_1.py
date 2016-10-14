import numpy as np
import util
import petsc_io as io
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'font.weight': 'bold'})
plt.rcParams.update({'lines.linewidth': 2})
plt.rcParams.update({'figure.figsize': (12, 10)})
axis_font = {'fontname':'Bitstream Vera Sans', 'weight':'bold' ,'size':'22'}
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='Bitstream Vera Sans', size=22, weight='bold')


format = '\\hline \\text{%s}  & %0.5f & %0.5f  & %0.5f & %0.5f & %0.5f\\\\\n'
format_s = '\\hline %s  & %0.5f & %0.5f  & %0.5f & %0.5f & %0.5f\\\\\n'
formatd = '\\hline %s  & %d & %d  & %d & %d & %d\\\\\n'
formats = '\\hline %s  & %s & %s & %s & %s & %s\\\\\n'
n = 3000
n_ = 1000
time_hd =275.8
data11 = time_hd/np.mean(np.load('simulation/reduced/full_trajectory/3_snapshots_per_month/POD_100_DEIM_50/sp0009ts2879N.petsc_timeings.npy')[:10])
data12 = time_hd/np.mean(np.load('simulation/reduced/full_trajectory/3_snapshots_per_month/sp0009ts2879N.petsc_timeings.npy')[:10])
data13 = time_hd/np.mean(np.load('simulation/reduced/full_trajectory/3_snapshots_per_month/1000/POD_100_DEIM_50/sp0009ts2879N.petsc_timeings.npy')[:10])
data14 = time_hd/np.mean(np.load('simulation/reduced/full_trajectory/3_snapshots_per_month/1000/POD_300_DEIM_300/sp0009ts2879N.petsc_timeings.npy')[:10])
print('\\text{Base}  & P100D50 & P300D300  & P100D50 & P300D300  & FOM\\\\')
print(formats % ('t_{setup} [s]','(9676,10,39)','(9676,870,327)','(2071,9,38)','(2071,989,350)','-'))
print(format_s % ('S_P',data11,data12,data13,data14,1))

print(formatd % ('S_R',1/data11 *1000 +2028,1/data12 *1000 +1995,1/data13 *1000 +2114,1/data14 *1000 +1996,0))
print(formatd % ('S_R',1/data11 *3000 +1711,1/data12 *3000 +40,1/data13 *3000 +2755,1/data14 *3000 +758,0))


sPOD = io.read_PETSc_vec('reduced_basis/full_trajectory/3_snapshots_per_month/s_POD.petsc')
sDEIM = io.read_PETSc_vec('reduced_basis/full_trajectory/3_snapshots_per_month/s_DEIM.petsc')
U = np.load('reduced_basis/full_trajectory/3_snapshots_per_month/POD_100_DEIM_50/U_DEIM_truncated.npy')
P = np.load('reduced_basis/full_trajectory/3_snapshots_per_month/POD_100_DEIM_50/PT.npy')
PTU = np.linalg.inv(P.T.dot(U))
data51 = np.linalg.norm(PTU,ord=2) * sDEIM[50]
data52 = np.sum(np.square(sPOD[100:]))

sPOD = io.read_PETSc_vec('reduced_basis/full_trajectory/3_snapshots_per_month/s_POD.petsc')
sDEIM = io.read_PETSc_vec('reduced_basis/full_trajectory/3_snapshots_per_month/s_DEIM.petsc')
U = np.load('reduced_basis/full_trajectory/3_snapshots_per_month/U_DEIM_truncated.npy')
P = np.load('reduced_basis/full_trajectory/3_snapshots_per_month/PT.npy')
PTU = np.linalg.inv(P.T.dot(U))
data53= np.linalg.norm(PTU,ord=2) * sDEIM[300]
data54 = np.sum(np.square(sPOD[300:]))


sPOD = io.read_PETSc_vec('reduced_basis/full_trajectory/3_snapshots_per_month/1000/s_POD.petsc')
sDEIM = io.read_PETSc_vec('reduced_basis/full_trajectory/3_snapshots_per_month/1000/s_DEIM.petsc')
U = np.load('reduced_basis/full_trajectory/3_snapshots_per_month/1000/POD_100_DEIM_50/U_DEIM_truncated.npy')
P = np.load('reduced_basis/full_trajectory/3_snapshots_per_month/1000/POD_100_DEIM_50/PT.npy')
PTU = np.linalg.inv(P.T.dot(U))
data55 = np.linalg.norm(PTU,ord=2) * sDEIM[50]
data56 = np.sum(np.square(sPOD[100:]))


sPOD = io.read_PETSc_vec('reduced_basis/full_trajectory/3_snapshots_per_month/1000/s_POD.petsc')
sDEIM = io.read_PETSc_vec('reduced_basis/full_trajectory/3_snapshots_per_month/1000/s_DEIM.petsc')
U = np.load('reduced_basis/full_trajectory/3_snapshots_per_month/1000/POD_300_DEIM_300/U_DEIM_truncated.npy')
P = np.load('reduced_basis/full_trajectory/3_snapshots_per_month/1000/POD_300_DEIM_300/PT.npy')
PTU = np.linalg.inv(P.T.dot(U))
data57 = np.linalg.norm(PTU,ord=2) * sDEIM[300]
data58 = np.sum(np.square(sPOD[300:]))

print(data51,data52,data53,data54,data55,data56,data57,data58)
print(format_s % ('Error Bound',(data51+data52)/1000,(data53+data54)/1000,(data55+data56)/1000,(data57+data58)/1000,0))

y0 = np.ones(52749) * 2.17
volumes = io.read_PETSc_vec('data/TMM/2.8/Geometry/volumes.petsc')
v0= np.sum(y0 *volumes)

data31 =  np.abs(np.sum(io.read_PETSc_vec('simulation/reduced/full_trajectory/3_snapshots_per_month/POD_100_DEIM_50/sp0999ts2879N.petsc')*volumes)-v0)/v0
data32 =  np.abs(np.sum(io.read_PETSc_vec('simulation/reduced/full_trajectory/3_snapshots_per_month/sp0999ts2879N.petsc')*volumes)-v0)/v0
data33 =  np.abs(np.sum(io.read_PETSc_vec('simulation/reduced/full_trajectory/3_snapshots_per_month/1000/POD_100_DEIM_50/sp0999ts2879N.petsc')*volumes)-v0)/v0
data34 =  np.abs(np.sum(io.read_PETSc_vec('simulation/reduced/full_trajectory/3_snapshots_per_month/1000/POD_300_DEIM_300/sp0999ts2879N.petsc')*volumes)-v0)/v0
data35 =  np.abs(np.sum(io.read_PETSc_vec('simulation/POD_DEIM/sp0999ts2879N.petsc')*volumes)-v0)/v0
print(format_s % ('M_{1000}',data31,data32,data33,data34,data35))


data31 =  np.abs(np.sum(io.read_PETSc_vec('simulation/reduced/full_trajectory/3_snapshots_per_month/POD_100_DEIM_50/sp2999ts2879N.petsc')*volumes)-v0)/v0
data32 =  np.abs(np.sum(io.read_PETSc_vec('simulation/reduced/full_trajectory/3_snapshots_per_month/sp2999ts2879N.petsc')*volumes)-v0)/v0
data33 =  np.abs(np.sum(io.read_PETSc_vec('simulation/reduced/full_trajectory/3_snapshots_per_month/1000/POD_100_DEIM_50/sp2999ts2879N.petsc')*volumes)-v0)/v0
data34 =  np.abs(np.sum(io.read_PETSc_vec('simulation/reduced/full_trajectory/3_snapshots_per_month/1000/POD_300_DEIM_300/sp2999ts2879N.petsc')*volumes)-v0)/v0
data35 =  np.abs(np.sum(io.read_PETSc_vec('simulation/POD_DEIM/sp2999ts2879N.petsc')*volumes)-v0)/v0
print(format_s % ('M_{3000}',data31,data32,data33,data34,data35))

Y_hd = np.zeros((52749,n))
for i in range(n):
    Y_hd[:,i] = io.read_PETSc_vec('simulation/POD_DEIM/sp' + str(i).zfill(4) + 'ts2879N.petsc')

hdnorm= np.linalg.norm(Y_hd,axis=0)
hdnorm_= np.linalg.norm(Y_hd[:,:999],axis=0)

Y_red = np.zeros((52749,n))
for i in range(n):
    Y_red[:,i] = io.read_PETSc_vec('simulation/reduced/full_trajectory/3_snapshots_per_month/POD_100_DEIM_50/sp' + str(i).zfill(4) + 'ts2879N.petsc')


data21 = np.sum(np.linalg.norm(Y_hd - Y_red,axis=0)/hdnorm)/n
data41 = np.sum(np.linalg.norm(Y_hd[:,:999] - Y_red[:,:999],axis=0)/hdnorm_)/n_


for i in range(n):
    Y_red[:,i] = io.read_PETSc_vec('simulation/reduced/full_trajectory/3_snapshots_per_month/1000/POD_100_DEIM_50/sp' + str(i).zfill(4) + 'ts2879N.petsc')


data23 = np.sum(np.linalg.norm(Y_hd - Y_red,axis=0)/hdnorm)/n
data43 = np.sum(np.linalg.norm(Y_hd[:,:999] - Y_red[:,:999],axis=0)/hdnorm_)/n_

#Y_red = np.zeros((52749,n))
for i in range(n):
        Y_red[:,i] = io.read_PETSc_vec('simulation/reduced/full_trajectory/3_snapshots_per_month/sp' + str(i).zfill(4) + 'ts2879N.petsc')

data22 = np.sum(np.linalg.norm(Y_hd - Y_red,axis=0)/hdnorm)/n
data42 = np.sum(np.linalg.norm(Y_hd[:,:999] - Y_red[:,:999],axis=0)/hdnorm_)/n_

#Y_red = np.zeros((52749,n))
for i in range(n):
        Y_red[:,i] = io.read_PETSc_vec('simulation/reduced/full_trajectory/3_snapshots_per_month/1000/POD_300_DEIM_300/sp' + str(i).zfill(4) + 'ts2879N.petsc')

data24 = np.sum(np.linalg.norm(Y_hd - Y_red,axis=0)/hdnorm)/n
data44 = np.sum(np.linalg.norm(Y_hd[:,:999] - Y_red[:,:999],axis=0)/hdnorm_)/n_

print(format_s % ('\\bar{\mathcal{E}}_{1000}',data41,data42,data43,data44,0))
print(format_s % ('\\bar{\mathcal{E}}_{3000}',data21,data22,data23,data24,0))


print(formatd % ('R_{1000}',2028,1995,2114,1996,0))
print(formatd % ('R_{3000}',1711,40,2755,758,0))




