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


Y_red = np.zeros((52749,3000))
for i in range(3000):
    Y_red[:,i] = io.read_PETSc_vec('simulation/reduced/10_lhc/1000/POD300DEIM300/lhc_0/sp' + str(i).zfill(4) + 'ts2879N.petsc')

Y_hd = np.zeros((52749,3000))
for i in range(3000):
    Y_hd[:,i] = io.read_PETSc_vec('simulation/lhc/0/sp' + str(i).zfill(4) + 'ts2879N.petsc')


hdnorm= np.linalg.norm(Y_hd,axis=0)
data1 = np.linalg.norm(Y_hd - Y_red,axis=0)/hdnorm

#Y_red = np.zeros((52749,3000))
for i in range(3000):
    Y_red[:,i] = io.read_PETSc_vec('simulation/reduced/full_trajectory/POD_100_DEIM_50_lhc_0/sp' + str(i).zfill(4) + 'ts2879N.petsc')

data2 = np.linalg.norm(Y_hd - Y_red,axis=0)/hdnorm

for i in range(3000):
    Y_red[:,i] = io.read_PETSc_vec('simulation/POD_DEIM/sp' + str(i).zfill(4) + 'ts2879N.petsc')

data3 = np.linalg.norm(Y_hd - Y_red,axis=0)/hdnorm
markers_on = [1000, 2000, 2998]

p = plt.subplot(111)
plt.plot(data1,color='black',linestyle='-',linewidth=3,marker='D',markevery=markers_on,markersize=10)
plt.plot(data2,color='darkgray',linestyle='--',linewidth=3,marker='o',markevery=markers_on,markersize=10)
plt.plot(data3,color='dimgray',linestyle='-.',linewidth=3,marker='s',markevery=markers_on,markersize=10)


labels = [r"LHC10\_36\_3000P300D300",r"12\_3000P100D50",r"FOM with $u_r$"]
            #r"$\parallel y_{hdl} -y_{hdl-1} \parallel_2 $",r"$\parallel y_{hd} -y_{op} \parallel_2 $"]
    #labels = [r"$\parallel y_{hdl} -y_{hdl-1} \parallel_2 $",r"$\parallel y_{hd} -y_{op} \parallel_2 $"]
    #labels = [r"$\parallel y_{rl} -y_{r-1} \parallel_2 $",r"$\parallel y_{r} -y_{op} \parallel_2 $"]
plt.legend(labels,loc='upper left')
plt.yscale('log')
p.set_ylim([10e-2,10e0])
    #plt.pyplot.xscale('log')
plt.xlabel(r"\textbf{Model Year}",**axis_font)
plt.ylabel(r"\textbf{Relative Error}  $\mathcal{E}$ ",**axis_font)

plt.savefig('error_norm_lhc_0.png', bbox_inches='tight')
plt.show()