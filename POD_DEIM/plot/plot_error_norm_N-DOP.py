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
    Y_red[:,i] = io.read_PETSc_vec('simulation/reduced/N-DOP/1000/POD100DEIM50/sp' + str(i).zfill(4) + 'ts2879N.petsc')

Y_hd = np.zeros((52749,3000))
for i in range(3000):
    Y_hd[:,i] = io.read_PETSc_vec('simulation/N-DOP/sp' + str(i).zfill(4) + 'ts2879N.petsc')


hdnorm= np.linalg.norm(Y_hd,axis=0)
data1 = np.linalg.norm(Y_hd - Y_red,axis=0)/hdnorm

Y_red = np.zeros((52749,2959))
for i in range(2959):
    Y_red[:,i] = io.read_PETSc_vec('simulation/reduced/N-DOP/1000/POD300DEIM300/sp' + str(i).zfill(4) + 'ts2879N.petsc')

data2 = np.linalg.norm(Y_hd[:,:2959] - Y_red,axis=0)/hdnorm[:2959]


Y_red = np.zeros((52749,3000))
for i in range(3000):
    Y_red[:,i] = io.read_PETSc_vec('simulation/reduced/N-DOP/1000/POD100DEIM50/sp' + str(i).zfill(4) + 'ts2879DOP.petsc')

for i in range(3000):
    Y_hd[:,i] = io.read_PETSc_vec('simulation/N-DOP/sp' + str(i).zfill(4) + 'ts2879DOP.petsc')


hdnorm= np.linalg.norm(Y_hd,axis=0)
data3 = np.linalg.norm(Y_hd - Y_red,axis=0)/hdnorm

Y_red = np.zeros((52749,2959))
for i in range(2959):
    Y_red[:,i] = io.read_PETSc_vec('simulation/reduced/N-DOP/1000/POD300DEIM300/sp' + str(i).zfill(4) + 'ts2879DOP.petsc')

data4 = np.linalg.norm(Y_hd[:,:2959] - Y_red,axis=0)/hdnorm[:2959]

markers_on = [1000, 2000, 2998]
markers_on_shift = [ 500, 1000,2000,2950]


p = plt.subplot(111)
plt.plot(data1,color='black',linestyle='-',linewidth=3,marker='D',markevery=markers_on,markersize=10)
plt.plot(data2,color='darkgray',linestyle='--',linewidth=3,marker='o',markevery=markers_on_shift,markersize=10)
plt.plot(data3,color='dimgray',linestyle='-.',linewidth=3,marker='s',markevery=markers_on,markersize=10)
plt.plot(data4,color='darkslategray',linestyle=':',linewidth=3,marker='^',markevery=markers_on_shift,markersize=10)


labels = [r"36\_1000P100D50N",r"36\_1000P300D300N",r"36\_1000P100D50DOP",r"36\_1000P300D300DOP"]
            #r"$\parallel y_{hdl} -y_{hdl-1} \parallel_2 $",r"$\parallel y_{hd} -y_{op} \parallel_2 $"]
    #labels = [r"$\parallel y_{hdl} -y_{hdl-1} \parallel_2 $",r"$\parallel y_{hd} -y_{op} \parallel_2 $"]
    #labels = [r"$\parallel y_{rl} -y_{r-1} \parallel_2 $",r"$\parallel y_{r} -y_{op} \parallel_2 $"]

plt.legend(labels,loc='upper left')
plt.yscale('log')
p.set_ylim([10e-5,10e2])
    #plt.pyplot.xscale('log')
plt.xlabel(r"\textbf{Model Year}",**axis_font)
plt.ylabel(r"\textbf{Relative Error}  $\mathcal{E}$ ",**axis_font)

plt.savefig('error_norm_N-DOP.png', bbox_inches='tight')
plt.show()