import numpy as np
import util
import petsc_io as io
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'font.weight': 'bold'})
plt.rcParams.update({'lines.linewidth': 2})
plt.rcParams.update({'figure.figsize': (13, 11)})
axis_font = {'fontname':'Bitstream Vera Sans', 'weight':'bold' ,'size':'22'}
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='Bitstream Vera Sans', size=22, weight='bold')


Y_red = np.zeros((52749,3000))
for i in range(3000):
    Y_red[:,i] = io.read_PETSc_vec('simulation/reduced/last_year/POD_150_DEIM_150/sp' + str(i).zfill(4) + 'ts2879N.petsc')

Y_hd = np.zeros((52749,3000))
for i in range(3000):
    Y_hd[:,i] = io.read_PETSc_vec('simulation/POD_DEIM/sp' + str(i).zfill(4) + 'ts2879N.petsc')


hdnorm= np.linalg.norm(Y_hd,axis=0)
data1 = np.linalg.norm(Y_hd - Y_red,axis=0)/hdnorm


#Y_red = np.zeros((52749,3000))
for i in range(3000):
        Y_hd[:,i] = io.read_PETSc_vec('simulation/POD_DEIM/sp2999ts2879N.petsc')

hdnorm= np.linalg.norm(Y_hd,axis=0)
data2 = np.linalg.norm(Y_hd - Y_red,axis=0)/hdnorm

markers_on = [1000, 2000, 2998]
markers_on_shift = [1500, 2500, 2898]


p = plt.subplot(111)
plt.plot(data1,color='black',linestyle='-',linewidth=3,marker='D',markevery=markers_on)
plt.plot(data2,color='darkgray',linestyle='--',linewidth=3,marker='o',markevery=markers_on)


labels = [r"$2880\_1_{3000}P150D150$ to FOM",r"$2880\_1_{3000}P150D150$ to FOM model year 3000"]
            #r"$\parallel y_{hdl} -y_{hdl-1} \parallel_2 $",r"$\parallel y_{hd} -y_{op} \parallel_2 $"]
    #labels = [r"$\parallel y_{hdl} -y_{hdl-1} \parallel_2 $",r"$\parallel y_{hd} -y_{op} \parallel_2 $"]
    #labels = [r"$\parallel y_{rl} -y_{r-1} \parallel_2 $",r"$\parallel y_{r} -y_{op} \parallel_2 $"]

plt.legend(labels,loc='upper left')
plt.yscale('log')
    #plt.pyplot.xscale('log')
p.set_ylim([1e-2,1e1])
plt.xlabel(r"\textbf{Model Year}",**axis_font)
plt.ylabel(r"\textbf{Relative Error}  $\mathcal{E}$ ",**axis_font)

plt.savefig('error_norm_last_year.png', bbox_inches='tight')
plt.show()