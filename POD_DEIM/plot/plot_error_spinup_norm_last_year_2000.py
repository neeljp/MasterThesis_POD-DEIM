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

spinnorm = np.zeros((52749,3000))
for i in range(3000):
    spinnorm[:,i] = io.read_PETSc_vec('simulation/reduced/last_year/POD_150_DEIM_150/sp' + str(i).zfill(4) + 'ts2879N.petsc')

snorm= np.zeros(2999)
for i in range(2999):
    snorm[i] = np.linalg.norm(spinnorm[:,i+1] - spinnorm[:,i])
    
    
spinnorm = np.zeros((52749,3000))
for i in range(3000):
    spinnorm[:,i] = io.read_PETSc_vec('simulation/POD_DEIM/sp' + str(i).zfill(4) + 'ts2879N.petsc')



snorm_hd= np.zeros(2999)
for i in range(2999):
    snorm_hd[i] = np.linalg.norm(spinnorm[:,i+1] - spinnorm[:,i])




print(np.argmin(snorm),np.min(snorm))

markers_on = [1000, 2000, 2998]
markers_on_shift = [1500, 2500, 2898]

plt.plot(snorm_hd,color='black',linestyle='-',linewidth=3,marker='D',markevery=markers_on)
plt.plot(snorm,color='darkgray',linestyle='--',linewidth=3,marker='o',markevery=markers_on)
labels = [r"FOM",r"$2880\_1_{3000}P150D150$"]
            #r"$\parallel y_{hdl} -y_{hdl-1} \parallel_2 $",r"$\parallel y_{hd} -y_{op} \parallel_2 $"]
    #labels = [r"$\parallel y_{hdl} -y_{hdl-1} \parallel_2 $",r"$\parallel y_{hd} -y_{op} \parallel_2 $"]
    #labels = [r"$\parallel y_{rl} -y_{r-1} \parallel_2 $",r"$\parallel y_{r} -y_{op} \parallel_2 $"]

plt.legend(labels,loc='best')
plt.yscale('log')
    #plt.pyplot.xscale('log')
plt.xlabel(r"\textbf{Model Year}",**axis_font)
plt.ylabel(r"\textbf{Norm }  $[m molP/m^3]$",**axis_font)

plt.savefig('spinupnorm_last_year_2000.png', bbox_inches='tight')
plt.show()