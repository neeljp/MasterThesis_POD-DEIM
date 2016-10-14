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
    spinnorm[:,i] = io.read_PETSc_vec('simulation/reduced/full_trajectory/POD_100_DEIM_50/sp' + str(i).zfill(4) + 'ts2879N.petsc')

snorm= np.zeros(2999)
for i in range(2999):
    snorm[i] = np.linalg.norm(spinnorm[:,i+1] - spinnorm[:,i])
    
    
spinnorm = np.zeros((52749,5000))
for i in range(5000):
    spinnorm[:,i] = io.read_PETSc_vec('simulation/POD_DEIM/sp' + str(i).zfill(4) + 'ts2879N.petsc')



snorm_hd= np.zeros(4999)
for i in range(4999):
    snorm_hd[i] = np.linalg.norm(spinnorm[:,i+1] - spinnorm[:,i])


spinnorm = np.zeros((52749,3000))
for i in range(3000):
    spinnorm[:,i] = io.read_PETSc_vec('simulation/reduced/full_trajectory/test01/sp' + str(i).zfill(4) + 'ts2879N.petsc')



snorm_300= np.zeros(2999)
for i in range(2999):
    snorm_300[i] = np.linalg.norm(spinnorm[:,i+1] - spinnorm[:,i])

spinnorm = np.zeros((52749,3000))
for i in range(3000):
    spinnorm[:,i] = io.read_PETSc_vec('simulation/reduced/full_trajectory/6000/sp' + str(i).zfill(4) + 'ts2879N.petsc')



snorm_6000_300= np.zeros(2999)
for i in range(2999):
    snorm_6000_300[i] = np.linalg.norm(spinnorm[:,i+1] - spinnorm[:,i])


spinnorm = np.zeros((52749,3000))
for i in range(3000):
    spinnorm[:,i] = io.read_PETSc_vec('simulation/reduced/full_trajectory/3_snapshots_per_month/POD_100_DEIM_50/sp' + str(i).zfill(4) + 'ts2879N.petsc')



snorm_3_100_50= np.zeros(2999)
for i in range(2999):
    snorm_3_100_50[i] = np.linalg.norm(spinnorm[:,i+1] - spinnorm[:,i])



spinnorm = np.zeros((52749,5000))
for i in range(5000):
        spinnorm[:,i] = io.read_PETSc_vec('simulation/reduced/full_trajectory/3_snapshots_per_month/sp' + str(i).zfill(4) + 'ts2879N.petsc')



snorm_3_per_month = np.zeros(4999)
for i in range(4999):
        if(i > 2995 and i < 3005):
                if(i <3000):
                        snorm_3_per_month[i] = snorm_3_per_month[i-1]
                else:
                        snorm_3_per_month[i] = np.linalg.norm(spinnorm[:,3006] - spinnorm[:,3005]) 
        else:
                snorm_3_per_month[i] = np.linalg.norm(spinnorm[:,i+1] - spinnorm[:,i])

spinnorm = np.zeros((52749,5000))
for i in range(5000):
        spinnorm[:,i] = io.read_PETSc_vec('simulation/reduced/full_trajectory/3_snapshots_per_month/1000/POD_300_DEIM_300/sp' + str(i).zfill(4) + 'ts2879N.petsc')

snorm_3_1000= np.zeros(4999)

for i in range(4999):
    snorm_3_1000[i] = np.linalg.norm(spinnorm[:,i+1] - spinnorm[:,i])

markers_on_4000 = [1000, 2000, 3000, 4000, 4500]
markers_on = [1000, 2000, 2998]
markers_on_shift = [1500, 2500, 2898]

plt.plot(snorm_hd,color='black',linestyle='-',linewidth=3,marker='D',markevery=markers_on_4000)
plt.plot(snorm,color='darkgray',linestyle='--',linewidth=3,marker='o',markevery=markers_on)
plt.plot(snorm_300,color='dimgray',linestyle='-.',linewidth=3,marker='s',markevery=markers_on)
plt.plot(snorm_6000_300,color='darkslategray',linestyle=':',linewidth=3,marker='^',markevery=markers_on_shift)
plt.plot(snorm_3_100_50,color='slategray',linestyle='--',linewidth=3,marker='h',markevery=markers_on)
plt.plot(snorm_3_per_month,color='gray',linestyle=':',linewidth=3,marker='D',markevery=markers_on_4000)
plt.plot(snorm_3_1000,color='lightgray',linestyle='-.',linewidth=3,marker='>',markevery=markers_on_4000)

labels = [r"FOM",r"12\_3000P100D50",r"12\_3000P300D300",r"12\_6000P300D300",r"36\_3000P100D50",r"36\_3000P300D300",r"36\_1000P300D300"]
            #r"$\parallel y_{hdl} -y_{hdl-1} \parallel_2 $",r"$\parallel y_{hd} -y_{op} \parallel_2 $"]
    #labels = [r"$\parallel y_{hdl} -y_{hdl-1} \parallel_2 $",r"$\parallel y_{hd} -y_{op} \parallel_2 $"]
    #labels = [r"$\parallel y_{rl} -y_{r-1} \parallel_2 $",r"$\parallel y_{r} -y_{op} \parallel_2 $"]

plt.legend(labels,loc='best')
plt.yscale('log')
    #plt.pyplot.xscale('log')
plt.xlabel(r"\textbf{Model Year}",**axis_font)
plt.ylabel(r"\textbf{Norm }  $[m molP/m^3]$",**axis_font)

plt.savefig('spinupnorm_base_compare.png', bbox_inches='tight')
plt.show()