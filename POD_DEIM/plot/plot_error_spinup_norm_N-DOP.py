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
    spinnorm[:,i] = io.read_PETSc_vec('simulation/reduced/N-DOP/1000/POD100DEIM50/sp' + str(i).zfill(4) + 'ts2879N.petsc')

snormN= np.zeros(2999)
for i in range(2999):
    snormN[i] = np.linalg.norm(spinnorm[:,i+1] - spinnorm[:,i])
    
    
spinnorm = np.zeros((52749,3000))
for i in range(3000):
    spinnorm[:,i] = io.read_PETSc_vec('simulation/N-DOP/sp' + str(i).zfill(4) + 'ts2879N.petsc')



snorm_hdN= np.zeros(2999)
for i in range(2999):
    snorm_hdN[i] = np.linalg.norm(spinnorm[:,i+1] - spinnorm[:,i])


spinnorm = np.zeros((52749,3000))
for i in range(3000):
    spinnorm[:,i] = io.read_PETSc_vec('simulation/reduced/N-DOP/1000/POD100DEIM50/sp' + str(i).zfill(4) + 'ts2879DOP.petsc')

snormDOP= np.zeros(2999)
for i in range(2999):
    snormDOP[i] = np.linalg.norm(spinnorm[:,i+1] - spinnorm[:,i])
    
    
spinnorm = np.zeros((52749,3000))
for i in range(3000):
    spinnorm[:,i] = io.read_PETSc_vec('simulation/N-DOP/sp' + str(i).zfill(4) + 'ts2879DOP.petsc')



snorm_hdDOP= np.zeros(2999)
for i in range(2999):
    snorm_hdDOP[i] = np.linalg.norm(spinnorm[:,i+1] - spinnorm[:,i])

spinnorm = np.zeros((52749,2959))
for i in range(2959):
    spinnorm[:,i] = io.read_PETSc_vec('simulation/reduced/N-DOP/1000/POD300DEIM300/sp' + str(i).zfill(4) + 'ts2879N.petsc')

snorm3N= np.zeros(2958)
for i in range(2958):
    snorm3N[i] = np.linalg.norm(spinnorm[:,i+1] - spinnorm[:,i])

spinnorm = np.zeros((52749,2959))
for i in range(2959):
    spinnorm[:,i] = io.read_PETSc_vec('simulation/reduced/N-DOP/1000/POD300DEIM300/sp' + str(i).zfill(4) + 'ts2879DOP.petsc')

snorm3DOP= np.zeros(2958)
for i in range(2958):
    snorm3DOP[i] = np.linalg.norm(spinnorm[:,i+1] - spinnorm[:,i])

markers_on = [1000, 2000, 2998]
markers_on_shift = [500,1500, 2559]
p = plt.subplot(111)
plt.plot(snorm_hdN,color='black',linestyle='-',linewidth=3,marker='D',markevery=markers_on,markersize=10)
plt.plot(snorm_hdDOP,color='dimgray',linestyle='-.',linewidth=3,marker='s',markevery=markers_on,markersize=10)
plt.plot(snormN,color='darkgray',linestyle='--',linewidth=3,marker='o',markevery=markers_on,markersize=10)
plt.plot(snormDOP,color='darkslategray',linestyle=':',linewidth=3,marker='^',markevery=markers_on,markersize=10)
plt.plot(snorm3N,color='slategray',linestyle='--',linewidth=3,marker='*',markevery=markers_on_shift,markersize=10)
plt.plot(snorm3DOP,color='gray',linestyle=':',linewidth=3,marker='>',markevery=markers_on_shift,markersize=10)

labels = [r"FOMN",r"FOMDOP",r"36\_1000P100D50N",r"36\_1000P100D50DOP",r"36\_1000P300D300N",r"36\_1000P300D300DOP"]
            #r"$\parallel y_{hdl} -y_{hdl-1} \parallel_2 $",r"$\parallel y_{hd} -y_{op} \parallel_2 $"]
    #labels = [r"$\parallel y_{hdl} -y_{hdl-1} \parallel_2 $",r"$\parallel y_{hd} -y_{op} \parallel_2 $"]
    #labels = [r"$\parallel y_{rl} -y_{r-1} \parallel_2 $",r"$\parallel y_{r} -y_{op} \parallel_2 $"]

plt.legend(labels,loc='best')
plt.yscale('log')
p.set_ylim([10e-6,10e2])
    #plt.pyplot.xscale('log')
plt.xlabel(r"\textbf{Model Year}",**axis_font)
plt.ylabel(r"\textbf{Norm }  $[m molP/m^3]$",**axis_font)

plt.savefig('spinupnorm_base_compare_N-DOP.png', bbox_inches='tight')
plt.show()