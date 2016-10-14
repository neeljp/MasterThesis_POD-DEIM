import numpy as np
import util
import petsc_io as io
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'font.weight': 'bold'})
plt.rcParams.update({'lines.linewidth': 2})
plt.rcParams.update({'figure.figsize': (16, 10)})
axis_font = {'fontname':'Bitstream Vera Sans', 'weight':'bold' ,'size':'22'}
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='Bitstream Vera Sans', size=22, weight='bold')

ny=52749
nspinup = 1
timesteps = 1000
Y_hd_python  = np.empty([ny,np.int_(nspinup *timesteps-4)],dtype='float_')
    #Y_DOP = np.empty([ny,np.int_(ndistribution * nspinup *(ntimestep-starttimestep))],dtype='float_')
counter = 0
for s in range(0,nspinup):
    for i in range(timesteps):
        if(i % 240 != 238):
            Y_hd_python[:,counter] = io.read_PETSc_vec('simulation/compare/exp01/'+"sp%.4dts%.4dN.petsc" % (s,i))
            counter+=1
Y_hd = np.empty([ny,np.int_(nspinup *timesteps-4)],dtype='float_')
    #Y_DOP = np.empty([ny,np.int_(ndistribution * nspinup *(ntimestep-starttimestep))],dtype='float_')
counter = 0
for s in range(0,nspinup):
    for i in range(timesteps):
        if(i % 240 != 238):
            Y_hd[:,counter] = io.read_PETSc_vec('simulation/POD_DEIM/'+"sp%.4dts%.4dN.petsc" % (s,i+1))
            counter+=1

data1 = np.linalg.norm(Y_hd - Y_hd_python[:,:Y_hd.shape[1]],axis=0)

plt.plot(data1,color='darkgray',linestyle='--',linewidth=3)


labels = [r"$\parallel y^{py}_j - y^{m3d}_j \parallel_2$"]
            #r"$\parallel y_{hdl} -y_{hdl-1} \parallel_2 $",r"$\parallel y_{hd} -y_{op} \parallel_2 $"]
    #labels = [r"$\parallel y_{hdl} -y_{hdl-1} \parallel_2 $",r"$\parallel y_{hd} -y_{op} \parallel_2 $"]
    #labels = [r"$\parallel y_{rl} -y_{r-1} \parallel_2 $",r"$\parallel y_{r} -y_{op} \parallel_2 $"]

plt.legend(labels,loc='best')
plt.yscale('log')
#plt.title(r"\textbf{Absolute  Error }\textbf{of a model run with python and with PETSc}")
plt.xlabel(r"\textbf{Model Year}",**axis_font)
plt.ylabel(r"\textbf{Absolute Error}  ",**axis_font)

plt.savefig('error_norm_python_petsc.png', bbox_inches='tight')
plt.show()