import numpy as np
import util
import petsc_io as io
import matplotlib
import matplotlib as plt

matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams.update({'font.weight': 'bold'})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'figure.figsize': (12, 10)})
axis_font = {'fontname':'Bitstream Vera Sans', 'weight':'bold' ,'size':'22'}

Y_reduced = util.constructSnapshotMatrix('simulation/reduced/full_trajectory/POD_100_DEIM_50/',"sp%.4dts%.4dN.petsc",nspinup,12,timesteps)
Y_reduced_summer = util.constructSnapshotMatrix('simulation/reduced/full_trajectory/one_month/',"sp%.4dts%.4dN.petsc",nspinup,12,timesteps)
Y_hd = util.constructSnapshotMatrix('simulation/POD_DEIM/',"sp%.4dts%.4dN.petsc",nspinup,12,timesteps)
n = np.ones(Y_reduced.shape[0]) * 2.17

    #plt.pyplot.title('Error norm of '+ distribution + str(ndistribution).zfill(2) + ' created with metos3d')
#plt.pyplot.title('Error norm of '+ distribution + str(ndistribution).zfill(2) + ' created with '+ reducedBasis )
plt.pyplot.title('Error norm of metos3d and implementation in python')


    #numerical_correction = np.arange(0,1000)*1000*2.5e-16
    #print(numerical_correction)
data1 = np.linalg.norm(Y_hd - Y_reduced[:,:Y_hd.shape[1]],axis=0)
data2 = np.linalg.norm(Y_hd - Y_reduced_summer[:,:Y_hd.shape[1]],axis=0)
    #data3 = np.linalg.norm(Y_hd - Y_reduced[:,:Y_hd.shape[1]],axis=0)/np.linalg.norm(Y_hd)
plt.pyplot.plot(data1,color='darkgray',linestyle='--',linewidth=2)
plt.pyplot.plot(data2,color='black',linestyle=':',linewidth=2)
    #plt.pyplot.plot(data3,color='dimgray',linestyle='-',linewidth=2)
    #plt.pyplot.plot(np.linalg.norm(Y_reduced[:,1:] - Y_reduced[:, 0:nspinup_reduced*(13-timesteps-1)-1] ,axis=0),color='black',linestyle=':',linewidth=2)
    #plt.pyplot.plot(np.linalg.norm(Y_reduced - n[:, np.newaxis] ,axis=0),color='slategray',linestyle='-.',linewidth=2)
    #plt.pyplot.plot(np.linalg.norm(Y_hd[:,1:] - Y_hd[:, 0:nspinup*(13-timesteps-1)-1] ,axis=0),color='dimgray',linestyle='-',linewidth=2)
    #plt.pyplot.plot(np.linalg.norm(Y_hd - n[:, np.newaxis] ,axis=0),color='black',linestyle='-',linewidth=2)

    #labels = [r"$\parallel y_{hd} -y_r \parallel_2 $",r"$\parallel y_{r} -y_{op} \parallel_2 $",r"$\parallel y_{rl} -y_{rl-1} \parallel_2 $",
            #r"$\parallel y_{hdl} -y_{hdl-1} \parallel_2 $",r"$\parallel y_{hd} -y_{op} \parallel_2 $"]
    #labels = [r"$\parallel y_{hdl} -y_{hdl-1} \parallel_2 $",r"$\parallel y_{hd} -y_{op} \parallel_2 $"]
    #labels = [r"$\parallel y_{rl} -y_{r-1} \parallel_2 $",r"$\parallel y_{r} -y_{op} \parallel_2 $"]

    #plt.pyplot.legend(labels,loc='best')
plt.pyplot.yscale('log')
    #plt.pyplot.xscale('log')
plt.pyplot.xlabel('Model Year')
plt.pyplot.ylabel(r"\textbf{norm}  $\frac{mmol P}{m^3} $")
plt.pyplot.show()