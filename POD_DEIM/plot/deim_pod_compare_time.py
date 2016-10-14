import numpy as np
import util
import petsc_io as io
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams.update({'font.weight': 'bold'})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'figure.figsize': (10, 8)})
axis_font = {'fontname':'Bitstream Vera Sans', 'weight':'bold' ,'size':'22'}

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='Bitstream Vera Sans', size=22, weight='bold')



j = 0
bases = [5,10,20,30,40,60,100,120,140,180,200,300]
t = np.zeros(len(bases),dtype=np.float_)
for i in bases:
    t[j] = np.average(np.load('simulation/reduced/full_trajectory/fixed_pod/deim_'+ str(i) +'/sp0099ts2879N.petsc_timeings.npy'))
    j += 1
    
j = 0
bases = [5,20,50,80,100,120,140,160,180,200,250,300]
t_deim = np.zeros(len(bases),dtype=np.float_)
for i in bases:
    t_deim[j] = np.average(np.load('simulation/reduced/full_trajectory/fixed_deim/pod_'+ str(i) +'/sp0099ts2879N.petsc_timeings.npy'))
    j += 1

t_hd = 200
p = plt.subplot(111)
plt.plot(bases,t_deim/t_hd , '--',color='black', marker='o',markersize=15)
plt.plot([5,10,20,30,40,60,100,120,140,180,200,300],t/t_hd, '--b',color='gray' ,marker='D' ,markersize=15)
plt.plot(bases,np.ones_like(bases),'--',color='black')
plt.yscale('log')
p.set_ylim([-0.5,1.2])
p.set_xlim([0,301])
#p.set_title(r"\textbf{Average scaled CPU time for one model year}",**axis_font)
plt.legend(["Fixed DEIM 150","Fixed POD 150","FOM"],loc='best')
plt.xlabel(r"\textbf{Size of other base}",**axis_font)
plt.ylabel(r"\textbf{CPU time in sec}", **axis_font)


plt.savefig('avarage_cpu_time.png', bbox_inches='tight')
plt.show()