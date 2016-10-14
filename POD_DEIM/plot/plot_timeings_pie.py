import numpy as np
import util
import petsc_io as io
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams.update({'font.weight': 'bold'})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'figure.figsize': (11, 10)})
axis_font = {'fontname':'Bitstream Vera Sans', 'weight':'bold' ,'size':'22'}
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rc('font', family='Bitstream Vera Sans', size=22, weight='bold')

labels = ["BGCStep", "Interpolation","Multiplication" ]
colors = ['gray', 'dimgray', 'lightgray']


spin = 10
bgc = np.zeros(spin)
#bgcStep = np.zeros(spin)
mult = np.zeros(spin)
interpolation = np.zeros(spin)
for i in range(1,spin):
    bgc[i] = np.average(np.load('simulation/reduced/full_trajectory/3_snapshots_per_month/POD_100_DEIM_50/sp'+ str(i).zfill(4) +'ts2879N.petsc_bgc_timeings.npy'))
    #bgcStep[i] = np.average(np.load('simulation/reduced/full_trajectory/POD_100_DEIM_50/sp'+ str(i).zfill(4) +'ts2879N.petsc_bgcStep_timeings.npy'))
    mult[i] = np.average(np.load('simulation/reduced/full_trajectory/3_snapshots_per_month/POD_100_DEIM_50/sp'+ str(i).zfill(4) + 'ts2879N.petsc_mult_timeings.npy'))
    interpolation[i] = np.average(np.load('simulation/reduced/full_trajectory/3_snapshots_per_month/POD_100_DEIM_50/sp' + str(i).zfill(4)+ 'ts2879N.petsc_interpolation_timeings.npy'))    




fracs = [np.floor(np.average(bgc)*100000),np.floor(np.average(interpolation)*100000),np.floor(np.average(mult)*200000)]

print(fracs)


plt.pie(fracs, colors=colors ,autopct='%1.1f%%',startangle=90, labels=labels)
                # The default startangle is 0, which would start
                # the Frogs slice on the x-axis.  With startangle=90,
                # everything is rotated counter-clockwise by 90 degrees,
                # so the plotting starts on the positive y-axis.
plt.savefig('timings_P100.png', bbox_inches='tight')
plt.show()

for i in range(spin):
    bgc[i] = np.average(np.load('simulation/reduced/full_trajectory/3_snapshots_per_month/sp'+ str(i).zfill(4) +'ts2879N.petsc_bgc_timeings.npy'))
    #bgcStep[i] = np.average(np.load('simulation/reduced/full_trajectory/POD_100_DEIM_50/sp'+ str(i).zfill(4) +'ts2879N.petsc_bgcStep_timeings.npy'))
    mult[i] = np.average(np.load('simulation/reduced/full_trajectory/3_snapshots_per_month/sp'+ str(i).zfill(4) + 'ts2879N.petsc_mult_timeings.npy'))
    interpolation[i] = np.average(np.load('simulation/reduced/full_trajectory/3_snapshots_per_month/sp' + str(i).zfill(4)+ 'ts2879N.petsc_interpolation_timeings.npy'))   

fracs = [np.floor(np.average(bgc)*100000),np.floor(np.average(interpolation)*100000),np.floor(np.average(mult)*200000)]
print(fracs)


plt.pie(fracs, colors=colors ,autopct='%1.1f%%',startangle=90, labels=labels)
                # The default startangle is 0, which would start
                # the Frogs slice on the x-axis.  With startangle=90,
                # everything is rotated counter-clockwise by 90 degrees,
                # so the plotting starts on the positive y-axis.

#plt.pie((np.average(bgc),np.average(mult),np.average(interpolation)))
plt.savefig('timings_P300.png', bbox_inches='tight')
plt.show()


for i in range(spin):
    bgc[i] = np.average(np.load('simulation/reduced/N-DOP/1000/POD100DEIM50/sp'+ str(i).zfill(4) +'ts2879N.petsc_bgc_timeings.npy'))
    #bgcStep[i] = np.average(np.load('simulation/reduced/full_trajectory/POD_100_DEIM_50/sp'+ str(i).zfill(4) +'ts2879N.petsc_bgcStep_timeings.npy'))
    mult[i] = np.average(np.load('simulation/reduced/N-DOP/1000/POD100DEIM50/sp'+ str(i).zfill(4) + 'ts2879N.petsc_mult_timeings.npy'))
    interpolation[i] = np.average(np.load('simulation/reduced/N-DOP/1000/POD100DEIM50/sp' + str(i).zfill(4)+ 'ts2879N.petsc_interpolation_timeings.npy'))   

fracs = [np.floor(np.average(bgc)*100000),np.floor(np.average(interpolation)*100000),np.floor(np.average(mult)*400000)]

print(fracs)


plt.pie(fracs, colors=colors ,autopct='%1.1f%%',startangle=90, labels=labels)
                # The default startangle is 0, which would start
                # the Frogs slice on the x-axis.  With startangle=90,
                # everything is rotated counter-clockwise by 90 degrees,
                # so the plotting starts on the positive y-axis.

#plt.pie((np.average(bgc),np.average(mult),np.average(interpolation)))
plt.savefig('timings_P100NDOP.png', bbox_inches='tight')
plt.show()

for i in range(spin):
    bgc[i] = np.average(np.load('simulation/reduced/N-DOP/1000/POD300DEIM300/sp'+ str(i).zfill(4) +'ts2879N.petsc_bgc_timeings.npy'))
    #bgcStep[i] = np.average(np.load('simulation/reduced/full_trajectory/POD_100_DEIM_50/sp'+ str(i).zfill(4) +'ts2879N.petsc_bgcStep_timeings.npy'))
    mult[i] = np.average(np.load('simulation/reduced/N-DOP/1000/POD300DEIM300/sp'+ str(i).zfill(4) + 'ts2879N.petsc_mult_timeings.npy'))
    interpolation[i] = np.average(np.load('simulation/reduced/N-DOP/1000/POD300DEIM300/sp' + str(i).zfill(4)+ 'ts2879N.petsc_interpolation_timeings.npy'))   


fracs = [np.floor(np.average(bgc)*100000),np.floor(np.average(interpolation)*100000),np.floor(np.average(mult)*400000)]

print(fracs)


plt.pie(fracs, colors=colors ,autopct='%1.1f%%',startangle=90, labels=labels)
                # The default startangle is 0, which would start
                # the Frogs slice on the x-axis.  With startangle=90,
                # everything is rotated counter-clockwise by 90 degrees,
                # so the plotting starts on the positive y-axis.

plt.savefig('timings_P300NDOP.png', bbox_inches='tight')
#plt.pie((np.average(bgc),np.average(mult),np.average(interpolation)))
plt.show()