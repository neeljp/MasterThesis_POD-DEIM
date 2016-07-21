import numpy as np
import model
import petsc_io as io
import pod
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=20)


lsm = io.read_PETSc_mat("data/TMM/2.8/Geometry/landSeaMask.petsc")
offset = 0
J = [] 
for ix in range(64):
    for iy in range(128):
        length = lsm[ix, iy]
        if not length == 0:
            #chage lenght of J to 52749
            #for i in range(length.astype(int)):
            #J.append(np.arange(offset,offset+length))
            J.append(offset)
            offset = offset + length
J.append(52749)
J = np.array(J,dtype=(int))



y1 = np.ones(52749,dtype=np.float64) * 2.17
y2 = np.ones(52749,dtype=np.float64) * 1e-4
y = np.zeros((52749,2),dtype=np.float64)
y[:,0] = y1
y[:,1] = y2

fice = np.zeros((4448,12),dtype=np.float64)
for i in range(12):
    fice[:,i] = io.read_PETSc_vec('data/TMM/2.8/Forcing/BoundaryCondition/fice_' + str(i).zfill(2)+ '.petsc')
    
lat = io.read_PETSc_vec('data/TMM/2.8/Forcing/BoundaryCondition/latitude.petsc')
dz = io.read_PETSc_vec('data/TMM/2.8/Forcing/DomainCondition/dz.petsc')
z = io.read_PETSc_vec('data/TMM/2.8/Forcing/DomainCondition/z.petsc')
lsm = io.read_PETSc_mat("data/TMM/2.8/Geometry/landSeaMask.petsc")

[a,b,j,k] = pod.linearinterpolation(2880,12)
bc = np.zeros(2,dtype=np.float64)
dc = np.zeros((52749,2),dtype=np.float64)
dc[:,0] = z
dc[:,1] = dz
bundtest = np.zeros(3,dtype=np.float64)
#print(lat[0],lat[4447])
#print(fice[:,0])

#check if q is zero in fortran routine 
q = np.zeros((52749,2),dtype=np.float64)
u = np.array([0.02,2.0,0.5,30.0,0.67,0.5,0.858],dtype=np.float64)
dt = 0.0003472222222222
t = 0
for spin in range(1):
        for s in range(2880):
            t = np.fmod(0 + s*dt, 1.0);
            
            for i in range(4448):
                   
                    bc[0] = lat[i]
                    bc[1] =a[s]*fice[i,j[s]] + b[s]*fice[i,k[s]]
                    #bundtest[0] = a[s]*fice[i,j[s]]
                    #bundtest[1] = b[s]*fice[i,k[s]]
                    #bundtest[2] = a[s]*fice[i,j[s]] + b[s]*fice[i,k[s]]
  
                    #print('profile: ', i ,t)
                   # if(i == 13):
                        #print("phi: ", bc,s, a[s],b[s],bundtest)
                    

                    q[J[i]:J[i+1],:] = model.metos3dbgc(dt,t,y[J[i]:J[i+1],:],u,bc,dc[J[i]:J[i+1],:])
                    #print(j[x],k[x],a[x],b[x])
            # if(s == 2879 ):
            v1 = io.read_PETSc_vec('simulation/DEIM/sp' + str(spin).zfill(4) + 'ts'+  str(s).zfill(4) + 'N.petsc')
            #io.write_PETSc_vec(y[:,0], 'simulation/compare/' + '/sp'+str(spin).zfill(4)+'ts'+str(s).zfill(4)+'N_python.petsc')
            v2 = io.read_PETSc_vec('simulation/DEIM/sp' + str(spin).zfill(4) + 'ts' + str(s).zfill(4)+ 'DOP.petsc')
            #io.write_PETSc_vec(y[:,1], 'simulation/compare/' + '/sp'+str(spin).zfill(4)+'ts'+str(s).zfill(4)+'DOP_python.petsc')
            print("spin: ", spin,"step: ", s,"t:", t,"norm: ", str(s).zfill(4) ,np.linalg.norm(y[:,0]-v1), np.linalg.norm(y[:,1]-v2))
            y = y + q