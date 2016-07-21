import numpy as np
import numpy.linalg as la
import petsc_io as io
import scipy as sp
import model
import util as util
#np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=20)


class ReducedModel:

        def __init__(self,config):
                self.config = util.parse_config(config)
                self.profiles = 4448
                self.ny = 52749


        def Init(self):
                #boundary and domain condition
                self.lat  = io.read_PETSc_vec(self.config["-Metos3DBoundaryConditionInputDirectory"][0] + self.config["-Metos3DLatitudeFileFormat"][0])
                dz        = io.read_PETSc_vec(self.config["-Metos3DDomainConditionInputDirectory"][0] + self.config["-Metos3DLayerHeightFileFormat"][0])
                z         = io.read_PETSc_vec(self.config["-Metos3DDomainConditionInputDirectory"][0] + self.config["-Metos3DLayerDepthFileFormat"][0])
                self.lsm  = io.read_PETSc_mat(self.config["-Metos3DProfileInputDirectory"][0] + self.config["-Metos3DProfileMaskFile"][0])
                self.fice = np.zeros((self.profiles,np.int_(self.config["-Metos3DIceCoverCount"][0])),dtype=np.float_)
                for i in range(np.int_(self.config["-Metos3DIceCoverCount"][0])):
                        self.fice[:,i] = io.read_PETSc_vec(self.config["-Metos3DBoundaryConditionInputDirectory"][0] + (self.config["-Metos3DIceCoverFileFormat"][0] % i))

                self.bc         = np.zeros(2,dtype=np.float_)
                self.dc         = np.zeros((self.ny,2),dtype=np.float_)
                self.dc[:,0]    = z
                self.dc[:,1]    = dz

                self.u          = np.array(self.config["-Metos3DParameterValue"],dtype=np.float_)
                self.dt         = np.float_(self.config["-Metos3DTimeStep"][0])
                self.nspinup    = np.int_(self.config["-Metos3DSpinupCount"][0])
                self.ntimestep  = np.int_(self.config["-Metos3DTimeStepCount"][0])


                self.matrixCount  = np.int_(self.config["-Metos3DMatrixCount"][0])
                self.U_POD        = np.load(self.config["-Metos3DMatrixInputDirectory"][0] + self.config["-Metos3DMatrixPODFileFormat"][0])
                self.U_DEIM       = np.load(self.config["-Metos3DMatrixInputDirectory"][0] + self.config["-Metos3DMatrixDEIMFileFormat"][0])
                self.DEIM_Indices = np.load(self.config["-Metos3DMatrixInputDirectory"][0] + self.config["-Metos3DDEIMIndicesFileFormat"][0])

                

                self.A = np.ndarray(shape=(self.matrixCount,self.U_POD.shape[1],self.U_POD.shape[1]), dtype=np.float_, order='C')
                for i in range(0,self.matrixCount):
                        self.A[i] = np.load(self.config["-Metos3DMatrixInputDirectory"][0] + self.config["-Metos3DMatrixReducedFileFormat"][0] % i)
        
                self.P = np.ndarray(shape=(self.matrixCount,self.U_POD.shape[1],self.U_DEIM.shape[1]), dtype=np.float_, order='C')
                for i in range(0,self.matrixCount):
                        self.P[i] = np.load(self.config["-Metos3DMatrixInputDirectory"][0] + self.config["-Metos3DMatrixReducedDEINFileFormat"][0] % i)

                #precomputin the interplaton indices for a year         
                [self.interpolation_a,self.interpolation_b,self.interpolation_j,self.interpolation_k] = util.linearinterpolation(2880,12,0.0003472222222222)

                self.y     = np.ones(self.ny,dtype=np.float_) * np.float_(self.config["-Metos3DTracerInitValue"])[0]
                self.y_red = np.dot(self.U_POD.T,self.y) 
                self.q     = np.zeros(self.DEIM_Indices.shape[0],dtype=np.float_)

                self.J,self.PJ = util.generateIndicesForNonlinearFunction(self.lsm,self.profiles,self.ny)

                self.out_path     = self.config["-Metos3DTracerOutputDirectory"][0] +self.config["-Metos3DSpinupMonitorFileFormatPrefix"][0] + self.config["-Metos3DSpinupMonitorFileFormatPrefix"][1] +self.config["-Metos3DTracerOutputFile"][0]
                self.monitor_path = self.config["-Metos3DTracerMointorDirectory"][0] +self.config["-Metos3DSpinupMonitorFileFormatPrefix"][0] + self.config["-Metos3DSpinupMonitorFileFormatPrefix"][1] +self.config["-Metos3DTracerOutputFile"][0]


        def __TimeStep(self,step):
                self.t = np.fmod(0 + step*self.dt, 1.0);
                counter = 0
                for i,l in self.PJ[self.DEIM_Indices]:
                        index = np.arange(self.J[i],self.J[i+1])
                        #transpose for input in biomodel
                        y = np.dot(self.U_POD[index,:],self.y_red).T
                        self.bc[0] = self.lat[i]
                        self.bc[1] = self.interpolation_a[step]*self.fice[i,self.interpolation_j[step]] + self.interpolation_b[step]*self.fice[i,self.interpolation_k[step]]
                        #print("bio: ", y, bc ,dc[J[i]:J[i+1],:],u)
                        self.q[counter] = model.metos3dbgc(self.dt,self.t,y,self.u,self.bc,self.dc[self.J[i]:self.J[i+1],:])[l]
                        counter += 1

                P_interpolation = self.interpolation_a[step]*self.P[self.interpolation_j[step]] + self.interpolation_b[step]*self.P[self.interpolation_k[step]]
                q_red = np.dot(P_interpolation,self.q)
                
                A_interpolation = self.interpolation_a[step]*self.A[self.interpolation_j[step]] + self.interpolation_b[step]*self.A[self.interpolation_k[step]]
                self.y_red = A_interpolation.dot(self.y_red) + q_red 


        def simulate(self):
                for spin in range(self.nspinup):
                        for step in range(self.ntimestep):
                                self.__TimeStep(step)
                                if(step % 240 == 239):
                                    y = self.U_POD.dot(self.y_red)
                                    y_hig = io.read_PETSc_vec(self.monitor_path % (spin,step))
                                    io.write_PETSc_vec(y, self.out_path % (spin,step))
                                    print("spin: ", spin,"step: ", step,"t:", self.t,"norm: ", np.linalg.norm(y-y_hig))




        def test(self,nspinup,ntimestep):
                y = np.ones(52749,dtype=np.float_) * 2.17
                #load high dim matrices
                Ae = []
                Ai = []
                for i in range(12):
                        Ai.append(io.read_PETSc_mat('data/TMM/2.8/Transport/Matrix5_4/1dt/Ai_'+str(i).zfill(2)+'.petsc'))
                        Ae.append(io.read_PETSc_mat('data/TMM/2.8/Transport/Matrix5_4/1dt/Ae_'+str(i).zfill(2)+'.petsc'))
                        
                #check if q is zero in fortran routine 
                q = np.zeros(52749,dtype=np.float_)
                t = 0
                #q_select = np.zeros(p.shape[0],dtype=np.float_)
                for spin in range(nspinup):
                        for step in range(ntimestep):
                                t = np.fmod(0 + step*self.dt, 1.0);
                                counter = 0
                                for i in range(4448):
                                        self.bc[0] = self.lat[i]
                                        self.bc[1] = self.interpolation_a[step]*self.fice[i,self.interpolation_j[step]] + self.interpolation_b[step]*self.fice[i,self.interpolation_k[step]]

                                        q[self.J[i]:self.J[i+1]] = model.metos3dbgc(self.dt,t,y[self.J[i]:self.J[i+1]],self.u,self.bc,self.dc[self.J[i]:self.J[i+1],:])[:,0]
                                        #print("q:", q[self.J[i]:self.J[i+1]])
                                

                                Aiint = self.interpolation_a[step]*Ai[self.interpolation_j[step]] + self.interpolation_b[step]*Ai[self.interpolation_k[step]]
                                Aeint = self.interpolation_a[step]*Ae[self.interpolation_j[step]] + self.interpolation_b[step]*Ae[self.interpolation_k[step]]    


                                v = io.read_PETSc_vec("simulation/compare/sp%.4dts%.4dN.petsc" % (spin,step+1))
                                Aey = io.read_PETSc_vec("simulation/compare/Aey_sp%.4dts%.4dN.petsc" % (spin,step))
                                Aeq = io.read_PETSc_vec("simulation/compare/Ae+q_sp%.4dts%.4dN.petsc" % (spin,step))
                                q_v = io.read_PETSc_vec("simulation/compare/q_sp%.4dts%.4dN.petsc" % (spin,step))
                                Aiint_metos = io.read_PETSc_mat("simulation/compare/A%.4d.petsc" % (step))
                                print("norm A interplaton: ", (Aiint-Aiint_metos))

                                ye = Aeint.dot(y)
                                yeq = ye +q 
                                io.write_PETSc_vec(yeq,"yeqts%.4dN.petsc" % step)
                                # A_saved = io.read_PETSc_mat("Ai_interpolatedts%.4d.petsc" % step)
                                y_j = Aiint.dot(yeq)
                                print("q:", np.linalg.norm(q_v-q))
                                print("before Ai:", np.linalg.norm(Aeq-yeq))
                                print("after Ai:",np.max(y_j-v))
                                io.write_PETSc_vec(y_j,"yts%.4dN.petsc" % step)
                                io.write_PETSc_mat(Aiint,"Ai%.4dN.petsc" % step)
                                # print(Aiint-A_saved.T)
                                y = y_j