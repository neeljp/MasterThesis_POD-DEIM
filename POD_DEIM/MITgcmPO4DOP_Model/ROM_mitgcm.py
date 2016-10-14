import numpy as np
import numpy.linalg as la
import scipy as sp
from util import petsc_io as io
import mitgcm
from util import util
import time
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
                self.U_PODN        = np.load(self.config["-Metos3DMatrixInputDirectory"][0] +'PO4/'+ self.config["-Metos3DMatrixPODFileFormat"][0])
                self.U_PODDOP       = np.load(self.config["-Metos3DMatrixInputDirectory"][0] +'DOP/'+ self.config["-Metos3DMatrixPODFileFormat"][0])
                self.U_DEIMN       = np.load(self.config["-Metos3DMatrixInputDirectory"][0] +'PO4/'+ self.config["-Metos3DMatrixDEIMFileFormat"][0])
                self.U_DEIMDOP       = np.load(self.config["-Metos3DMatrixInputDirectory"][0] +'DOP/'+ self.config["-Metos3DMatrixDEIMFileFormat"][0])
                self.DEIM_IndicesN = np.load(self.config["-Metos3DMatrixInputDirectory"][0] +'PO4/'+ self.config["-Metos3DDEIMIndicesFileFormat"][0])
                self.DEIM_IndicesDOP = np.load(self.config["-Metos3DMatrixInputDirectory"][0] +'DOP/'+ self.config["-Metos3DDEIMIndicesFileFormat"][0])

                

                self.AN = np.ndarray(shape=(self.matrixCount,self.U_PODN.shape[1],self.U_PODN.shape[1]), dtype=np.float_, order='C')
                self.ADOP = np.ndarray(shape=(self.matrixCount,self.U_PODDOP.shape[1],self.U_PODDOP.shape[1]), dtype=np.float_, order='C')

                for i in range(0,self.matrixCount):
                        self.AN[i] = np.load(self.config["-Metos3DMatrixInputDirectory"][0] +'PO4/'+ self.config["-Metos3DMatrixReducedFileFormat"][0] % i)
                        self.ADOP[i] = np.load(self.config["-Metos3DMatrixInputDirectory"][0] +'DOP/'+ self.config["-Metos3DMatrixReducedFileFormat"][0] % i)
        
                self.PN = np.ndarray(shape=(self.matrixCount,self.U_PODN.shape[1],self.U_DEIMN.shape[1]), dtype=np.float_, order='C')
                self.PDOP = np.ndarray(shape=(self.matrixCount,self.U_PODDOP.shape[1],self.U_DEIMDOP.shape[1]), dtype=np.float_, order='C')
                for i in range(0,self.matrixCount):
                        self.PN[i] = np.load(self.config["-Metos3DMatrixInputDirectory"][0] +'PO4/'+ self.config["-Metos3DMatrixReducedDEINFileFormat"][0] % i)
                        self.PDOP[i] = np.load(self.config["-Metos3DMatrixInputDirectory"][0] +'DOP/'+ self.config["-Metos3DMatrixReducedDEINFileFormat"][0] % i)

                #precomputin the interplaton indices for a year         
                [self.interpolation_a,self.interpolation_b,self.interpolation_j,self.interpolation_k] = util.linearinterpolation(2880,12,0.0003472222222222)

                self.yN     = np.ones(self.ny,dtype=np.float_) * np.float_(self.config["-Metos3DTracerInitValue"])[0]
                self.yDOP     = np.ones(self.ny,dtype=np.float_) * np.float_(self.config["-Metos3DTracerInitValue"])[1]
                self.y_redN = np.dot(self.U_PODN.T,self.yN)
                self.y_redDOP = np.dot(self.U_PODDOP.T,self.yDOP)

                self.qN     = np.zeros(self.DEIM_IndicesN.shape[0],dtype=np.float_)
                self.qDOP     = np.zeros(self.DEIM_IndicesDOP.shape[0],dtype=np.float_)

                self.J,self.PJ = util.generateIndicesForNonlinearFunction(self.lsm,self.profiles,self.ny)

                self.out_pathN     = self.config["-Metos3DTracerOutputDirectory"][0] +self.config["-Metos3DSpinupMonitorFileFormatPrefix"][0] + self.config["-Metos3DSpinupMonitorFileFormatPrefix"][1] +self.config["-Metos3DTracerOutputFile"][0]
                self.out_pathDOP     = self.config["-Metos3DTracerOutputDirectory"][0] +self.config["-Metos3DSpinupMonitorFileFormatPrefix"][0] + self.config["-Metos3DSpinupMonitorFileFormatPrefix"][1] +self.config["-Metos3DTracerOutputFile"][1]
                self.monitor_path = self.config["-Metos3DTracerMointorDirectory"][0] +self.config["-Metos3DSpinupMonitorFileFormatPrefix"][0] + self.config["-Metos3DSpinupMonitorFileFormatPrefix"][1] +self.config["-Metos3DTracerOutputFile"][0]


        def __TimeStep(self,step):
                self.t = np.fmod(0 + step*self.dt, 1.0);
                counter = 0
                bgcTime = time.time()
                #bgcStepTime = 0
                for i,l in self.PJ[self.DEIM_IndicesN]:
                        index = np.arange(self.J[i],self.J[i+1])
                        #transpose for input in biomodel
                        y =  np.dot(self.U_PODN[index,:],self.y_redN).T
                        y = np.zeros((y.shape[0],2))
                        y[:,0] = np.dot(self.U_PODN[index,:],self.y_redN).T
                        y[:,1] = np.dot(self.U_PODDOP[index,:],self.y_redDOP).T

                        self.bc[0] = self.lat[i]
                        self.bc[1] = self.interpolation_a[step]*self.fice[i,self.interpolation_j[step]] + self.interpolation_b[step]*self.fice[i,self.interpolation_k[step]]
                        #print("bio: ", y, bc ,dc[J[i]:J[i+1],:],u)
                        #bgcStepTime_tmp = time.time()
                        self.qN[counter] = mitgcm.metos3dbgc(self.dt,self.t,y,self.u,self.bc,self.dc[self.J[i]:self.J[i+1],:])[l,0]
                        #bgcStepTime += time.time() - bgcStepTime_tmp
                        counter += 1
                counter = 0
                for i,l in self.PJ[self.DEIM_IndicesDOP]:
                        index = np.arange(self.J[i],self.J[i+1])
                        #transpose for input in biomodel
                        y =  np.dot(self.U_PODN[index,:],self.y_redN).T
                        y = np.zeros((y.shape[0],2))
                        y[:,0] = np.dot(self.U_PODN[index,:],self.y_redN).T
                        y[:,1] = np.dot(self.U_PODDOP[index,:],self.y_redDOP).T


                        self.bc[0] = self.lat[i]
                        self.bc[1] = self.interpolation_a[step]*self.fice[i,self.interpolation_j[step]] + self.interpolation_b[step]*self.fice[i,self.interpolation_k[step]]
                        #print("bio: ", y, bc ,dc[J[i]:J[i+1],:],u)
                        #bgcStepTime_tmp = time.time()
                        self.qDOP[counter] = mitgcm.metos3dbgc(self.dt,self.t,y,self.u,self.bc,self.dc[self.J[i]:self.J[i+1],:])[l,1]
                        #bgcStepTime += time.time() - bgcStepTime_tmp
                        counter += 1
                bgcTime = time.time()- bgcTime
                interpolationTime = time.time()

                P_interpolationN = self.interpolation_a[step]*self.PN[self.interpolation_j[step]] + self.interpolation_b[step]*self.PN[self.interpolation_k[step]]
                P_interpolationDOP = self.interpolation_a[step]*self.PDOP[self.interpolation_j[step]] + self.interpolation_b[step]*self.PDOP[self.interpolation_k[step]]
                A_interpolationN = self.interpolation_a[step]*self.AN[self.interpolation_j[step]] + self.interpolation_b[step]*self.AN[self.interpolation_k[step]]
                A_interpolationDOP = self.interpolation_a[step]*self.ADOP[self.interpolation_j[step]] + self.interpolation_b[step]*self.ADOP[self.interpolation_k[step]]
                P_block = sp.sparse.block_diag((P_interpolationN,P_interpolationDOP), format='csc', dtype=np.float_)
                A_block = sp.sparse.block_diag((A_interpolationN,A_interpolationDOP), format='csc', dtype=np.float_)

                interpolationTime = time.time() - interpolationTime
                multTime = time.time()
                q_red = P_block.dot(np.hstack((self.qN,self.qDOP)))
                y_red = A_block.dot(np.hstack((self.y_redN,self.y_redDOP))) + q_red 
                self.y_redN,self.y_redDOP = np.hsplit(y_red,2)
                multTime = time.time()- multTime
                return bgcTime,multTime,interpolationTime

        def simulate(self):
                starttime = time.time()
                timings = np.zeros(self.nspinup)
                bgcTime = np.zeros(self.ntimestep)
                #bgcStepTime = np.zeros(self.ntimestep)
                multTime = np.zeros(self.ntimestep)
                interpolationTime = np.zeros(self.ntimestep)

                for spin in range(self.nspinup):
                        for step in range(self.ntimestep):
                                if((step ==2879) & (step > 0)):

                                    N = self.U_PODN.dot(self.y_redN)
                                    DOP = self.U_PODDOP.dot(self.y_redDOP)

                                    #y_hig = io.read_PETSc_vec(self.monitor_path % (spin,step))
                                    io.write_PETSc_vec(N, self.out_pathN % (spin,step))
                                    io.write_PETSc_vec(DOP, self.out_pathDOP % (spin,step))
                                    print("time: ", time.time() - starttime ,"spin: ", spin,"step: ", step,"t:", self.t, "spinup norm: ", np.linalg.norm(N-self.yN) ,np.linalg.norm(DOP-self.yDOP))
                                    timings[spin] = time.time() - starttime
                                    starttime = time.time()
                                    self.yN = N 
                                    self.yDOP = DOP 
                                bgcTime[step],multTime[step],interpolationTime[step] = self.__TimeStep(step)
                                #self.__TimeStep(step)
                        np.save(self.out_pathN %(spin,step) + "_bgc_timeings.npy",bgcTime)
                        np.save(self.out_pathN %(spin,step) + "_timeings.npy",timings)
                        np.save(self.out_pathN %(spin,step) + "_mult_timeings.npy",multTime)
                        np.save(self.out_pathN %(spin,step) + "_interpolation_timeings.npy",interpolationTime)

                np.save(self.out_pathN %(spin,step) + "_timeings.npy",timings)

