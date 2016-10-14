import numpy as np
import numpy.linalg as la
import petsc_io as io
import model
import class_pod
import util
import cProfile
import time
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=20)

bases = [5,10,20,30,40,60,100,120,140,160,180,200,300]

for i in bases:
        starttime = time.time()
        print( "time: ", time.time() -starttime , "run model deim_%d " %i)
        n_model = class_pod.ReducedModel("reduced_basis/full_trajectory/fixed_pod/deim_" + str(i) + "/option.txt")
        n_model.Init()
        cProfile.run("n_model.simulate()","simulation/reduced/full_trajectory/fixed_pod/deim_" + str(i) + "/cProfile.txt") 
        print( "time: ", time.time() -starttime , "finished model deim_%d " %i)
