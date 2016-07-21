def generate_prameter_samples(n):
        ''' generates Latin-Hypercubesamples with pyDOE and
        transforms bounds.
        n: number of samples
        return: 2d-array of parameter samples (n,7)
        '''
        import pyDOE
        import numpy as np
        lh = pyDOE.lhs(7, samples=n)
        #upper and lower bound of parameters un MIT-gcm-PO4-DOP
        b=np.array([0.75,200,0.95,1.5,50,0.05,1.5])
        a=np.array([0.25,1.5,0.05,0.25,10,0.01,0.7])
        lh = lh*(b-a)+a
        return lh