import numpy as np
import pandas as pd

def unpack(filename):
	data = pd.read_csv(filename)
	t = data.columns.values[1:-2].astype('float64')
	M_obs = data.values[:,1:-2].astype('float64').T
	n_t = len(t)
	m0 = float(filename[11:-4])/10**6
	N_exp = M_obs.shape[1]
	return {'t':t,'m0':m0,'M_obs':M_obs,'n_t':n_t,'N_exp':N_exp}
