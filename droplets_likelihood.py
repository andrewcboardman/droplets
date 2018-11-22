import droplets_model
import numpy as np
def rss(data,params,std):
	# create parameter-time grids which specify every input point
	param_grids = np.meshgrid(*params,data['t'],indexing='ij')
	# Calculate M_obs for each point in parameter-time space
	pred = droplets_model.calculate(*param_grids,data['m0'])
	# Calculate RSS for each run
	rss = np.zeros((*param_grids[0].shape[:-1],data['N_exp']))

	for i in range(data['N_exp']):
		obs = data['M_obs'][:,i]
		errors = pred - obs
		rss[...,i] = np.sum(errors**2,axis=-1)
	return rss


