
import numpy as np
from modules import model

def integrate(likelihood,params):
	"""Integrates the likelihood grid to find the normalisation factor"""
	# Find non-constant directions in the array which need to be integrated over
	# and the values of the parameters in those directions
	for ix in range(len(likelihood.shape)):
		if likelihood.shape[ix] > 1:
	# Integrate over non-constant dimensions
			likelihood = np.trapz(likelihood,params[ix],axis=ix)
	# Keep array shape the same by expanding dimensions
			likelihood = np.expand_dims(likelihood,ix)
		else:
			pass
	# Remove unnecessary dimensions and return a scalar
	return np.squeeze(likelihood)

def marginalise(posterior,params,marg_ix):
	"""Integrate the posterior over all but one variable to find the marginalised posterior probability"""
	for ix in range(len(posterior.shape)):
		# Don't integrate over the desired variable
		if ix == marg_ix:
			pass
		# Integrate over all others; keep dimensionality the same until the end 
		elif posterior.shape[ix] > 1:
			posterior = np.trapz(posterior,params[ix],axis=ix)
			posterior = np.expand_dims(posterior,ix)
		# Don't integrate over constants
		else:
			pass
		# Remove unnecesary dimensions at end to return 1D array
	return np.squeeze(posterior)





def old_likelihood(input_data,params,likelihood_noise):
	
	no_runs = input_data['conditions'].shape[1]
	no_times = input_data['times'].size
	# It would be wasteful to compute the predicted concs again for each repeat
	# Therefore work out how many unique sets of conditions there are
	# Store this in a dictionary which can be called later

	unique_conditions = np.unique(input_data['conditions'],axis=1).T
	predicted_data = {}
	for condition in unique_conditions:
		grids = np.meshgrid(*params,input_data['times'],indexing='ij')
		predicted_data[tuple(condition)] = model.calculate(*grids,condition=condition)

	# Now iterate through each run and calculate the sum of squares of residuals
	# Add the log likelihood from each run to the total log likelihood
	loglikelihood = 1
	for run in range(no_runs):
		condition = input_data['conditions'][:,run]
		sumresiduals = np.sum((predicted_data[tuple(condition)]-input_data['agg_concs'][:,run])**2,axis=5) 
		loglikelihood += -sumresiduals/(2*likelihood_noise**2)
	# Rescale and normalise likelihood in linear space
	loglikelihood = loglikelihood-np.max(loglikelihood)
	likelihood = np.exp(loglikelihood)
	likelihood = likelihood/integrate(likelihood,params)

	return likelihood
	

def new_likelihood(input_data,params,likelihood_noise):
	
	no_runs = input_data['conditions'].shape[1]
	no_times = input_data['times'].size
	# It would be wasteful to compute the predicted concs again for each repeat
	# Therefore work out how many unique sets of conditions there are, and where they are in the file
	unique_conditions = np.unique(input_data['conditions'],axis=1).T
	log_likelihood = 0
	uc_likelihoods = []
	for unique_condition in unique_conditions:
	# Store the predicted kinetic data for each unique condition (for all allowed values of the kinetic parameters) in a dictionary which can be called later
		unique_condition_likelihood = 0
		grids = np.meshgrid(*params,input_data['times'],indexing='ij')
		predicted_data = model.calculate(*grids,condition=unique_condition)
	# Now iterate through all the repeats sharing the same conditions and calculate the likelihood of the observed data for all values of the parameters
	# Add the likelihoods together for all similar runs
		same_condition_ix = np.all(input_data['conditions'].T == unique_condition,axis=1)
		same_condition_agg_concs = input_data['agg_concs'][:,same_condition_ix]
		for run in same_condition_agg_concs.T:
			#print(predicted_data.shape)
			sqres = np.sum((predicted_data-run)**2,axis=5) 
			unique_condition_likelihood += np.exp(-sqres/(2*likelihood_noise**2))

	# Now multiply the likelihoods for non-similar runs (by summing log-likelihoods)
	# This presents difficulties if some likelihoods have been rounded to zero (underflow)
	# Find the lowest non-zero value in the array, and set all zero values to this value
		unique_condition_likelihood = np.clip(unique_condition_likelihood,1e-300,None)
		#printnp.any(new_unique_condition_likelihood==unique_condition_likelihood)
		uc_likelihoods.append(np.log(unique_condition_likelihood))

		log_likelihood += np.log(unique_condition_likelihood)
		#likelihood += unique_condition_likelihood
	# Shift the log likelihood so that the maximum is at 0: this should prevent errors in normalisation
	log_likelihood = log_likelihood - np.max(log_likelihood)

	return  log_likelihood, uc_likelihoods

def likelihood3(input_data,params,likelihood_noise):
	
	no_runs = input_data['conditions'].shape[1]
	no_times = input_data['times'].size
	
	# It would be wasteful to compute the predicted concs again for each repeat
	# Therefore work out how many unique sets of conditions there are

	unique_conditions = np.unique(input_data['conditions'],axis=1).T

	# Find the positions of all these unique sets of conditions

	likelihood = 0
	param_grids = np.meshgrid(*params,indexing='ij')
	for condition in unique_conditions:
		condition_pos = np.all(condition == input_data['conditions'].T,axis=1)
		for repeat in range(np.sum(condition_pos)):
			sumresiduals = 0 
			agg_concs = input_data['agg_concs'][:,condition_pos]
			for time in range(len(input_data['times'])):
				predicted_datapoint = model.calculate(*param_grids,input_data['times'][time],condition=condition)
				observed_datapoint = agg_concs[time,repeat]
				sumresiduals += (predicted_datapoint-observed_datapoint)**2 
			run_likelihood = np.exp(-sumresiduals/(2*likelihood_noise**2))
	# Assuming the run likelihood has not been truncated by the prior we can normalise it by summing over k space
	# This is necessary in order to weight each run equally
			run_norm_factor = integrate(run_likelihood,params)
			likelihood += run_likelihood/run_norm_factor


	return likelihood/no_runs
def likelihood4(input_data,params,likelihood_noise):
	
	no_runs = input_data['conditions'].shape[1]
	no_times = input_data['times'].size
	
	# It would be wasteful to compute the predicted concs again for each repeat
	# Therefore work out how many unique sets of conditions there are

	unique_conditions = np.unique(input_data['conditions'],axis=1).T

	# Find the positions of all these unique sets of conditions

	likelihood = 0
	param_grids = np.meshgrid(*params,indexing='ij')
	for condition in unique_conditions:
		condition_pos = np.all(condition == input_data['conditions'].T,axis=1)
		sumresiduals = 0 
		for repeat in range(np.sum(condition_pos)):
			agg_concs = input_data['agg_concs'][:,condition_pos]
			for time in range(len(input_data['times'])):
				predicted_datapoint = model.calculate(*param_grids,input_data['times'][time],condition=condition)
				observed_datapoint = agg_concs[time,repeat]
				sumresiduals += (predicted_datapoint-observed_datapoint)**2 
		condition_likelihood = np.exp(-sumresiduals/(2*likelihood_noise**2))
	# Assuming the run likelihood has not been truncated by the prior we can normalise it by summing over k space
	# This is necessary in order to weight each run equally
		run_norm_factor = integrate(condition_likelihood,params)
		likelihood += condition_likelihood/run_norm_factor


	return likelihood/unique_conditions.shape[0]

def likelihood5(input_data,params,likelihood_noise):
	
	no_runs = input_data['conditions'].shape[1]
	no_times = input_data['times'].size
	
	# It would be wasteful to compute the predicted concs again for each repeat
	# Therefore work out how many unique sets of conditions there are

	unique_conditions = np.unique(input_data['conditions'],axis=1).T

	# Find the positions of all these unique sets of conditions

	sumresiduals = 0 
	param_grids = np.meshgrid(*params,indexing='ij')
	for condition in unique_conditions:
		condition_pos = np.all(condition == input_data['conditions'].T,axis=1)
		for repeat in range(np.sum(condition_pos)):
			agg_concs = input_data['agg_concs'][:,condition_pos]
			for time in range(len(input_data['times'])):
				predicted_datapoint = model.calculate(*param_grids,input_data['times'][time],condition=condition)
				observed_datapoint = agg_concs[time,repeat]
				sumresiduals += (predicted_datapoint-observed_datapoint)**2 
	likelihood = np.exp(-sumresiduals/(2*likelihood_noise**2))
	norm_factor = integrate(likelihood,params)


	return likelihood/norm_factor
