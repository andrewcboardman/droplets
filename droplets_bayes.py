import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infile', type=str, action='store', dest='infile',	help='Name of input file')
parser.add_argument('-pf', '--params_file',type=str, action='store', dest='params_file',
	help='Name of file containing parameter values')
parser.add_argument('-s', '--likelihood_noise',type=float,action='store',dest='std',default=0.01,help='Standard deviation assumed in calculation of the likelihood function')
parser.add_argument('-of', '--outfile', type=str, action= 'store', dest = 'outfile',
	help='Name of file to pickle outputs to')
parser.add_argument('-sc', '--skip-calc',action='store_true',dest='skip_calc')
args = parser.parse_args()
if not args.skip_calc:
	# Import droplet data
	import droplets_read
	data = droplets_read.unpack(args.infile)
	# Define parameter space
	import read_params
	params = read_params.read_params(args.params_file)

	# Calculate likelihood values over the parameter space
	import droplets_likelihood
	rss = droplets_likelihood.rss(data,params,args.std)

	import matplotlib.pyplot as plt

	import pickle
	with open('{}.pkl'.format(args.outfile),'wb') as pickle_file:
		pickle.dump((data,params,rss),pickle_file)

if args.skip_calc:
	import pickle
	with open('{}.pkl'.format(args.outfile),'rb') as pickle_file:
		pickle_data = pickle.load(pickle_file)
		data,params,rss = pickle_data
import matplotlib.pyplot as plt
import numpy as np

scaled_rss = np.squeeze(rss/(2*args.std**2))
print(np.min(scaled_rss),np.median(scaled_rss),np.max(scaled_rss))
likelihood_per_run = np.exp(-scaled_rss)
print(np.min(likelihood_per_run),np.max(likelihood_per_run))
fig = plt.figure(figsize=(15,5))

ax1 = fig.add_subplot(131,aspect='auto')

#map1 = ax1.imshow(np.sum(-scaled_rss+np.min(scaled_rss),axis=-1))
ax1.set_title('Heatmap of the -ve log likelihood')
map1 = ax1.imshow(-scaled_rss[...,0],interpolation='none',vmin=-10000)
ax1.set_ylabel(r'log $k_{+}k_{n}$')
ax1.set_xlabel(r'log $k_{+}k_{2}$')
plt.colorbar(map1)

ax2 = fig.add_subplot(132,aspect='auto')
ax2.plot(params[1],np.sum(-scaled_rss,axis=(1,2)))
#ax2.plot(params[1],np.exp(np.sum(-scaled_rss/10**6,axis=(1,2))))
ax2.set_xlabel(r'log $k_{+}k_{n}$')
ax2.set_ylabel(r'-log p')

ax3 = fig.add_subplot(133,aspect='auto')
ax3.plot(params[2],np.sum(-scaled_rss,axis=(0,2)))
ax3.set_xlabel(r'log $k_{+}k_{2}$')
ax3.set_ylabel(r'-log p')
# -data['n_t']*np.log(2*np.pi*args.std**2)/2)),axis=-1)))

plt.show()