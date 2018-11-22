import numpy as np

def values(line):
	""" Reads the contents of a line either as a single fixed parameter, or as a range of values which the parameter can take"""
	contents = line.split()
	if len(contents) == 1:
		return np.array(contents,dtype=float)
	elif len(contents) == 3:
		return np.linspace(float(contents[0]),float(contents[1]),int(contents[2]))
	else:
		print("There's something wrong with your parameter input file")



def read_params(filename):
	"""Takes the contents of a text file and outputs a list of parameter values"""
	params = []
	with open(filename,'r') as file:
		lines = file.readlines()
		for nline in range(0,len(lines),2):
			params.append(values(lines[nline+1]))
	return params




