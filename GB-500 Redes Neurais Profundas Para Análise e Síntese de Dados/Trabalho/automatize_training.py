import sys

test = '3-3'
data_split = 3

for ds in range(data_split, data_split+1):
	file_name = open('main.py')
	file = file_name.read()
	sys.argv = ['file.py', 'network=vgg13', 'test={}'.format(test), 'data_split={}'.format(ds)]
	exec(file)
	file_name.close()

	file_name = open('visualize.py')
	file = file_name.read()
	sys.argv = ['file.py', 'network=vgg13', 'test={}'.format(test), 'data_split={}'.format(ds)]
	exec(file)
	file_name.close()

	file_name = open('plot_results.py')
	file = file_name.read()
	sys.argv = ['file.py', 'network=vgg13', 'test={}'.format(test), 'data_split={}'.format(ds)]
	exec(file)
	file_name.close()