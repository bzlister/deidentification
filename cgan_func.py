import numpy as np
import re
from matplotlib import pyplot
import cv2
import random
import math
import os

def read_pgm(filename, byteorder='>'):
	with open(filename, 'rb') as f:
		buffer = f.read()
	try:
		header, maxval = re.search(
			b"(^P5\s(?:\s*#.*[\r\n])*"
			b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
	except AttributeError:
		raise ValueError("Not a raw PGM file: '%s'" % filename)
	offset=0 if len(header) != 13 else 13
	while (True):
		try:
			np.frombuffer(buffer, dtype='u1' if int(maxval) < 256 else byteorder+'u2',count=4096,offset=offset)
			offset += 1
		except ValueError:
			offset-=1
			break
	image = np.frombuffer(buffer, dtype='u1' if int(maxval) < 256 else byteorder+'u2',count=4096,offset=offset)
	return image.reshape((64, 64))

def batchnorm(X, beta=0, gamma=1):
	w,h = X[0].shape
	average = np.zeros((w, h))
	for x in X:
		average += x
	average/=len(X)
	var = 0
	for x in X:
		var += (x - average)**2
	var/=len(X)
	Y = []
	for x in X:
		norm_x = (x - average)/(var + 1e-8)**0.5
		Y.append(gamma*norm_x + beta)
	return Y

def standard_activation(x):
    return x

def leaky_relu(x):
    if (x > 0):
        return x
    else:
        return 0.01*x

def relu(x):
	if (x > 0):
		return x
	else:
		return 0

def transpose_convolve(input, filter, stride, bias, activation):
	in_w, in_h = input.shape
	f_w, f_h = filter.shape
	output = np.zeros((stride*(in_w-1) + f_w, stride*(in_h-1) + f_h))
	assert(in_w == in_h) #For now, input matrix must be square
	d_aug = in_w + (stride-1)*(in_w-1) + 2*f_w - 2
	input_aug = np.zeros((d_aug, d_aug))
	p = f_h-1
	for a in range(0, in_w):
		q = f_w-1
		for b in range(0, in_h):
			input_aug[a + p][b + q] = input[a][b]
			q += 1
		p += 1
	return convolve(input_aug, filter, 1, bias, activation)
	#C = convolution_matrix(in_w, stride*(in_w-1) + f_w, filter, stride, transpose=True)
	#return C.dot(input.reshape((-1, 1))).reshape((out_w, out_w))
	
def convolve(input, filter, stride, bias, activation):
	in_w, in_h = input.shape
	f_w, f_h = filter.shape
	out_w = (in_w - f_w)//stride + 1	
	output = np.zeros(((in_w - f_w)//stride + 1, (in_h - f_h)//stride + 1))
	a = 0
	i = 0
	while (i <= in_w - f_w):
		j = 0
		b = 0
		while (j <= in_h - f_h):
			sum = bias
			for p in range(0, f_w):
				for q in range(0, f_h):
					sum += input[i + p][j + q]*filter[p][q]
			output[a][b] = activation(sum)
			b += 1
			j += stride
		a += 1
		i += stride
	#C = convolution_matrix(in_w, int((in_w - f_w)/stride)+1, filter, stride)
	#return C.dot(input.reshape((-1, 1))).reshape((out_w, out_w))
	return output

def convolve2(input, kernel, s, bias, activation):
	k_w, k_h = kernel.shape
	i_w, i_h = input.shape
	assert(i_w == i_h and k_w == k_h) #For now, only square inputs and convolutions
	d = (i_w - k_w)//s + 1
	out = np.zeros((d, d))
	for i in range(0, d):
		for j in range(0, d):
			sum = bias
			for k in range(0, k_w):
				for l in range(0, k_h):
					sum += kernel[k][l]*input[2*i+k][2*j+l]
			out[i][j] = activation(sum)
	return out
	

def resize(image):
    return cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    
def view(image):
	pyplot.imshow(image, pyplot.cm.gray)
	pyplot.show()

def example():
	image = read_pgm('lfwcrop_grey/faces/Aaron_Eckhart_0001.pgm')
	print(image.shape)
	for i in range(0, 4):
		image = convolve(image, np.ones((4, 4)), 2, np.zeros(image.shape))
		print(image.shape)
	for j in range(0, 4):
		in_w, in_h = image.shape
		bias = np.zeros((2*(in_w-1) + 4, 2*(in_h-1) + 4)) #stride*(input_dim-1) + kernel_dim
		image = transpose_convolve(image, np.ones((4, 4)), 2, bias)	
		print(image.shape)

def explore():
	image = read_pgm('lfwcrop_grey/faces/Aaron_Eckhart_0001.pgm')
	filter = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
	convolutions = []
	inputs = []
	outputs = []
	for a in range(0, 4):
		in_w, in_h = image.shape
		output = convolve(image, filter, 2, np.zeros(image.shape))
		out_w, out_h = output.shape
		i = image.reshape((in_w*in_h, 1))
		o = output.reshape((out_w*out_h, 1))
		convolutions.append(getC(i,o))
		inputs.append(i)
		outputs.append(o)
	return convolutions, inputs, outputs

#Optimization for convolutional layers. Fails for deconvolutional
def convolution_matrix(i_dim, o_dim, filter, stride, transpose=False):
	C = np.zeros((i_dim**2, o_dim**2)) if transpose else np.zeros((o_dim**2, i_dim**2))
	pad = 0
	count = 0
	for i in range(0, o_dim**2):
		offset = 0
		for a in range(0, len(filter)):
			for b in range(0, len(filter)):
				C[i][b + pad + offset] = filter[a][b]
			offset += i_dim		
		count+=1
		pad += stride
		if (count == o_dim):
			pad += len(filter) + i_dim - stride
			count = 0
	if (transpose):
		C = C.T
	return C

#Generator
layers = []
#Convolutional
layers.append({'func': convolve, 'kernel': np.random.rand(4,4), 'bias': random.uniform(0,1), 'activation': standard_activation, 'dropout': 0, 'partials_w': np.zeros((4,4)), 'partial_b': 0})
layers.append({'func': convolve, 'kernel': np.random.rand(4,4), 'bias': random.uniform(0,1), 'batchnorm': [random.uniform(0,1), random.uniform(0,1)], 'activation': leaky_relu, 'dropout': 0, 'partials_w': np.zeros((4,4)), 'partial_b': 0})
layers.append({'func': convolve, 'kernel': np.random.rand(4,4), 'bias': random.uniform(0,1), 'batchnorm': [random.uniform(0,1), random.uniform(0,1)], 'activation': leaky_relu, 'dropout': 0, 'partials_w': np.zeros((4,4)), 'partial_b': 0})
layers.append({'func': convolve, 'kernel': np.random.rand(4,4), 'bias': random.uniform(0,1), 'activation': leaky_relu, 'dropout': 0, 'partials_w': np.zeros((4,4)), 'partial_b': 0})
#Deconvolutional
layers.append({'func': transpose_convolve, 'kernel': np.random.rand(4,4), 'bias': random.uniform(0,1), 'batchnorm': [random.uniform(0,1), random.uniform(0,1)], 'activation': relu, 'dropout':random.randint(0,1), 'partials_w': np.zeros((4,4)), 'partial_b': 0})
layers.append({'func': transpose_convolve, 'kernel': np.random.rand(4,4), 'bias': random.uniform(0,1), 'batchnorm': [random.uniform(0,1), random.uniform(0,1)], 'activation': relu, 'dropout':random.randint(0,1), 'partials_w': np.zeros((4,4)), 'partial_b': 0})
layers.append({'func': transpose_convolve, 'kernel': np.random.rand(4,4), 'bias': random.uniform(0,1), 'batchnorm': [random.uniform(0,1), random.uniform(0,1)], 'activation': relu, 'dropout': 0, 'partials_w': np.zeros((4,4)), 'partial_b': 0})
layers.append({'func': transpose_convolve, 'kernel': np.random.rand(4,4), 'bias': random.uniform(0,1), 'activation': relu, 'dropout': 0, 'partial_b': 0})

#Hyperparameters
lambda1 = 0.5
lambda2 = 0.5
alpha = 0.000001

def generator_pass(x, skip=0):
	input = x
	for i in range(skip, len(layers)):
		layer = layers[i]
		if (layer['dropout'] == 0):
			output = layer['func'](input, layer['kernel'], 2, layer['bias'], layer['activation'])
			#Need to figure out how to incorporate batchnorm
			if (i == 1 or i == 2):
				generator_pass(output, skip=i+3)
			input = output
		else:
			layer['dropout'] = random.randint(0,1)
		if (i == len(layers)-1):
			print(output)
    #Error evaluation and backpropogation

#Discriminator
def discriminator():
	discrim = []
	discrim.append({'func': convolve, 'stride': 2, 'kernel': 2*np.random.rand(4,4)-1, 'bias': random.uniform(-1,1), 'activation': standard_activation, 'partials_w': np.zeros((4,4)), 'partial_b': 0})
	discrim.append({'func': convolve, 'stride': 2, 'kernel': 2*np.random.rand(4,4)-1, 'bias': random.uniform(-1,1), 'batchnorm': [random.uniform(0,1), random.uniform(0,1)], 'activation': leaky_relu, 'partials_w': np.zeros((4,4)), 'partial_b': 0})
	discrim.append({'func': convolve, 'stride': 1, 'kernel': 2*np.random.rand(4,4)-1, 'bias': random.uniform(-1,1), 'batchnorm': [random.uniform(0,1), random.uniform(0,1)], 'activation': leaky_relu, 'partials_w': np.zeros((4,4)), 'partial_b': 0})
	discrim.append({'func': convolve, 'stride': 1, 'kernel': 2*np.random.rand(4,4)-1, 'bias': random.uniform(-1,1), 'activation': leaky_relu, 'partials_w': np.zeros((4,4)), 'partial_b': 0})
	X = load_dataset()
	for x in X[:10]:
		out = discriminator_pass(x, 1, discrim)
		out2 = discriminator_pass(255*np.random.rand(64,64), 0, discrim)
		print(out, out2)
	return discrim

def discriminator_pass(x, real, discrim):
	input = x
	for l in range(0, len(discrim)):
		discrim[l]['input'] = input
		output = discrim[l]['func'](input, discrim[l]['kernel'], discrim[l]['stride'], discrim[l]['bias'], discrim[l]['activation'])
		input = output
	output = [[1/(1+math.exp(-1*element)) for element in row] for row in output]
	#Backpropagation
	upper_partials = [[patch-real for patch in row] for row in output]
	print(upper_partials)
	l = len(discrim)-1
	while (l >= 0):
		mat = discrim[l]['input']
		kern = discrim[l]['kernel']
		partials = discrim[l]['partials_w']
		s = discrim[l]['stride']
		for i in range(0, len(upper_partials)):
			for j in range(0, len(upper_partials)):
				kernel_sum = 0
				for r in range(0, len(kern)):
					for c in range(0, len(kern)):
						kernel_sum += kern[r][c]*mat[s*i+r][s*j+c]
				for p in range(0, len(kern)):
					for q in range(0, len(kern)):
						partials[p][q] += upper_partials[i][j]*mat[s*i+p][s*j+q]*kernel_sum
		upper_partials = partials
		print('Layer %d' %(l))
		print(upper_partials)
		discrim[l]['partials_w'] = partials
		for a in range(0, len(kern)):
			for b in range(0, len(kern)):
				kern[a][b] -= alpha*partials[a][b]
		discrim[l]['kernel'] = kern
		l-=1
	votes = 0
	for i in range(0, len(output)):
		for j in range(0, len(output[j])):
			votes += output[i][j]
	return votes/64
	
#PLACEHOLDER! Actual loss is a function of the discriminator and generator
#This simply calculates distance from the average face of the dataset
def loss(x):
	avg = average_face(load_dataset)
	error = 0
	for i in range(0, 64):
		for j in range(0, 64):
			error += (avg[i][j]/255 - x[i][j]/255)**2	

def load_dataset():
	dataset = []
	holding = []	
	files = os.listdir('lfwcrop_grey/faces')
	for file in files:
		dataset.append(read_pgm('lfwcrop_grey/faces/' + file))
	return dataset

def average_face(dataset):
	average = np.zeros((64, 64))
	count = 0
	for face in dataset:
		if (count == 0):
			average += face
		else:
			average = (average + face/count)*(count/(count+1))
	return average


def simulate_convolvements():
	X = [['x'+str(i) + '_' + str(j) for j in range(0, 64)] for i in range(0, 64)]
	kernel = [['k' + str(k) + '_' + str(l) for l in range(0, 4)] for k in range(0, 4)]
	
	