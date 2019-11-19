import numpy as np
import re
from matplotlib import pyplot

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
	average = np.zeros((64, 64))
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

def relu(x):
	return max(0, x)

def transpose_convolve(input, filter, stride, bias):
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
	return convolve(input_aug, filter, 1, bias)
	#C = convolution_matrix(in_w, stride*(in_w-1) + f_w, filter, stride, transpose=True)
	#return C.dot(input.reshape((-1, 1))).reshape((out_w, out_w))
	
def convolve(input, filter, stride, bias):
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
			sum = 0
			for p in range(0, f_w):
				for q in range(0, f_h):
					sum += input[i + p][j + q]*filter[p][q]
			output[a][b] = sum + bias[a][b]
			b += 1
			j += stride
		a += 1
		i += stride
	#C = convolution_matrix(in_w, int((in_w - f_w)/stride)+1, filter, stride)
	#return C.dot(input.reshape((-1, 1))).reshape((out_w, out_w))
	return output

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