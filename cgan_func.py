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
	d_aug = 2*in_w + 2*f_w - 3
	input_aug = np.zeros((d_aug, d_aug))
	p = f_h-1
	for a in range(0, in_w):
		q = f_w-1
		for b in range(0, in_h):
			input_aug[a + p][b + q] = input[a][b]
			q += 1
		p += 1
	return convolve(input_aug, filter, 1, np.zeros((stride*(in_w-1) + f_w, stride*(in_h-1) + f_h)))
	
def convolve(input, filter, stride, bias):
	in_w, in_h = input.shape
	f_w, f_h = filter.shape
	output = np.zeros(((in_w - f_w)//stride + 1, (in_h - f_h)//stride + 1), dtype='uint8')
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
			output[a][b] = int(sum + bias[a][b])
			b += 1
			j += stride
		a += 1
		i += stride
	return output

def view(image):
	pyplot.imshow(image, pyplot.cm.gray)
	pyplot.show()

image = read_pgm('lfwcrop_grey/faces/Aaron_Eckhart_0001.pgm')
for i in range(0, 4):
	output = convolve(image, np.ones((4, 4),dtype='uint8'), 2, np.zeros(image.shape))
	image = output
	print(image.shape)
for j in range(0, 4):
	output = transpose_convolve(image, np.ones((4, 4), dtype='uint8'), 2, None)
	image = output
	print(image.shape)