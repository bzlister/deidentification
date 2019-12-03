from keras import layers
from keras import models
from keras.models import Model
from keras.layers import Activation, BatchNormalization, Dropout, Reshape, Lambda, LeakyReLU, Input, Concatenate
import cv2
import numpy as np
from keras.backend import tf as ktf
import random
from keras.optimizers import Adam
from keras.layers.merge import concatenate

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

def view(image):
	pyplot.imshow(image, pyplot.cm.gray)
	pyplot.show()

def load_dataset():
	dataset = []
	holding = []	
	files = os.listdir('lfwcrop_grey/faces')
	for file in files:
		dataset.append(read_pgm('lfwcrop_grey/faces/' + file)/255)
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

#PLACEHOLDER! Actual loss is a function of the discriminator and generator
#This simply calculates distance from the average face of the dataset
def avg_loss(truth, x):
	avg = average_face(load_dataset)
	error = 0
	for i in range(0, 64):
		for j in range(0, 64):
			error += (avg[i][j]/255 - x[i][j]/255)**2	

def buildGenerator():
	inputs = Input((64, 64, 1))
	input_layer = layers.Conv2D(filters=1, kernel_size=(4,4), strides=2, activation=None, input_shape=(64, 64, 1))(inputs)

	c1 = layers.Conv2D(filters=1, kernel_size=(4,4), strides=2, activation=None, input_shape=(31, 31, 1))(input_layer)
	c1 = layers.BatchNormalization()(c1)
	c1 = LeakyReLU(alpha=0.01)(c1)

	c2 = layers.Conv2D(filters=1, kernel_size=(4,4), strides=2, activation=None, input_shape=(14, 14, 1))(c1)
	c2 = layers.BatchNormalization()(c2)
	c2 = LeakyReLU(alpha=0.01)(c2)

	c3 = layers.Conv2D(filters=1, kernel_size=(4,4), strides=2, activation=None, input_shape=(6, 6, 1))(c2)
	c3 = LeakyReLU(alpha=0.01)(c3)

	d1 = layers.Conv2DTranspose(filters=1, kernel_size=(4,4), strides=2, activation=None, input_shape=(2, 2, 1))(c3)
	d1 = layers.BatchNormalization()(d1)
	d1 = layers.Dropout(0.5)(d1)
	d1 = Activation('relu')(d1)
	d1 = concatenate([d1, c2])

	d2 = layers.Conv2DTranspose(filters=1, kernel_size=(4,4), strides=2, activation=None, input_shape=(6, 6, 1))(d1)
	d2 = layers.BatchNormalization()(d2)
	d2 = layers.Dropout(0.5)(d2)
	d2 = Activation('relu')(d2)
	d2 = concatenate([d2, c1])
	
	d3 = layers.Conv2DTranspose(filters=1, kernel_size=(4,4), strides=2, activation=None, input_shape=(14, 14, 1))(d2)
	d3 = layers.BatchNormalization()(d3)
	d3 = Activation('relu')(d3)
	d3 = Lambda(lambda image: ktf.image.resize_images(image, (31,31)), output_shape=(31, 31, 1))(d3)

	output_layer = layers.Conv2DTranspose(filters=1, kernel_size=(4,4), strides=2, activation=None, input_shape=(31, 31, 1))(d3)
	outputs = Activation('relu')(output_layer)
	model = Model(inputs=[inputs], outputs=[outputs])
	return model

def buildDiscriminator():
	input_src = Input(shape=(64,64,1))
	input_target = Input(shape=(64,64,1))
	merged = Concatenate()([input_src, input_target])
	layer1 = layers.Conv2D(filters=1, kernel_size=(4,4), strides=2, activation=None, input_shape=(64,64,1))(merged)
	
	layer2 = layers.Conv2D(filters=1, kernel_size=(4,4), strides=2, activation=None, input_shape=(31,31,1))(layer1)
	layer2 = layers.BatchNormalization()(layer2)
	layer2 = LeakyReLU(alpha=0.01)(layer2)
	
	layer3 = layers.Conv2D(filters=1, kernel_size=(4,4), strides=1, activation=None, input_shape=(14,14,1))(layer2)
	layer3 = layers.BatchNormalization()(layer3)
	layer3 = LeakyReLU(alpha=0.01)(layer3)
	
	layer4 = layers.Conv2D(filters=1, kernel_size=(4,4), strides=1, activation=None, input_shape=(11,11,1))(layer3)
	layer4 = LeakyReLU(alpha=0.01)(layer4)
	
	out = Activation('sigmoid')(layer4)
	
	opt = Adam(lr=0.0002, beta_1=0.5)
	model = Model([input_src, input_target], out)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model

def buildGan():
	g_model = buildGenerator()
	d_model = buildDiscriminator()
	image_shape =  (64,64,1)
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model

gan = buildGan()
print(gan.summary())