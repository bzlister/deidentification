from keras import layers
from keras import models
from keras.layers import Activation, BatchNormalization, Dropout, Reshape, Lambda
from keras.layers import LeakyReLU
import cv2
import numpy as np
from keras.backend import tf as ktf

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

def resize_output(input_shape):
	return (31, 31)

def buildGenerator():
	model = models.Sequential()
	model.add(layers.Conv2D(filters=1, kernel_size=(4,4), strides=2, activation='relu', input_shape=(64, 64, 1)))

	model.add(layers.Conv2D(filters=1, kernel_size=(4,4), strides=2, activation=None, input_shape=(31, 31, 1)))
	model.add(layers.BatchNormalization())
	model.add(LeakyReLU(alpha=0.01))

	model.add(layers.Conv2D(filters=1, kernel_size=(4,4), strides=2, activation=None, input_shape=(14, 14, 1)))
	model.add(layers.BatchNormalization())
	model.add(LeakyReLU(alpha=0.01))

	model.add(layers.Conv2D(filters=1, kernel_size=(4,4), strides=2, activation=None, input_shape=(6, 6, 1)))
	model.add(LeakyReLU(alpha=0.01))

	model.add(layers.Conv2DTranspose(filters=1, kernel_size=(4,4), strides=2, activation=None, input_shape=(2, 2, 1)))
	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(0.5))
	model.add(Activation('relu'))

	model.add(layers.Conv2DTranspose(filters=1, kernel_size=(4,4), strides=2, activation=None, input_shape=(6, 6, 1)))
	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(0.5))
	model.add(Activation('relu'))

	model.add(layers.Conv2DTranspose(filters=1, kernel_size=(4,4), strides=2, activation=None, input_shape=(14, 14, 1)))
	model.add(layers.BatchNormalization())
	model.add(Activation('relu'))
	model.add(Lambda(lambda image: ktf.image.resize_images(image, (31,31)), output_shape=(31, 31, 1)))

	model.add(layers.Conv2DTranspose(filters=1, kernel_size=(4,4), strides=2, activation='relu', input_shape=(31, 31, 1)))
	return model

model = buildGenerator()
print(model.summary())
