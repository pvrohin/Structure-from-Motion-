import json
from PIL import Image
import numpy as np

def read_json(jsonPath):
	# open the json file
	with open(jsonPath, "r") as fp:
		# read the json data
		data = json.load(fp)
	
	# return the data
	return data

def get_image_c2w(jsonData, datasetPath):
	# define a list to store the image paths
	imagePaths = []
	
	# define a list to store the camera2world matrices
	c2ws = []
	# iterate over each frame of the data
	for frame in jsonData["frames"]:
		# grab the image file name
		imagePath = frame["file_path"]
		imagePath = imagePath.replace(".", datasetPath)
		imagePaths.append(f"{imagePath}.png")
		# grab the camera2world matrix
		c2ws.append(frame["transform_matrix"])
	
	# return the image file names and the camera2world matrices
	return (imagePaths, c2ws)

class GetImages():
	def __init__(self, imageWidth, imageHeight):
		# define the image width and height
		self.imageWidth = imageWidth
		self.imageHeight = imageHeight
	def __call__(self, imagePath):
		# OPTION 1: Original TensorFlow approach (requires: import tensorflow as tf)
		# read the image file
		# image = tf.io.read_file(imagePath)
		# decode the image string
		# image = tf.image.decode_jpeg(image, channels=3)
		# convert the image dtype from uint8 to float32
		# image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		# resize the image to the height and width in config
		# image = tf.image.resize(image, (self.imageHeight, self.imageWidth))
		# image = tf.reshape(image, (self.imageHeight, self.imageWidth, 3))
		
		# OPTION 2: PIL/Pillow approach (requires: from PIL import Image; import numpy as np)
		# read and decode the image
		# image = Image.open(imagePath).convert('RGB')
		# resize the image
		# image = image.resize((self.imageWidth, self.imageHeight), Image.Resampling.LANCZOS)
		# convert to numpy array and normalize to [0, 1]
		# image = np.array(image, dtype=np.float32) / 255.0
		
		# OPTION 3: OpenCV approach (requires: import cv2; import numpy as np)
		# read the image (BGR format)
		# image = cv2.imread(imagePath)
		# convert BGR to RGB
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# resize the image
		# image = cv2.resize(image, (self.imageWidth, self.imageHeight), interpolation=cv2.INTER_LINEAR)
		# convert to float32 and normalize to [0, 1]
		# image = image.astype(np.float32) / 255.0
		
		# OPTION 4: PIL with explicit shape handling (requires: from PIL import Image; import numpy as np)
		# read and decode the image
		image = Image.open(imagePath).convert('RGB')
		# resize the image
		image = image.resize((self.imageWidth, self.imageHeight), Image.Resampling.LANCZOS)
		# convert to numpy array
		image = np.array(image, dtype=np.float32)
		# normalize to [0, 1]
		image = image / 255.0
		# ensure correct shape
		image = image.reshape((self.imageHeight, self.imageWidth, 3))
		
		# return the image
		return image

class GetRays:
	def __init__(self, focalLength, imageWidth, imageHeight, near, 
		far, nC):
		# define the focal length, image width, and image height
		self.focalLength = focalLength
		self.imageWidth = imageWidth
		self.imageHeight = imageHeight
		# define the near and far bounding values
		self.near = near
		self.far = far
		# define the number of samples for coarse model
		self.nC = nC   

    def __call__(self, camera2world):
		# create a meshgrid of image dimensions
		(x, y) = tf.meshgrid(
			tf.range(self.imageWidth, dtype=tf.float32),
			tf.range(self.imageHeight, dtype=tf.float32),
			indexing="xy",
		)
		# define the camera coordinates
		xCamera = (x - self.imageWidth * 0.5) / self.focalLength
		yCamera = (y - self.imageHeight * 0.5) / self.focalLength
		# define the camera vector
		xCyCzC = tf.stack([xCamera, -yCamera, -tf.ones_like(x)],
			axis=-1)
		# slice the camera2world matrix to obtain the rotation and
		# translation matrix
		rotation = camera2world[:3, :3]
		translation = camera2world[:3, -1]

        # expand the camera coordinates to 
		xCyCzC = xCyCzC[..., None, :]
		
		# get the world coordinates
		xWyWzW = xCyCzC * rotation
		
		# calculate the direction vector of the ray
		rayD = tf.reduce_sum(xWyWzW, axis=-1)
		rayD = rayD / tf.norm(rayD, axis=-1, keepdims=True)
		# calculate the origin vector of the ray
		rayO = tf.broadcast_to(translation, tf.shape(rayD))
		# get the sample points from the ray
		tVals = tf.linspace(self.near, self.far, self.nC)
		noiseShape = list(rayO.shape[:-1]) + [self.nC]
		noise = (tf.random.uniform(shape=noiseShape) * 
			(self.far - self.near) / self.nC)
		tVals = tVals + noise
		# return ray origin, direction, and the sample points
		return (rayO, rayD, tVals) 