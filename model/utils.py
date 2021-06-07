from imageio import imread, imsave
from skimage.util import img_as_ubyte
import os


def img_read(path):
	im = imread(path)
	if im.shape[2] > 3:
		im = im[:, :, :3]
	return im


def img_save(im, path):
	dir = os.path.splitext(os.path.dirname(path))[0]
	if not os.path.isdir(dir):
		os.mkdir(dir)
	imsave(path, img_as_ubyte(im))
