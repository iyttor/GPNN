import argparse
from model.gpnn import gpnn
from model.parser import *
from skimage.transform import rescale, resize

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser = parser_general(parser)
	parser = parser_analogies(parser)
	config = vars(parser.parse_args())
	model = gpnn(config)
	refine_img = model.run(to_save=False)
	model.coarse_img = resize(refine_img, model.coarse_img.shape[:2])
	model.run()
