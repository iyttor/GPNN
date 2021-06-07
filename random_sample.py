import argparse
from model.gpnn import gpnn
from model.parser import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser = parser_general(parser)
	parser = parser_sample(parser)
	config = vars(parser.parse_args())
	model = gpnn(config)
	model.run()
