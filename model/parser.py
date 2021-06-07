def parser_general(parser):
	parser.add_argument('-out', '--out_dir', type=str, required=False, default='./output', help='path of output dir')
	parser.add_argument('--coarse_dim', type=int, default=14, required=False, help='the height of the coarsest pyramid level. default is 14 (int)')
	parser.add_argument('--out_size', type=int, default=0, required=False, help='output image height. should be smaller than original image. default is 0 - as input (int)')
	parser.add_argument('--patch_size', type=int, default=7, required=False, help='the size of the square patches to use in nearest neighbors. default is 7 (int)')
	parser.add_argument('--stride', type=int, default=1, required=False, help='the stride between patches in the nearest neighbros method. default is 1 (int)')
	parser.add_argument('--iters', type=int, default=10, required=False, help='number of refinement iterations in each pyramid scale. default is 10 (int)')
	parser.add_argument('--pyramid_ratio', type=float, default=4 / 3, required=False, help='the ratio between pyramid scales. default is 4/3 (float)')
	parser.add_argument('--faiss', action='store_true', default=False, help='indicate to use faiss approximate nearest-neighbor. default is False (boolean)')
	parser.add_argument('--no_cuda', action='store_true', default=False, help='indicate to run only on cpu. default is False (boolean)')
	return parser


def parser_sample(parser):
	parser.add_argument('-in', '--input_img', type=str, required=True, help='path of input image')
	parser.add_argument('--sigma', type=float, default=0.75, required=False, help='noise level to adjust the variatonality of the new sample. default is 0.75 (float)')
	parser.add_argument('--alpha', type=float, default=0.005, required=False, help='alpha parameter of the normalizing distance matrix. small alpha encourages completeness. default is 0.005 (float)')
	parser.add_argument('--task', type=str, default='random_sample')
	return parser


def parser_analogies(parser):
	parser.add_argument('-a', '--img_a', type=str, required=True, help='path of image A - the content')
	parser.add_argument('-b', '--img_b', type=str, required=True, help='path of image B - the structure')
	parser.add_argument('--alpha', type=float, default=0.005, required=False, help='alpha parameter of the normalizing distance matrix. small alpha encourages completeness. default is 0.005 (float)')
	parser.add_argument('--task', type=str, default='structural_analogies')
	return parser


def parser_inpainting(parser):
	parser.add_argument('-in', '--input_img', type=str, required=True, help='path of input image')
	parser.add_argument('-m', '--mask', type=str, required=True, help='path of an image with ones where the inpainting is in the input image and zeroes elsewhere')
	parser.add_argument('--alpha', type=float, default=1, required=False, help='alpha parameter of the normalizing distance matrix. small alpha encourages completeness. default is 1.0 (float)')
	parser.add_argument('--task', type=str, default='inpainting')
	return parser
