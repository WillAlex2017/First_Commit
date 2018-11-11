import argparse
from neural_networks.rnn_one_hot import RNNOneHot
from lazy.markov_model import MarkovModel
from lazy.user_knn import UserKNN
from neural_networks.stacked_denoising_autoencoder import StackedDenoisingAutoencoder
from factorization.bprmf import BPRMF
from helpers.early_stopping import early_stopping_command_parser, get_early_stopper
from neural_networks.recurrent_layers import recurrent_layers_command_parser, get_recurrent_layers
from neural_networks.update_manager import update_manager_command_parser, get_update_manager
from neural_networks.sequence_noise import sequence_noise_command_parser, get_sequence_noise
from neural_networks.target_selection import target_selection_command_parser, get_target_selection

def command_parser(*sub_command_parser):
	''' *sub_command_parser should be callables that will add arguments to the command parser
	'''

	parser = argparse.ArgumentParser()

	for scp in sub_command_parser:
		scp(parser)

	args = parser.parse_args()
	return args

def predictor_command_parser(parser):
	parser.add_argument('-m', dest='method', choices=['RNN', 'SDA', 'BPRMF', 'FPMC', 'FISM', 'Fossil', 'LTM', 'UKNN', 'MM', 'POP'],
	 help='Method', default='RNN')
	parser.add_argument('-b', dest='batch_size', help='Batch size', default=16, type=int)
	parser.add_argument('-l', dest='learning_rate', help='Learning rate', default=0.01, type=float)
	parser.add_argument('-r', dest='regularization', help='Regularization (positive for L2, negative for L1)', default=0., type=float)
	parser.add_argument('-g', dest='gradient_clipping', help='Gradient clipping', default=100, type=int)
	parser.add_argument('-H', dest='hidden', help='Number of hidden neurons (for LTM and BPRMF)', default=20, type=int)
	parser.add_argument('-L', dest='layers', help='Layers (for SDA)', default="20", type=str)
	parser.add_argument('--db', dest='diversity_bias', help='Diversity bias (for RNN with CCE, TOP1, BPR or Blackout loss)', default=0.0, type=float)
	parser.add_argument('--rf', help='Use rating features.', action='store_true')
	parser.add_argument('--mf', help='Use movie features.', action='store_true')
	parser.add_argument('--uf', help='Use users features.', action='store_true')
	parser.add_argument('--ns', help='Neighborhood size (for UKNN).', default=80, type=int)
	parser.add_argument('--cooling', help='Simulated annealing', default=1., type=float)
	parser.add_argument('--init_sigma', help='Sigma of the gaussian initialization (for MF)', default=1, type=float
	parser.add_argument('--no_adaptive_sampling', help='No adaptive sampling (for MF)', action='store_true')
	parser.add_argument('--fpmc_bias', help='Sampling bias (for MF)', default=100., type=float)
	parser.add_argument('--ltm_no_trajectory', help='Do not use users trajectory in LTM, just use word2vec', action='store_true')
	parser.add_argument('--max_length', help='Maximum length of sequences during training (for RNNs)', default=30, type=int)
	parser.add_argument('--repeated_interactions', help='The model can recommend items with which the user already interacted', action='store_true')
	update_manager_command_parser(parser)
	recurrent_layers_command_parser(parser)
	sequence_noise_command_parser(parser)
	target_selection_command_parser(parser)

def get_predictor(args):
	args.layers = map(int, args.layers.split('-'))

	updater = get_update_manager(args)
	recurrent_layer = get_recurrent_layers(args)
	sequence_noise = get_sequence_noise(args)
	target_selection = get_target_selection(args)

	if args.method == "MF":
		return BPRMF(k=args.hidden, reg = args.regularization, learning_rate = args.learning_rate, annealing=args.cooling, init_sigma = args.init_sigma, adaptive_sampling=(not args.no_adaptive_sampling), sampling_bias=args.fpmc_bias)
	elif args.method == "UKNN":
		return UserKNN(neighborhood_size=args.ns)
	elif args.method == "MM":
		return MarkovModel()
	elif args.method == 'RNN':
		return RNNOneHot(interactions_are_unique=(not args.repeated_interactions), max_length=args.max_length, diversity_bias=args.diversity_bias, regularization=args.regularization, updater=updater, target_selection=target_selection, sequence_noise=sequence_noise, recurrent_layer=recurrent_layer, use_ratings_features=args.rf, use_movies_features=args.mf, use_users_features=args.uf, batch_size=args.batch_size)