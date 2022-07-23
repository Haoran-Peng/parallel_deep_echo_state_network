import pandas as pd
import ast
from DeepESN import DeepESN
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from bayes_opt import SequentialDomainReductionTransformer
import numpy as np
import argparse
import sys

if __name__ == '__main__':

	parser =argparse.ArgumentParser("Multiple Layered Echo State Network")
	parser.add_argument('dataset_file',nargs='?',
                        default=sys.stdin,
                        help='The dataset file to process. Reads from stdin by default.')
	args = parser.parse_args()

	data = []
	with open(args.dataset_file + ".txt",'r') as f:
		dt = f.readlines()
		for y in dt:
			data.append(ast.literal_eval(y))

	TrainData = data[:-100]
	TestData = data[-100:]
	model = DeepESN(TrainData, TestData, inSize=1, outSize=1, reg = 1e-8)

	pbounds = {'L0_1': (-0.5, 0.5), 'L0_2': (-0.5, 0.5), 'L0_3':(-0.5, 0.5), 'L0_4': (-0.5, 0.5), 'L0_5': (-0.5, 0.5)}

	# bounds_transformer = SequentialDomainReductionTransformer()
	optimizer = BayesianOptimization(
		f=model.fit,
		pbounds=pbounds,
		random_state=10,
		# bounds_transformer=bounds_transformer
	)
	optimizer.maximize(
		init_points=2,
		n_iter=10
	)

	print(optimizer.max)
	f = open("params_"+args.dataset_file, "w")
	f.write(str(optimizer.max['params']))
	f.close()

	losses = []
	for i, res in enumerate(optimizer.res):
		#print("Iteration {}: \n\t{}".format(i, res))
		losses.append(-res['target'])
	np.savetxt("deepESN_loss_"+args.dataset_file+".csv", losses, delimiter=",")
	plt.xlabel('episodes')
	plt.ylabel('Loss')
	plt.plot(losses,color='deepskyblue', linewidth=4.0, linestyle='--')
	plt.show()
