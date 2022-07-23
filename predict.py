import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import argparse
import sys

class ESN:

    def __init__(self, W, W_reservoir,a):
        self.W = W
        self.W_reservoir = W_reservoir
        self.a = a

    def esn(self,x):

        self.x = x
        return np.multiply(self.a, self.x) + \
               np.multiply((1 - self.a), np.tanh(np.add(np.dot(self.W, self.x), np.dot(self.W_reservoir, self.x))))

def toExcel(filename,data,data2,name,name2):
	#data = data.reshape((1,100))
	data_df = pd.DataFrame(data)
	data_df2 = pd.DataFrame(data2)
	# create and writer pd.DataFrame to excel
	writer = pd.ExcelWriter(filename)
	data_df.to_excel(writer,name,float_format='%.8f') # float_format 控制精度
	data_df2.to_excel(writer,name2,float_format='%.8f') # float_format 控制精度
	writer.save()

if __name__ == '__main__':

    parser =argparse.ArgumentParser("Multiple Layered Echo State Network")
    parser.add_argument('dataset_file',nargs='?',
                        default=sys.stdin,
                        help='The dataset file to process. Reads from stdin by default.')
    args = parser.parse_args()

    trainLen = 3404
    testLen = 101
    initLen = 0
    data = []

    with open(args.dataset_file + ".txt",'r') as f:
        dt = f.readlines()
        for y in dt:
            data.append(ast.literal_eval(y))

    inSize=1
    outSize=1
    resSize=5
    a=0.4
    number_of_layers = 4

    #read params
    f = open('params_'+args.dataset_file)
    params = eval(f.read())

    W_L0=[[params['L0_1']],[params['L0_2']],[params['L0_3']],[params['L0_4']],[params['L0_5']]]
    W_reservoir_L0 = np.random.rand(resSize,resSize)-0.5
    W=np.random.rand(number_of_layers,resSize,resSize)-0.5
    W_reservoir = (np.random.rand(number_of_layers,resSize,resSize)-0.3)
    W_reservoir = W_reservoir * 0.13

    print(W_L0)
    X = np.zeros((resSize, trainLen - initLen))
    Yt = np.transpose(data[initLen+1:trainLen+1])
    x = np.zeros((resSize,1))

    for t in range(1,trainLen):

        u = data[t]

        # Layer 0
        x = np.multiply((1 - a), x) + \
            np.multiply(a, np.tanh(np.add(np.multiply(W_L0, u), np.dot(W_reservoir_L0, x))))

        # Layer 1
        x = ESN(W[0, :, :], W_reservoir[0, :, :], a).esn(x)

        # Layer 2
        x = ESN(W[1, :, :], W_reservoir[1, :, :], a).esn(x)

        # Layer 3
        x = ESN(W[2, :, :], W_reservoir[2, :, :], a).esn(x)
        # .
        # .
        # .

        # Layer N
        # x = ESN(W_Ln, W_reservoir_Ln, a).esn(x)

        if t > initLen:
            X[:resSize,t-initLen] = np.transpose(x)

    reg = 1e-8
    X_T = np.transpose(X)
    Wout = np.dot(Yt, np.linalg.pinv(X))
    Wout = np.expand_dims(Wout, axis=1)

    Y = np.transpose(np.zeros((outSize, testLen)))
    u = data[trainLen+1]

    for t in range(1,testLen):

        # print(u)

        x = np.multiply((1 - a), x) + \
            np.multiply(a, np.tanh(np.add(np.multiply(W_L0, u), np.dot(W_reservoir_L0, x))))

        # Layer 1
        x = ESN(W[0, :, :], W_reservoir[0, :, :], a).esn(x)

        # Layer 2
        x = ESN(W[1, :, :], W_reservoir[1, :, :], a).esn(x)

        # Layer 3
        x = ESN(W[2, :, :], W_reservoir[2, :, :], a).esn(x)


        # .
        # .
        # .

        # Layer N
        # x = ESN(W_Ln, W_reservoir_Ln, a).esn(x)

        y = np.asscalar(np.dot(np.transpose(Wout), x))

        Y[t] = y
        u = data[trainLen + t + 1]
	
    error = data[trainLen + 2] - Y[1]
    for i in range(0,len(Y)):
        Y[i] = Y[i] + error
    fig, ax = plt.subplots()
    font1 = {'family':'Times New Roman','weight':'normal','size':40}
    font2 = {'family': 'Times New Roman','weight': 'normal','size': 40}

    plt.tick_params(labelsize=25)
    plt.xlabel('Sequence number of positions',font2)
    plt.ylabel(args.dataset_file, font2)

    ax.ticklabel_format(useOffset=False)
    A = plt.plot(Y[1:],color='deepskyblue', linewidth=4.0, linestyle='--')
    B = plt.plot(data[trainLen + 1:trainLen + testLen + 1],color='coral',linewidth=4.0)

    legend = plt.legend(['Actual', 'Predicted'], prop=font1)

    plt.show()
    errorLen = 100
    mse = np.divide(np.sum(data[trainLen + 1:] - Y[1:]) ** 2, errorLen)
    print("Mean Squared Error:", (mse))
	
    toExcel(args.dataset_file + "_result.xls",data[trainLen + 1:],Y[1:],"Alon","Plon")

