import numpy as np

class ESN:

	def __init__(self, W, W_reservoir,a):
		self.W = W
		self.W_reservoir = W_reservoir
		self.a = a

	def esn(self,x):

		self.x = x
		return np.multiply(self.a, self.x) + \
			   np.multiply((1 - self.a), np.tanh(np.add(np.dot(self.W, self.x), np.dot(self.W_reservoir, self.x))))

class DeepESN:

	def __init__(self, TrainData, TestData, inSize, outSize, reg = 1e-8):
		self.TrainData = TrainData
		self.TestData = TestData
		self.inSize = inSize
		self.outSize = outSize
		self.reg = reg
		self.W_L0 = None

	def fit(self, L0_1, L0_2, L0_3, L0_4, L0_5):
		trainLen = len(self.TrainData)-1
		testLen = len(self.TestData)-1
		initLen = 0
		data = self.TrainData
		reg = self.reg

		inSize=1
		outSize=1
		resSize=5
		a=0.4
		number_of_layers = 4

		L0_list = []
		L0_list.append([L0_1])
		L0_list.append([L0_2])
		L0_list.append([L0_3])
		L0_list.append([L0_4])
		L0_list.append([L0_5])

		W_L0= np.array(L0_list)
		#(np.random.rand(resSize,1)-0.5) * 1
		W_reservoir_L0 = np.random.rand(resSize,resSize)-0.5
		W=np.random.rand(number_of_layers,resSize,resSize)-0.5
		W_reservoir = (np.random.rand(number_of_layers,resSize,resSize)-0.3)
		W_reservoir = W_reservoir * 0.13

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

		X_T = np.transpose(X)
		Wout = np.dot(Yt, np.linalg.pinv(X))
		Wout = np.expand_dims(Wout, axis=1)

		Y = np.transpose(np.zeros((outSize, testLen)))
		# u = data[trainLen+1]

		for t in range(1,testLen):
			u = self.TestData[t]

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
			
		
		error = self.TestData[1] - Y[1]
		for i in range(0,len(Y)):
			Y[i] = Y[i] + error

		errorLen = testLen
		mse = np.divide(np.sum(self.TestData[1:] - Y[1:]) ** 2, errorLen)

		self.W_L0 = W_L0
		return -mse
