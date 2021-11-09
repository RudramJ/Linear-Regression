import numpy as np
import matplotlib.pyplot as plt
import time
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

filename = "ex1data1.txt"

theta0 = 0
theta1 = 0
alpha = 0.01
prevCostValue = 0
xTheta0 = []
yTheta1 = []
zCost = []

def cost_function(t0, t1, arrayLength, xData, yData):
	addition = 0
	for i in range(0, arrayLength):
		temp = (t0 + (t1 * xData[i]) - yData[i])
		sqr = temp * temp
		addition = addition + sqr
	costValue = addition/(2 * arrayLength)
	return costValue


def cost_function_using_np(tempTheta0Const, tempThetaVector, tempXData, tempYData):
	
	retVal = 0
	thetaConstant = np.matrix([tempTheta0Const, tempXData])
	thetaConstant = np.transpose(thetaConstant)

	hypothesisFunc = np.dot(thetaConstant, tempThetaVector)
	hypoMinusY = np.subtract(hypothesisFunc, tempYData)

	hypoMinYSq = np.square(hypoMinusY)
	hypoMinYSqSum = np.sum(hypoMinYSq)
	retVal = hypoMinYSqSum/(2*len(tempXData))

	return retVal


def update_theta0(t0, t1, arrayLength, xData, yData, alpha):
	retVal = 0
	addition = 0
	for i in range(0, arrayLength):
		temp = (t0 + (t1 * xData[i]) - yData[i])
		addition = addition + temp	
	thetaVal0 = addition/arrayLength
	retVal = t0 - (alpha * thetaVal0)
	# print (retVal)
	return retVal 

def update_theta0_using_np(tempTheta0Const, tempThetaVector, tempXData, tempYData):
	
	retVal = 0
	addition = 0
	theta0Constant = np.matrix([tempTheta0Const, tempXData])
	theta0Constant = np.transpose(theta0Constant)

	hypothesisFunc = np.dot(theta0Constant, tempThetaVector)
	hypoMinusY = np.subtract(hypothesisFunc, tempYData)
	
	#calculation for theta0
	hypoMinYSum = np.sum(hypoMinusY)
	hypoMinYSumMean = hypoMinYSum/(len(tempXData))
	retValTheta0 = (tempThetaVector[0] - (alpha * hypoMinYSumMean))
	
	#calculation for theta1
	hypoMinusYMulX = np.multiply(hypoMinusY, tempXData)
	hypoMinusYMulXSum = np.sum(hypoMinusYMulX)
	hypoMinusYMulXSumMean = hypoMinusYMulXSum/(len(tempXData))
	retValTheta1 = (tempThetaVector[1] - (alpha * hypoMinusYMulXSumMean))

	return retVal 


def update_theta1(t0, t1, arrayLength, xData, yData, alpha):
	retVal = 0
	addition = 0
	for i in range(0, arrayLength):
		temp = (t0 + (t1 * xData[i]) - yData[i])
		mult = temp * xData[i]
		addition = addition + mult	
	thetaVal1 = addition/arrayLength
	retVal = t1 - (alpha * thetaVal1)
	# print (retVal)
	return retVal

def update_theta1_using_np(tempTheta0Const, tempThetaVector, tempXData, tempYData):
	
	retVal = 0
	addition = 0
	theta1Constant = np.matrix([tempTheta0Const, tempXData])
	theta1Constant = np.transpose(theta1Constant)

	hypothesisFunc = np.dot(theta1Constant, tempThetaVector)
	hypoMinusY = np.subtract(hypothesisFunc, tempYData)

	hypoTheta1 = np.multiply(hypoMinusY, tempXData)
	hypoMinYSum = np.sum(hypoTheta1)
	hypoMinYSumMean = hypoMinYSum/(len(tempXData))

	retVal = (tempThetaVector[1] - (alpha * hypoMinYSumMean))

	return retVal 


def update_theta_using_np(tempTheta0Const, tempThetaVector, tempXData, tempYData):
	
	retValTheta = np.array([0, 0])
	retValTheta = retValTheta.astype(float)
	# retValTheta0 = 0
	# retValTheta1 = 0
	addition = 0
	theta0Constant = np.matrix([tempTheta0Const, tempXData])
	theta0Constant = np.transpose(theta0Constant)

	hypothesisFunc = np.dot(theta0Constant, tempThetaVector)
	hypoMinusY = np.subtract(hypothesisFunc, tempYData)
	
	#calculation for theta0
	hypoMinYSum = np.sum(hypoMinusY)
	hypoMinYSumMean = hypoMinYSum/(len(tempXData))
	retValTheta[0] = (tempThetaVector[0] - (alpha * hypoMinYSumMean))
	# retValTheta0 = (tempThetaVector[0] - (alpha * hypoMinYSumMean))
	
	#calculation for theta1
	hypoMinusYMulX = np.multiply(hypoMinusY, tempXData)
	hypoMinusYMulXSum = np.sum(hypoMinusYMulX)
	hypoMinusYMulXSumMean = hypoMinusYMulXSum/(len(tempXData))
	retValTheta[1] = (tempThetaVector[1] - (alpha * hypoMinusYMulXSumMean))
	# retValTheta1 = (tempThetaVector[1] - (alpha * hypoMinusYMulXSumMean))

	# print (retValTheta0)
	# print (retValTheta1)
	# print (retValTheta)
	return retValTheta



localtime = time.asctime( time.localtime(time.time()) )
print ((localtime))

data = np.loadtxt(filename,delimiter=",")
arrayLength = len(data)

xData = data[:,0]
yData = data[:,1]

thetaVector = np.array([0, 0])
thetaVector = thetaVector.astype(float)
theta0Const = np.ones(97)
print(thetaVector)
print(theta0Const)

prevCostValue = cost_function_using_np(theta0Const, thetaVector, xData, yData)

noIteration = 0
if(prevCostValue != 0):
	while 1:

		# theta0Temp = update_theta0_using_np(theta0Const, thetaVector, xData, yData)
		# theta1Temp = update_theta1_using_np(theta0Const, thetaVector, xData, yData)
		# thetaVector[0] = theta0Temp
		# thetaVector[1] = theta1Temp
		thetaVector = update_theta_using_np(theta0Const, thetaVector, xData, yData)
		tempCost = cost_function_using_np(theta0Const, thetaVector, xData, yData)
		# prevCostValue = tempCost
		if(prevCostValue != tempCost):
			diff = abs(tempCost - prevCostValue)
			if(diff < 0.000000001):
				localtime = time.asctime( time.localtime(time.time()) )
				print (localtime)
				print ("theta0", thetaVector[0])
				print ("theta1", thetaVector[1])
				print ("finalCostValue", tempCost)
				break
			prevCostValue = tempCost
		else:
			print ("i m here")
			break  
		noIteration = noIteration + 1
print (noIteration)

plt.plot(xData, yData, 'ro')
plt.plot(xData, xData*thetaVector[1] + thetaVector[0], 'g')
plt.show()