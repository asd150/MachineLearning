import numpy as np
import pandas as pd
from sklearn.linear_model import lars_path
# from generator import data_generator
import scipy.linalg as la


def p_lasso(A, lambda_=0.01, iter=300):
	# row: features, column: samples
	m = A.shape[1]
	k = A.shape[0]
	S = np.cov(A)
	# W: covariance matrix, T: precision matrix (W's inverse)
	W = S.copy() + 0.001 * np.eye(k) # how to choose the lambda value?
	T = np.linalg.inv(W)

	print('Before:')
	print(W)
	print('\n')

	indices = np.arange(k)
	for n in range(iter):
		for i in range(k):
			# upper left sub-matrix of W
			W_11 = W[indices != i].T[indices != i]
			S_12 = W[i][indices != i]
			X = la.sqrtm(W_11)
			y = np.dot(np.linalg.inv(X), S_12)
			# print(y)
			coefs_ = [0 for j in range(k - 1)]
			coefs_ = lasso_regression(X, y, coefs_, lambda_)
			# print(W[indices != i,i].shape)

			T[i][i] = 1 / (W[i][i] - np.dot(W[indices != i,i], coefs_))
			# print("coef ",coefs_)
			# print(T[i][i])
			T[indices != i, i] = - T[i, i] * np.asarray(coefs_)
			T[i, indices != i] = - T[i, i] * np.asarray(coefs_)
			# print(T[i][i])
			temp_coefs = np.dot(W_11, coefs_)
			W[i, indices != i] = temp_coefs
			W[indices != i, i] = temp_coefs

		# 	S_12 = W[i][indices != i]
		# 	#solve the lasso problem
		# 	'''
		# 	_, _, coefs_ = lars_path( W_11, S_12, Xy = S_12, Gram = W_11,
		# 								alpha_min = lambda_/(k-1), copy_Gram = True,
		# 								method = "lasso")
		# 	coefs_ = coefs_[:,-1] # only the last column
		# 	'''
		# 	X = la.sqrtm(W_11)
		# 	y = np.dot(np.linalg.inv(X), S_12)
		# 	coefs_ = [0 for j in range(k-1)]
		#
		# 	coefs_ = lasso_regression(X, y, coefs_, lambda_)
		# 	print(coefs_)
		# 	#update the precision matrix.
		# 	T[i][i] = 1 / (W[i][i] - np.dot(W[indices != i][i], coefs_))
		# 	T[indices != i][i] = - T[i][i] * coefs_
		# 	T[i][indices != i] = - T[i][i] * coefs_
		# 	temp_coefs = np.dot(W_11, coefs_)
		# 	W[i][indices != i] = temp_coefs
		# 	W[indices != i][i] = temp_coefs
		#
		print('Iter:', n+1)
		print(W)
		# print('\n')

def p_lasso2(A, lambda_=0.01, iter=500):
	# row: features, column: samples
	m = A.shape[1]
	k = A.shape[0]
	S = np.cov(A)
	# W: covariance matrix, T: precision matrix (W's inverse)
	W = S.copy() + 0.001 * np.eye(k) # how to choose the lambda value?
	T = np.linalg.inv(W)

	print('Before:')
	print(W)
	print('\n')

	indices = np.arange(k)
	for n in range(iter):
		for i in range(k):
			# upper left sub-matrix of W
			W_11 = W[indices != i].T[indices != i]
			S_12 = W[i, indices != i]
			#solve the lasso problem

			_, _, coefs_ = lars_path( W_11, S_12, Xy = S_12, Gram = W_11,
										alpha_min = lambda_/(k-1), copy_Gram = True,
										method = "lasso")
			coefs_ = coefs_[:,-1] # only the last column

			# X = la.sqrtm(W_11)
			# y = np.dot(np.linalg.inv(X), S_12)
			# coefs_ = [0 for j in range(k-1)]
			# coefs_ = lasso_regression(X, y, coefs_, lambda_)
			# print(coefs_)
			# #update the precision matrix.
			T[i, i] = 1 / (W[i,i] - np.dot(W[indices != i, i], coefs_))
			T[indices != i, i] = - T[i, i] * np.asarray(coefs_)
			T[i, indices != i] = - T[i, i] * np.asarray(coefs_)
			temp_coefs = np.dot(W_11, coefs_)
			W[i, indices != i] = temp_coefs
			W[indices != i, i] = temp_coefs

		print('Iter:', n+1)
		print(W)
		print('\n')


def lasso_regression(X, y, weights, lambda_, iter=500):
	# k = number of coordinates
	k = len(weights)
	for n in range(iter):
		for i in range(k):
			ceiling = -np.dot(X[:, i], y-np.dot(X, weights))
			floor = np.dot(X[:, i], X[:, i])
			upper = (ceiling + lambda_/2) / floor
			lower = (ceiling - lambda_/2) / floor
			if weights[i] > upper:
				weights[i] -= upper
			elif weights[i] < lower:
				weights -= lower
			else:
				weights[i] = 0
	return weights




def read_data(name):
	data = pd.read_csv(name)
	return data


if __name__ == '__main__':
	
	# row: features, column: samples
	X = np.array([[1, 5, 6],
				  [4, 3, 9],
				  [4, 2, 9]])


	# the input matrix is the tranpose of the original data set
	# p_lasso2(X)
	'''
	data = read_data('example.csv')
	data = data.values
	print(data)
	'''