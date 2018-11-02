import numpy as np
from regression import OLS, Ridge,Lasso
np.random.seed(12)
from sklearn.metrics import mean_squared_error, r2_score

def K_fold(x,y,k,alpha,model):
	"""Function to who calculate the average MSE and R2 using k-fold.
	Takes in x,y and z varibles for a dataset, k number of folds, alpha and which method beta shall use. (OLS,Ridge or Lasso)
	Returns average MSE and average R2"""
	m=x.shape[1]
	#if len(x.shape) > 1:
	#	x = np.ravel(x)
	#	y = np.ravel(y)

	n=x.shape[0]
	n_k=int(n/k)
	if n_k*k!=n:
		print("k needs to be a multiple of ", n,k)
	i=np.arange(n)
	np.random.shuffle(i)

	MSE_=0
	R2_=0

	for t in range(k):
		x_,y_,x_test,y_test=train_test_data(x,y,i[t*n_k:(t+1)*n_k])
		#X= create_X(x_,y_,n=m)
		#X_test= create_X(x_test,y_test,n=m)

		model.fit(x_,y_)


		MSE_+=mean_squared_error(y_test, model.predict(x_test))
		R2_+=r2_score(y_test, model.predict(x_test))


	return [MSE_/k, R2_/k]

def train_test_data(x_,y_,i):
	"""
	Takes in x,y and z arrays, and a array with random indesies iself.
	returns learning arrays for x, y and z with (N-len(i)) dimetions
	and test data with length (len(i))
	"""
	x_learn=np.delete(x_,i,0)
	y_learn=np.delete(y_,i,0)
	x_test=np.take(x_,i,0)
	y_test=np.take(y_,i,0)


	return x_learn,y_learn,x_test,y_test

def savefigure(name, figure = "gcf"):
	"""
	Function for saving figures as a .tex-file for easier integration with latex.
	"""
	try:
		from matplotlib2tikz import save as tikz_save
		tikz_save(name.replace(" ", "_") + ".tex", figure = figure, figureheight='\\figureheight', figurewidth='\\figurewidth')
	except ImportError:
		print("Please install matplotlib2tikz to save figure as a .tex-file.")
		import matplotlib.pyplot as plt
		if figure == "gcf":
			plt.savefig(name+".pdf")
		else:
			fig=figure
			fig.savefig(name+".pdf")
