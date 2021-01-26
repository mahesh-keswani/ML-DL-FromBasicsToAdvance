import numpy as np 

def do_rmsprop(init_w=None, init_b=None, X, Y, lr=0.01, eps=1e-8, max_epochs=100, beta = 0.9):
	# assuming parameters as scaler
	w, b, lr = 2, 2, lr
	# for accumulating history of parameters
	v_w, v_b = 0, 0
	# making sure denominator is not zero
	eps = eps

	for i in range(max_epochs):
		dw, db = 0, 0

		for x, y in zip(X, Y):
			dw += grad_w(w, b, x, y)
			db += grad_b(w, b, x, y)

		# accumulating the frequency of updates for every parameter
		# also controlling the denominator, so as the timesteps increases
		# the learning rate won't just vanish, resulting in almost no updates 
		# for parameters
		v_w = beta * v_w + (1 - beta) * dw**2
		v_b = beta * v_b + (1 - beta) * db**2

		# decaying (or adapting) the learning rate depending on the frequency of update
		w = w - ( lr / np.sqrt(v_w + eps) ) * dw
		b = b - ( lr / np.sqrt(v_b + eps) ) * db
		