import numpy as np 

def do_adagrad(init_w=None, init_b=None, X, Y, lr=0.01, eps=1e-8, max_epochs=100):
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
		v_w = v_w + dw**2
		v_b = v_b + db**2

		# decaying (or adapting) the learning rate depending on the frequency of update
		w = w - ( lr / np.sqrt(v_w + eps) ) * dw
		b = b - ( lr / np.sqrt(v_b + eps) ) * db
		