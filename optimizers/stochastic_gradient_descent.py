def do_gradient_descent(init_w = None, init_b=None, lr=0.01, max_epochs=100, X, Y):
	# assuming parameters as scaler
	w, b, lr, max_epochs = 2, 2, lr, max_epochs

	for i in range(max_epochs):
		dw, db = 0, 0
		for x, y in zip(X, Y):
			dw += grad_w(w, b, x, y)
			db += grad_b(w, b, x, y)

			# updating parameters for every data point
			w = w - lr*dw
			b = b - lr*db

	return w, b