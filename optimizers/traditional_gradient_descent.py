def do_gradient_descent(init_w, init_b, lr, max_epochs, X, Y):
	# init_w: can be problems specific: scaler, vector or matrix, or tensor
	w, b, lr, max_epochs = 2, 2, lr, max_epochs

	for i in range(max_epochs):
		dw, db = 0, 0
		for x, y in zip(X, Y):
			dw += grad_w(w, b, x, y)
			db += grad_b(w, b, x, y)

		w = w - lr*dw
		b = b - lr*db

	return w, b