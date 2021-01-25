def mini_batch_gradient_descent(init_w = None, init_b = None, X, Y, batch_size, lr=0.01, max_epochs=100):
	# assuming parameters as scaler
	w, b, lr, max_epochs = 2, 2, lr, max_epochs
	batch_size = batch_size
	# for number of points seen
	num_points_seen = 0

	for i in range(max_epochs):
		dw, db = 0, 0
		for x, y in zip(X, Y):
			dw += grad_w(w, b, x, y)
			db += grad_b(w, b, x, y)

			num_points_seen += 1
			# if condition satisfied , means one batch parsed completely
			if num_points_seen % batch_size == 0:
				w = w - lr * dw
				b = b - lr * db

				# resetting parameters
				dw, db = 0, 0