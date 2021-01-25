def line_search(init_w = None, init_b = None, X, Y, lr, max_epochs, lr_s, cost_fn):
	# assuming parameters as scaler
	# lr_s is a list of learning_rates e.g lr_s = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

	w, b, lr_s = 2, 2, lr_s

	for i in range(max_epochs):
		dw, db = 0, 0
		for x, y in zip(X, Y):
			dw += grad_w(w, b, x, y)
			db += grad_b(w, b, x, y)

		# some large error
		min_error = 9999
		best_w, best_b = w, b

		for lr in lr_s:
			temp_w = w - lr * dw
			temp_b = w - lr * db

			error = cost_fn(temp_w, temp_b)

			# use the learning rate whichever is best for the current position in space
			# this also leads to more computation resources
			if error < min_error:
				best_w = temp_w
				best_b = temp_b
				min_error = error

		w = best_w
		b = best_b