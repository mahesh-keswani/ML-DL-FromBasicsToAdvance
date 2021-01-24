def do_momentum_gradient_descent(init_w, init_b, lr, max_epochs, gamma, X, Y):
	w, b, lr, max_epochs =  2, 2, lr, max_epochs
	prev_w_update, prev_b_update, gamma = 0, 0, gamma

	for i in range(max_epochs):
		dw, db = 0, 0
		for x, y in zip(X, Y):
			dw += grad_w(w, b, x, y)
			db += grad_b(w, b, x, y)

		
		update_w = gamma * prev_w_update + lr * dw
		update_b = gamma * prev_b_update + lr * db

		# finally making update
		w = w - update_w
		b = b - update_b

		prev_w_update = w
		prev_b_update = b