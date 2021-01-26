import numpy as np 

def do_adagrad(init_w=None, init_b=None, X, Y, lr=0.01, eps=1e-8, max_epochs=100, beta1 = 0.9, beta2 = 0.99):
	# assuming parameters as scaler
	w, b, lr = 2, 2, lr
	# for accumulating history of parameters
	v_w, v_b, beta2 = 0, 0, beta2
	# making sure denominator is not zero
	eps = eps
	# for momentum and beta1 for controlling it
	m_w, m_b, beta1 = 0, 0, beta1
	for i in range(max_epochs):
		dw, db = 0, 0

		for x, y in zip(X, Y):
			dw += grad_w(w, b, x, y)
			db += grad_b(w, b, x, y)

		# accumulating the frequency of updates for every parameter
		# controlling history by beta2
		v_w = beta2 * v_w + (1 - beta2) * dw**2
		v_b = beta2 * v_b + (1 - beta2) * db**2

		# gaining momentum
		m_w = beta1 * m_w + (1 - beta1) * dw
		m_b = beta1 * m_b + (1 - beta1) * db
		
		# doing bias correction, so that the value of the gradient
		# stays near the mean of the distribution 
		m_what = m_w / (1 - pow(beta1, i + 1))
		m_bhat = m_b / (1 - pow(beta1, i + 1))

		# decaying (or adapting) the learning rate depending on the frequency of update
		# also adding momentum :)
		w = w - ( lr / np.sqrt(v_w + eps) ) * m_what
		b = b - ( lr / np.sqrt(v_b + eps) ) * m_bhat
		