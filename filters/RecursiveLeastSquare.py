import numpy as np

# def RLS_step(w, P, x, d, lmbd):
#     y = (w.T * x)[0, 0]
#     e = d - y
#     Px = P * x
#     k = Px * (lmbd + x.T * Px).I
#     P = (P - k * x.T * P) / lmbd
#     w = w + e * k
#     return w, P, e

# def my_RLS(x, d, N = 4, lmbd = 0.999, delta = 0.0002):
#     L = min(len(x),len(d))
#     w = np.mat(np.zeros((N, 1)))
#     P = np.mat(np.eye(N)/delta)
#     e = np.zeros(L-N)
#     for k in range(L-N):
#         xk = np.mat(x[k:k+N]).T
#         w, P, e[k] = RLS_step(w, P, xk, d[k], lmbd)
    
#     return np.array(w.T)[0]
   
'''
    Recursive least square filter
'''
class RLS_filter:
    def __init__(self, order, forgetting_factor=0.99, delta=0.0002):
        self.order = order
        self.forgetting_factor = forgetting_factor
        self.theta = np.zeros((order, 1))
        self.P = np.eye(order) / delta

    def update(self, x, d):
        x = np.array(x).reshape(-1, 1)
        d = np.array(d).reshape(-1, 1)

        # Prediction error
        e = d - np.dot(x.T, self.theta)

        # Gain vector
        k = np.dot(self.P, x) / (self.forgetting_factor + np.dot(np.dot(x.T, self.P), x))

        # Update filter coefficients
        self.theta = self.theta + np.dot(k, e)

        # Update covariance matrix
        self.P = (self.P - np.dot(np.dot(k, x.T), self.P)) / self.forgetting_factor

        return np.squeeze(np.dot(x.T, self.theta))
    
    def __call__(self, x, d):
        L = min(len(x), len(d))
        N = L - self.order
        y_est = np.zeros(N)
        for i in range(N):
            y_est[i] = self.update(x[i:i+self.order], d[i])

        return self.theta


'''
    Polynomial fitting using recursive least square
'''
class RLS_polynomialFit(RLS_filter):
    def __call__(self, x, d):
        assert(len(x) == len(d))
        for i, e in enumerate(x):
            xk = [e**m for m in range(self.order)]
            self.update(xk, d[i])

        return self.theta


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    def polynomial(x, theta):
        res = 0
        for i, e in enumerate(theta):
            res = res + e * x ** i
        return res

    def my_func_test(x):
        theta = [0, 0, 4]
        return polynomial(x, theta)
    
    N = 50
    x = [_ for _ in range(N)]
    # noise = np.random.normal(0, 0.4, N)
    # x = x + noise
    d = [my_func_test(_) for _ in range(N)]

    test = RLS_polynomialFit(4)
    theta = test(x, d)
    print(theta)
    est = [polynomial(_, theta) for _ in x]

    plt.plot(d, label='desire')
    plt.plot(est, label='est')
    plt.legend()
    plt.show()


