import numpy as np

np.random.seed(42)  # 固定随机种子，确保每次生成的随机数据一样
delta_t = 0.1  # 每秒采样一次
end_t = 7  # 时间长度
time_t = end_t * 10  # 采样次数
t = np.arange(0, end_t, delta_t)  # 设置时间长度
v_var = 4  # 测量噪声的方差
v_noise = np.round(np.random.normal(0, v_var, time_t), 2)  # 定义测量噪声
a = 1  # 加速度
s = np.add((1 / 2 * a * t ** 2), v_noise)  # 定义测量的位置
v = a * t  # 定义速度数组

def generate_data():
    X = np.mat([s, v])  # 定义状态矩阵
    U = np.mat([[a], [a]])  # 定义外界对系统作用矩阵
    A = np.mat([[1, delta_t], [0, 1]])  # 定义状态转移矩阵
    B = np.mat([[1 / 2 * (delta_t ** 2), 0], [0, delta_t]])  # 定义输入控制矩阵
    P = np.mat([[1, 0], [0, 1]])  # 定义初始协方差矩阵
    Q = np.mat([[0.1, 0], [0, 0.1]])  # 定义测量噪声协方差矩阵
    H = np.mat([1.0, 0])  # 定义观测矩阵
    R = np.mat([1])  # 定义观测噪声协方差矩阵
    Z = H * X

    return X, U, A, B, P, Q, H, R, Z, t, 1/2*a*t**2

def display_matrix(name:str, matrix):
    print(30 * '-' + "{}".format(name) + 30 * '-')
    print(matrix)
    print(35 * '-' + "end" + 35 * '-')
    print()

if __name__ == '__main__':
    X, U, A, B, P, Q, H, R, Z, t, true_value = generate_data()
    display_matrix("state variable X", X)
    display_matrix("control variable U", U)
    display_matrix("state-transition matrix A", A)
    display_matrix("control matrix B", B)
    display_matrix("covariance matrix P", P)
    display_matrix("covariance matrix of observation Q", Q)
    display_matrix("observation matrix H", H)
    display_matrix("covariance matrix R", R)
    display_matrix("observation variable", Z)
