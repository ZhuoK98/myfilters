import matplotlib.pyplot as plt


def plot_figure(t, estimate_value, measure_value, true_value, algorithm_name=None, title=None):
    plt.rcParams['font.sans-serif'] = ['SimHei']           	# 设置正常显示中文
    plt.rcParams['axes.unicode_minus'] = False              # 正常显示坐标轴负号
    plt.plot(t, estimate_value, label='预测优化值——{}'.format(algorithm_name))
    plt.plot(t, measure_value, "r--", label='测量值')
    plt.plot(t, true_value, 'black', label='真实值')
    plt.xlabel("时间")                                     	# 设置X轴的名字
    plt.ylabel("位移")                                     	# 设置Y轴的名字
    plt.title("{}".format(title)) 								# 设置标题
    plt.legend()                                           	 # 设置图例
    plt.show()                                             	 # 显示图表

