import numpy as np
import matplotlib.pyplot as plt


fp = '/local/home/air/code/IGNNK/result_best/k=1_T=24_Z=100/air/result.npz'

res = np.load(fp)

i = 87
t_len = 100
t_offset = 2000

plt.plot(res['pred'][t_offset:t_offset+t_len, i], label='pred')
plt.plot(res['truth'][t_offset:t_offset+t_len, i], label='truth')

plt.title('node: %d' % i)
plt.legend()
plt.show()


print('debug')