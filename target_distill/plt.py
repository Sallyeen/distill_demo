import numpy as np
import matplotlib.pyplot as plt

logits = np.array([-5, 2, 7, 9])

softmax_1 = np.exp(logits) / sum(np.exp(logits))
plt.plot(softmax_1, label='T=1')

T = 3
softmax_2 = np.exp(logits/T) / sum(np.exp(logits/T))
plt.plot(softmax_2, label='T=3')

T = 5
softmax_3 = np.exp(logits/T) / sum(np.exp(logits/T))
plt.plot(softmax_3, label='T=5')

T = 10
softmax_4 = np.exp(logits/T) / sum(np.exp(logits/T))
plt.plot(softmax_4, label='T=10')

T = 100
softmax_5 = np.exp(logits/T) / sum(np.exp(logits/T))
plt.plot(softmax_5, label='T=100')

plt.xticks(np.arange(4), ['cat', 'dog', 'donkey', 'horse'])
plt.legend()
plt.show()