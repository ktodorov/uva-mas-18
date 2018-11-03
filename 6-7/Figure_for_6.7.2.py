import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-1.2, 1.2, 256, endpoint=True)
C = (1+np.cos(np.pi*X))/2
fig = plt.figure()
ax = fig.gca()
plt.plot(X, C, color='darkblue', linewidth=2.0)
plt.xlim(-2.0, 2.0)
plt.ylim(0, 1.1)
plt.grid()

ax.annotate(r'$\frac{1+cos(\pi*x)}{2}$', xy=(0, 1), xytext=(1, 0.95), fontsize=16,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
            )
plt.fill(X, C, facecolor='blue', alpha=0.5)
plt.ylabel('Probability')
plt.show()