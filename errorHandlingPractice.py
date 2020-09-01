import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import math as m

x = np.linspace(-m.pi, m.pi, 150)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.arctan(x)

plt.plot(x, y1, label = 'sin(x)')
plt.plot(x, y2, label = 'cos(x)')
plt.plot(x, y3, label = 'arctan(x)')

plt.title("A plot of a trig function")
plt.ylabel("y - values")
plt.xlabel("x - values")
#plt.plot(x, y, color='green', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=12)
plt.legend()
plt.show()