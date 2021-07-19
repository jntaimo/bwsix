
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import randint
x = []
y = []

fig, ax = plt.subplots()

def basic_animate(i):
    pt = randint(1,9)
    x.append(i)
    y.append(pt)

    ax.clear()
    ax.plot(x,y)
    ax.set_xlim([0,20])
    ax.set_ylim([0,10])

    
ani = FuncAnimation(fig, basic_animate, frames=200, interval=20, repeat=False)
plt.show()