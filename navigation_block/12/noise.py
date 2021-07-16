# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import cv2

# %% [markdown]
# Demonstrates the Turtles with a single image

# %%
turtle = cv2.imread("baby_turtle.jpg")
turtle_hsv = cv2.cvtColor(turtle, cv2.COLOR_RGB2HSV)
hfilt = cv2.boxFilter(turtle_hsv[:,:,1], cv2.CV_32F, (40,40))
fig, ax = plt.subplots(1,3, figsize = (25,15))
ax[0].imshow(turtle)
ax[0].set_title("Original Image")
ax[1].imshow(turtle_hsv[:,:,1], cmap="hsv")
ax[1].set_title("Saturation channel")
fig.colorbar(cm.ScalarMappable(cmap="hsv"), ax=ax[1])
ax[2].imshow(hfilt, cmap="hsv")
ax[2].set_title("Saturation with filter")
fig.colorbar(cm.ScalarMappable(cmap="hsv"),ax=ax[2])
plt.show()


# %%
fig, ax = plt.subplots()
def animate(i):
    hfilt = cv2.boxFilter(turtle_hsv[:,:,1], cv2.CV_32F, (i+1,i+1))
    ax.clear()
    ax.set_title("Smoothing filter with size: " + str(i+1))
    ax.imshow(hfilt,cmap="hsv")
    ax.set_xlim([0, turtle_hsv.shape[1]])
    ax.set_ylim([0, turtle_hsv.shape[0]])

ani = FuncAnimation(fig, animate, frames=100, interval=200)
plt.show()



# %%
x = []
y = []

fig, ax = plt.subplots()
