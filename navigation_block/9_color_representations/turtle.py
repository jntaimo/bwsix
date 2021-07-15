# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2


# %%
shark = cv2.imread("baby_shark.jpg")
turtle = cv2.imread("baby_turtle.jpg")


# %%
red_turtle = turtle[:,:,2]
green_turtle = turtle[:,:,1]
blue_turtle = turtle[:,:,0]


# %%
fig, ax = plt.subplots(1,3, figsize = (15, 10))
ax[0].imshow(red_turtle, cmap="Reds")
ax[1].imshow(green_turtle, cmap="Greens")
ax[2].imshow(blue_turtle, cmap="Blues")
plt.show()


# %%
hsv_turtle = cv2.cvtColor(turtle, cv2.COLOR_RGB2HSV )
h_turtle = hsv_turtle[:,:,0]
s_turtle = hsv_turtle[:,:,1]
v_turtle = hsv_turtle[:,:,2]
fig,ax = plt.subplots(1,3, figsize = (15,10))
ax[0].imshow(h_turtle, cmap="Reds")
ax[1].imshow(s_turtle, cmap="Greens")
ax[2].imshow(v_turtle, cmap="Blues")
fig.colorbar(cm.ScalarMappable())
plt.show()



# %%
