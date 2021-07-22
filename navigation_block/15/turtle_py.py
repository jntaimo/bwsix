# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2

# %% [markdown]
# 1)

# %%

#load the image
img = cv2.imread("baby_turtle.jpg")
#downsample
img = cv2.resize(img,(640,480))

# %% [markdown]
# 2)

# %%
#convert to hsv
imhsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# %% [markdown]
# 3)

# %%
#apply box filter to reduce noise
imhsv = cv2.boxFilter(imhsv, -1, (10, 10))

# %% [markdown]
# 4)

# %%
#plot the three channels
fig, ax = plt.subplots(1, 3, figsize = (15, 5))
for channel_i in range(3):
    ax[channel_i].imshow(imhsv[:,:,channel_i], cmap="hsv")
#plt.show()
#H 45-70
#S 100-170
#V 45-90

# %% [markdown]
# 5)

# %%
#H 45-70
#S 100-170
#V 45-90

# %% [markdown]
# 6/7)

# %%
#making a decision rule
h_min, h_max = 45, 70
s_min, s_max = 100, 190
v_min, v_max = 45, 90

img_thresh_hue = np.logical_and(imhsv[:,:,0] > h_min, imhsv[:,:,0] < h_max)
img_thresh_sat = np.logical_and(imhsv[:,:,1] > s_min, imhsv[:,:,1] < s_max)
img_thresh_val = np.logical_and(imhsv[:,:,2] > v_min, imhsv[:,:,2] < v_max)

img_thresholds = img_thresh_hue, img_thresh_sat, img_thresh_val
# %% [markdown]
# 8)

# %%
hsv = "hue", "saturation", "value"
fig, ax = plt.subplots(3, 3)
#plot the original images on the first row
for i in range(3):
    ax[0][i].imshow(imhsv[:,:,i], cmap="hsv")
    ax[0][i].set_title( hsv[i] + " channel")

#plot the thresholds in the second row
for i in range(3):
    ax[1][i].imshow(img_thresholds[i])
    ax[1][i].set_title(hsv[i] + " threshold")

#flip GBR to RGB to display image
imrgb = np.flip(img, axis=2)
#plot the combined values in the third row
for i in range(3):
    ax[2][i].imshow(imrgb*np.expand_dims(img_thresholds[i], axis=2))
    ax[2][i].set_title(hsv[i] + " threshold applied")


# %% [markdown]
# 9-12)

# %%
img_thresh_hsv = np.logical_and(np.logical_and(img_thresh_hue, img_thresh_sat), img_thresh_val)
fig, ax = plt.subplots(1,2)
plt.title("Combined Channels")
ax[0].imshow(img_thresh_hsv)
ax[1].imshow(imrgb*np.expand_dims(img_thresh_hsv, axis=2))

# %% [markdown]
# 13)

# %%
object_detection_surface = cv2.boxFilter(img_thresh_hsv.astype(int), -1, (50,50), normalize=False)
plt.figure()
plt.imshow(object_detection_surface)
# %%
thresh = 1000
above_thresh = np.argwhere(object_detection_surface > thresh)

# %%
avg_y, avg_x= np.average(above_thresh, axis=0)
# %%
plt.figure()

plt.imshow(imrgb)
plt.plot(avg_x, avg_y, marker="o", color="red")
plt.show()
# %%
