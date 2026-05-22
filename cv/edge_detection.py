import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load Image 
img = Image.open("image/puzzle_template.jpg").convert("L")
I = np.array(img, dtype=float)

# Gaussian Smoothing

# Image Gradient 

# Non-maxima Suppression

# Double Thresholding

# Edge linking 

# Result Visualization 
plt.imshow(I, cmap='gray')
plt.title('Loaded Grayscale Image')
plt.axis('off')
plt.show()