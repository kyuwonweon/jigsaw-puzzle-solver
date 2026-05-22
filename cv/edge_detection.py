import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load Image 
img = Image.open("image/puzzle_template.jpg").convert("L")
I = np.array(img, dtype=float)

# Gaussian Smoothing
def gaussian_kernel(size, sigma):
    k = size // 2
    y,x = np.mgrid[-k:k+1, -k:k+1]
    G = np.exp(-(x**2 + y**2)/(2*sigma**2))
    G_norm = G/G.sum()
    return G_norm

def conv2d(img, kernel):
    k = kernel.shape[0]//2
    padded = np.pad(img, k, mode ="reflect")
    output = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):     
            patch = padded[i:i+kernel.shape[0],
                           j:j+kernel.shape[1]]
            output[i,j] = np.sum(patch*kernel)
    return output

k = gaussian_kernel(size = 5, sigma = 2)
S = conv2d(I, k)

# Image Gradient

# Non-maxima Suppression

# Double Thresholding

# Edge linking 

# Result Visualization 
plt.imshow(S, cmap='gray')
plt.title('Loaded Grayscale Image')
plt.axis('off')
plt.show()