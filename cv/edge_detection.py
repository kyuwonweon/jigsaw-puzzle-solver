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
Gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=float)
Gy = np.array([[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]], dtype=float)
Ix = conv2d(S, Gx)
Iy = conv2d(S, Gy)
M = np.sqrt(Ix**2 + Iy**2)
T = np.arctan(Iy,Ix)

# Non-maxima Suppression
def non_maxima_suppression(M, T):
    rows, cols = M.shape
    NMS = np.zeros_like(M)
    angle = np.degrees(T) % 180

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            a = angle[i, j]

            # Left and right
            if (0 <= a < 22.5) or (157.5 <= a < 180):
                p, q = M[i, j+1], M[i, j-1]
            # Diagonal
            elif 22.5 <= a < 67.5:
                p, q = M[i+1, j-1], M[i-1, j+1]
            # Up and down
            elif 67.5 <= a < 112.5:
                p, q = M[i+1, j], M[i-1, j]
            # The other diagonal
            else:
                p, q = M[i-1, j-1], M[i+1, j+1]

            if M[i, j] >= p and M[i, j] >= q:
                NMS[i, j] = M[i, j]

    return NMS


NMS = non_maxima_suppression(M, T)

# Double Thresholding
M_norm = NMS / NMS.max()
T_high = .15
T_low = T_high*.5
strong = M_norm >= T_high
weak = (M_norm >= T_low) & (M_norm < T_high) 
result = np.zeros_like(M_norm)
result[strong] = 1.0
result[weak] = 0.5

# Edge linking 

# Result Visualization 
plt.imshow(result, cmap='gray')
plt.title('Gradient Magnitude')
plt.axis('off')
plt.show()