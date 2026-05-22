import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

# Load Image
img = Image.open("image/puzzle_template.jpg").convert("L").resize((600,450))
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

k = gaussian_kernel(size = 11, sigma = 1.8)
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
T = np.arctan2(Iy,Ix)

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
T_high = 0.16
T_low = T_high * 0.35
strong = M_norm >= T_high
weak = (M_norm >= T_low) & (M_norm < T_high) 
result = np.zeros_like(M_norm)
result[strong] = 1.0
result[weak] = 0.5

# Edge linking - keep any weak pixel in the same connected component as a strong pixel
def hysteresis(strong, weak):
    combined = strong | weak
    labeled, _ = ndi.label(combined, structure=np.ones((3, 3)))
    strong_labels = np.unique(labeled[strong])
    strong_labels = strong_labels[strong_labels != 0]
    return np.isin(labeled, strong_labels).astype(np.uint8)
edges = hysteresis(strong, weak)

# Size filtering - remove components too small to be state borders
min_size = 80
labeled, _ = ndi.label(edges, structure=np.ones((3, 3)))
component_sizes = np.bincount(labeled.ravel())
component_sizes[0] = 0
edges = (component_sizes[labeled] >= min_size).astype(np.uint8)

# Result Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0,0].imshow(I, cmap='gray')
axes[0,0].set_title('Original Grayscale')
axes[0,0].axis('off')

axes[0,1].imshow(S, cmap='gray')
axes[0,1].set_title('Gaussian Smoothed')
axes[0,1].axis('off')

axes[0,2].imshow(M / M.max(), cmap='gray')
axes[0,2].set_title('Gradient Magnitude')
axes[0,2].axis('off')

axes[1,0].imshow(NMS / NMS.max(), cmap='gray')
axes[1,0].set_title('Non-Maxima Suppression')
axes[1,0].axis('off')

axes[1,1].imshow(result, cmap='gray')
axes[1,1].set_title('Double Threshold')
axes[1,1].axis('off')

axes[1,2].imshow(edges, cmap='gray')
axes[1,2].set_title('Final Canny Edges')
axes[1,2].axis('off')

plt.suptitle('Canny Edge Detection Pipeline', fontsize=16)
plt.tight_layout()
plt.savefig("pipeline.png", dpi=150)
plt.show()