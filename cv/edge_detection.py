"""Edge detection module implementing Canny edge detection algorithm."""
from PIL import Image

import matplotlib.pyplot as plt

import numpy as np

from scipy import ndimage as ndi

img = Image.open('image/puzzle_template.jpg').convert('L').resize((600, 450))
img_gray = np.array(img, dtype=float)


def gaussian_kernel(size, sigma):
    """Generate a Gaussian kernel for image smoothing.

    Args:
        size: The size of the kernel (must be odd).
        sigma: The standard deviation of the Gaussian distribution.

    Returns:
        A normalized Gaussian kernel as a 2D numpy array.
    """
    k = size // 2
    y, x = np.mgrid[-k:k+1, -k:k+1]
    G = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return G / G.sum()


def conv2d(image, kernel):
    """Perform 2D convolution on an image using a given kernel.

    Args:
        image: Input image as a 2D numpy array.
        kernel: Convolution kernel as a 2D numpy array.

    Returns:
        Convolved image as a 2D numpy array with the same shape as input.
    """
    k = kernel.shape[0] // 2
    padded = np.pad(image, k, mode='reflect')
    output = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            patch = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            output[i, j] = np.sum(patch * kernel)
    return output


k = gaussian_kernel(size=11, sigma=1.8)
S = conv2d(img_gray, k)

Gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=float)
Gy = np.array([[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]], dtype=float)
Ix = conv2d(S, Gx)
Iy = conv2d(S, Gy)
M = np.sqrt(Ix**2 + Iy**2)
T = np.arctan2(Iy, Ix)


def non_maxima_suppression(mag, ang):
    """Perform non-maxima suppression on gradient magnitude image.

    Args:
        mag: Gradient magnitude as a 2D numpy array.
        ang: Gradient angle as a 2D numpy array in radians.

    Returns:
        Non-maxima suppressed image as a 2D numpy array.
    """
    rows, cols = mag.shape
    nms = np.zeros_like(mag)
    angle = np.degrees(ang) % 180

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            a = angle[i, j]

            if (0 <= a < 22.5) or (157.5 <= a < 180):
                p, q = mag[i, j+1], mag[i, j-1]
            elif 22.5 <= a < 67.5:
                p, q = mag[i+1, j-1], mag[i-1, j+1]
            elif 67.5 <= a < 112.5:
                p, q = mag[i+1, j], mag[i-1, j]
            else:
                p, q = mag[i-1, j-1], mag[i+1, j+1]

            if mag[i, j] >= p and mag[i, j] >= q:
                nms[i, j] = mag[i, j]

    return nms


NMS = non_maxima_suppression(M, T)

M_norm = NMS / NMS.max()
T_high = 0.16
T_low = T_high * 0.35
strong = M_norm >= T_high
weak = (M_norm >= T_low) & (M_norm < T_high)
result = np.zeros_like(M_norm)
result[strong] = 1.0
result[weak] = 0.5


def hysteresis(strong_mask, weak_mask):
    """Apply hysteresis thresholding to connect weak edges to strong edges.

    Args:
        strong_mask: Boolean mask of strong edges as a 2D numpy array.
        weak_mask: Boolean mask of weak edges as a 2D numpy array.

    Returns:
        Binary image with weak edges connected to strong edges as a 2D numpy array.
    """
    combined = strong_mask | weak_mask
    labeled, _ = ndi.label(combined, structure=np.ones((3, 3)))
    strong_labels = np.unique(labeled[strong_mask])
    strong_labels = strong_labels[strong_labels != 0]
    return np.isin(labeled, strong_labels).astype(np.uint8)


edges = hysteresis(strong, weak)

min_size = 80
labeled, _ = ndi.label(edges, structure=np.ones((3, 3)))
component_sizes = np.bincount(labeled.ravel())
component_sizes[0] = 0
edges = (component_sizes[labeled] >= min_size).astype(np.uint8)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(img_gray, cmap='gray')
axes[0, 0].set_title('Original Grayscale')
axes[0, 0].axis('off')

axes[0, 1].imshow(S, cmap='gray')
axes[0, 1].set_title('Gaussian Smoothed')
axes[0, 1].axis('off')

axes[0, 2].imshow(M / M.max(), cmap='gray')
axes[0, 2].set_title('Gradient Magnitude')
axes[0, 2].axis('off')

axes[1, 0].imshow(NMS / NMS.max(), cmap='gray')
axes[1, 0].set_title('Non-Maxima Suppression')
axes[1, 0].axis('off')

axes[1, 1].imshow(result, cmap='gray')
axes[1, 1].set_title('Double Threshold')
axes[1, 1].axis('off')

axes[1, 2].imshow(edges, cmap='gray')
axes[1, 2].set_title('Final Canny Edges')
axes[1, 2].axis('off')

plt.suptitle('Canny Edge Detection Pipeline', fontsize=16)
plt.tight_layout()
plt.savefig('pipeline.png', dpi=150)
plt.show()
