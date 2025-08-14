import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# ---------- Tunables ----------
# Color thresholds (RGB) for bright lane-like pixels (adjust to your image)
RED_T, GREEN_T, BLUE_T = 170, 170, 170     # try 170–220 for daytime, lower for dusk

# Canny params
LOW_T, HIGH_T = 50, 150                    # lower for faint lanes, higher for noise
GAUSS_KSIZE = 5

# Hough params
RHO = 1
THETA = np.pi / 180
HOUGH_THRESH = 20
MIN_LINE_LEN = 20
MAX_LINE_GAP = 10

# ROI vertices as fractions of image size (trapezoid is often better than triangle)
# (x_frac, y_frac) with y=0 top, y=1 bottom
ROI = np.array([
    [0.10, 0.98],   # left-bottom
    [0.45, 0.60],   # left-top
    [0.55, 0.60],   # right-top
    [0.95, 0.98],   # right-bottom
], dtype=np.float32)
# --------------------------------

# Read image (matplotlib returns RGB)
image = mpimg.imread('road1.jpg').astype(np.uint8)
h, w = image.shape[:2]

# Build ROI polygon in pixels
roi_px = (ROI * np.array([w, h])).astype(np.int32)
roi_px = roi_px.reshape(1, -1, 2)

# --- Color mask (bright-ish lanes) ---
# Keep pixels that are above thresholds in all channels
color_mask = (
    (image[:, :, 0] >= RED_T) &
    (image[:, :, 1] >= GREEN_T) &
    (image[:, :, 2] >= BLUE_T)
).astype(np.uint8) * 255

# --- ROI mask ---
mask_roi = np.zeros((h, w), dtype=np.uint8)
cv2.fillPoly(mask_roi, roi_px, 255)

# --- Grayscale + Canny edges ---
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gray, (GAUSS_KSIZE, GAUSS_KSIZE), 0)
edges = cv2.Canny(blur, LOW_T, HIGH_T)

# --- Fuse masks: ROI ∧ (edges ∧ color OR edges) ---
# Start strict: edges & color within ROI (very clean). If too sparse, OR with plain edges in ROI.
strict_edges = cv2.bitwise_and(edges, color_mask)
masked_edges = cv2.bitwise_and(strict_edges, mask_roi)
if np.count_nonzero(masked_edges) < 500:  # fallback if too few
    masked_edges = cv2.bitwise_and(edges, mask_roi)

# --- Hough transform ---
lines = cv2.HoughLinesP(masked_edges, RHO, THETA, HOUGH_THRESH,
                        minLineLength=MIN_LINE_LEN, maxLineGap=MAX_LINE_GAP)

# --- Draw lines ---
line_image = np.zeros_like(image)
if lines is not None:
    for l in lines:
        x1, y1, x2, y2 = l[0]
        # OpenCV uses BGR; for RED in RGB display, use (0,0,255)
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 8)

# Overlay on original
overlay = cv2.addWeighted(image, 0.8, line_image, 1.0, 0)

# --- Visualize debug panels ---
fig, axs = plt.subplots(1, 4, figsize=(20, 6))
axs[0].set_title('Original + ROI')
axs[0].imshow(image)
# draw ROI outline
roi_closed = np.vstack([roi_px[0], roi_px[0][0]])
axs[0].plot(roi_closed[:, 0], roi_closed[:, 1], 'y--', lw=3)

axs[1].set_title('Color mask (within ROI)')
axs[1].imshow(cv2.bitwise_and(color_mask, mask_roi), cmap='gray')

axs[2].set_title('Masked edges')
axs[2].imshow(masked_edges, cmap='gray')

axs[3].set_title('Detected lanes')
axs[3].imshow(overlay)
for a in axs: a.axis('off')
plt.tight_layout()
plt.show()
