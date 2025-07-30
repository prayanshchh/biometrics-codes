import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_crossing_number(neighbors):
    return sum(abs(neighbors[i] - neighbors[i + 1]) for i in range(8)) // 2

def detect_minutiae(thinned_img):
    minutiae_img = cv2.cvtColor(thinned_img, cv2.COLOR_GRAY2BGR)
    height, width = thinned_img.shape

    # Define 8-neighborhood positions
    offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1),
               (1, 0), (1, -1), (0, -1), (-1, -1)]

    ridge_endings = []
    bifurcations = []

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if thinned_img[y, x] == 255:
                # Collect 8 neighbors
                neighbors = [1 if thinned_img[y + dy, x + dx] == 255 else 0 for dx, dy in offsets]
                neighbors.append(neighbors[0])  # wrap around
                cn = get_crossing_number(neighbors)

                if cn == 1:
                    ridge_endings.append((x, y))
                    cv2.circle(minutiae_img, (x, y), 2, (255, 0, 0), 1)  # blue dot
                elif cn == 3:
                    bifurcations.append((x, y))
                    cv2.circle(minutiae_img, (x, y), 2, (0, 255, 255), 1)  # yellow dot

    return minutiae_img, ridge_endings, bifurcations

# ---------- STEP 1: Load and Grayscale ----------
image_path = "/home/prayansh-chhablani/fingerprint-aadu.jpeg"
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------- STEP 2: CLAHE Contrast Enhancement ----------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

# ---------- STEP 3: Gaussian Blur for Smoothing ----------
blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

# ---------- STEP 4: Otsu Thresholding ----------
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# ---------- STEP 5: Morphological Cleanup ----------
kernel = np.ones((3, 3), np.uint8)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

# ---------- STEP 6: Thinning ----------
def morphological_thinning(img):
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, open_img)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel

thinned = morphological_thinning(opened.copy())

# ---------- STEP 7: Pattern Detection ----------
contours, _ = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
pattern_img = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 200:
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(pattern_img, ellipse, (0, 255, 0), 1)
            cv2.putText(pattern_img, 'Oval/Loop', (int(ellipse[0][0]), int(ellipse[0][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        hull = cv2.convexHull(cnt, returnPoints=False)
        if hull is not None and len(hull) > 3:
            defects = cv2.convexityDefects(cnt, hull)
            if defects is not None and len(defects) >= 2:
                cv2.drawContours(pattern_img, [cnt], -1, (0, 0, 255), 1)
                cv2.putText(pattern_img, 'Delta', tuple(cnt[0][0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

# ---------- STEP 8: Show All Stages ----------
titles = ['Original', 'Grayscale + CLAHE', 'Gaussian Blurred', 'Otsu Binary',
          'Morph Opened', 'Thinned', 'Pattern Detected']
images = [img, enhanced, blurred, binary, opened, thinned, pattern_img]

plt.figure(figsize=(18, 10))
for i in range(len(images)):
    plt.subplot(2, 4, i+1)
    cmap = 'gray' if len(images[i].shape) == 2 else None
    plt.imshow(images[i], cmap=cmap)
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig('fingerprint_processing_steps.png')

minutiae_img, ridge_endings, bifurcations = detect_minutiae(thinned)

print(f"Ridge endings: {len(ridge_endings)}")
print(f"Bifurcations: {len(bifurcations)}")

plt.figure(figsize=(6, 6))
plt.imshow(minutiae_img)
plt.title("Minutiae Points")
plt.axis('off')
plt.tight_layout()
plt.savefig("minutiae_points.png")
