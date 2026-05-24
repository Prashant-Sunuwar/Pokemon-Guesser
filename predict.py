import cv2
import numpy as np
import json
from scipy.spatial.distance import euclidean

# -----------------------------
# LOAD FEATURES DATABASE
# -----------------------------
with open("features/features.json", "r") as f:
    database = json.load(f)

# -----------------------------
# INPUT IMAGE
# -----------------------------
QUERY_IMAGE = "query.png"

img = cv2.imread(QUERY_IMAGE)

img = cv2.resize(img, (256, 256))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5,5), 0)

_, thresh = cv2.threshold(
    blur,
    120,
    255,
    cv2.THRESH_BINARY_INV
)

# Find contours
contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

largest = max(contours, key=cv2.contourArea)

# Compute Hu Moments
moments = cv2.moments(largest)

query_hu = cv2.HuMoments(moments).flatten()

# -----------------------------
# MATCH AGAINST DATASET
# -----------------------------
best_match = None
best_score = float("inf")

for name, features in database.items():

    features = np.array(features)

    score = euclidean(query_hu, features)

    if score < best_score:
        best_score = score
        best_match = name

# -----------------------------
# RESULT
# -----------------------------
print("\nPredicted Pokemon:")
print(best_match)

print("\nSimilarity Score:")
print(best_score)