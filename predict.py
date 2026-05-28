import cv2
import numpy as np
import pickle

# Load dataset
with open("features/orb_features.pkl", "rb") as f:
    feature_database = pickle.load(f)

orb = cv2.ORB_create(nfeatures=1000)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

QUERY_IMAGE = "QUERY/query.png"

img = cv2.imread(QUERY_IMAGE, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Failed to load query image")
    exit()

# ORB features for query
kp1, des1 = orb.detectAndCompute(img, None)

if des1 is None:
    print("No features found in query image")
    exit()

best_match = None
best_score = 0  # higher is better in ORB matching

for sample in feature_database:

    des2 = sample["descriptors"]

    if des2 is None:
        continue

    # KNN matching
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []

for pair in matches:

    if len(pair) != 2:
        continue

    m, n = pair

    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
    score = len(good_matches)

    if score > best_score:
        best_score = score
        best_match = sample

print("\n============")
print("BEST MATCH:")
print("Pokemon:", best_match["pokemon"])
print("File:", best_match["file"])
print("Match Score:", best_score)
print("============")