import cv2
import os
import pickle

CONTOUR_ROOT = "processed/grayscale"  

orb = cv2.ORB_create(nfeatures=1000)

feature_database = []

for pokemon_name in os.listdir(CONTOUR_ROOT):

    folder = os.path.join(CONTOUR_ROOT, pokemon_name)

    if not os.path.isdir(folder):
        continue

    for filename in os.listdir(folder):

        path = os.path.join(folder, filename)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        
        kp, des = orb.detectAndCompute(img, None)

        if des is None:
            continue

        feature_database.append({
            "pokemon": pokemon_name,
            "file": filename,
            "descriptors": des
        })

# Save database
os.makedirs("features", exist_ok=True)

with open("features/orb_features.pkl", "wb") as f:
    pickle.dump(feature_database, f)

print("ORB feature extraction completed!")
print("Total samples:", len(feature_database))