import cv2
import numpy as np
import os

# Dataset folder
input_root = "datasets/images"

# Output folders
gray_root = "processed/grayscale"
edge_root = "processed/edges"
silhouette_root = "processed/silhouettes"
contour_root = "processed/contours"

# Create output folders
os.makedirs(gray_root, exist_ok=True)
os.makedirs(edge_root, exist_ok=True)
os.makedirs(silhouette_root, exist_ok=True)
os.makedirs(contour_root, exist_ok=True)

# Loop through all pokemon folders
for pokemon_name in os.listdir(input_root):

    pokemon_input_path = os.path.join(
        input_root,
        pokemon_name
    )

    if not os.path.isdir(pokemon_input_path):
        continue

    # Create pokemon output folders
    gray_pokemon_path = os.path.join(
        gray_root,
        pokemon_name
    )

    edge_pokemon_path = os.path.join(
        edge_root,
        pokemon_name
    )

    silhouette_pokemon_path = os.path.join(
        silhouette_root,
        pokemon_name
    )

    contour_pokemon_path = os.path.join(
        contour_root,
        pokemon_name
    )

    os.makedirs(gray_pokemon_path, exist_ok=True)
    os.makedirs(edge_pokemon_path, exist_ok=True)
    os.makedirs(silhouette_pokemon_path, exist_ok=True)
    os.makedirs(contour_pokemon_path, exist_ok=True)

    # Process every image
    for filename in os.listdir(pokemon_input_path):

        image_path = os.path.join(
            pokemon_input_path,
            filename
        )

        img = cv2.imread(image_path)

        if img is None:
            print(f"Failed to load {image_path}")
            continue

        # Resize image
        img = cv2.resize(img, (256, 256))

        # Convert to grayscale
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        # Blur image
        blur = cv2.GaussianBlur(
            gray,
            (5,5),
            0
        )

        # Edge detection
        edges = cv2.Canny(
            blur,
            50,
            150
        )

        # Morphological closing
        kernel = np.ones((5,5), np.uint8)

        closed = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel
        )

        # Find contours
        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Create blank images
        silhouette = np.zeros_like(gray)

        contour_image = np.zeros_like(gray)

        # Draw largest contour
        if contours:

            largest_contour = max(
                contours,
                key=cv2.contourArea
            )

            # Filled silhouette
            cv2.drawContours(
                silhouette,
                [largest_contour],
                -1,
                255,
                thickness=cv2.FILLED
            )

            # Contour outline
            cv2.drawContours(
                contour_image,
                [largest_contour],
                -1,
                255,
                thickness=2
            )

        # Save all outputs
        cv2.imwrite(
            os.path.join(
                gray_pokemon_path,
                filename
            ),
            gray
        )

        cv2.imwrite(
            os.path.join(
                edge_pokemon_path,
                filename
            ),
            edges
        )

        cv2.imwrite(
            os.path.join(
                silhouette_pokemon_path,
                filename
            ),
            silhouette
        )

        cv2.imwrite(
            os.path.join(
                contour_pokemon_path,
                filename
            ),
            contour_image
        )

        print(f"Processed: {pokemon_name}/{filename}")

print("All preprocessing completed!")