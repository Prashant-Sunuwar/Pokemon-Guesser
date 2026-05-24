import os
import cv2

dataset_path = "datasets/images"
output_path = "outlines"

os.makedirs(output_path, exist_ok=True)

for pokemon in os.listdir(dataset_path):

    pokemon_folder = os.path.join(
        dataset_path,
        pokemon
    )

    if not os.path.isdir(pokemon_folder):
        continue

    save_folder = os.path.join(
        output_path,
        pokemon
    )

    os.makedirs(save_folder, exist_ok=True)

    for image_name in os.listdir(pokemon_folder):

        if not image_name.endswith(
            (".jpg", ".png", ".jpeg")
        ):
            continue

        image_path = os.path.join(
            pokemon_folder,
            image_name
        )

        img = cv2.imread(image_path)

        if img is None:
            continue

        # Resize
        img = cv2.resize(img, (256,256))

        # Grayscale
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        # Blur
        blur = cv2.GaussianBlur(
            gray,
            (5,5),
            0
        )

        # Edge Detection
        edges = cv2.Canny(
            blur,
            100,
            200
        )

        save_path = os.path.join(
            save_folder,
            image_name
        )

        cv2.imwrite(save_path, edges)

print("Outline dataset created!")