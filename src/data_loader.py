import os
import cv2

def load_images_from_folder(folder_path, accepted_exts={'.jpg', '.jpeg', '.png', '.bmp'}) -> list:
    images = []

    # Check if the folder exists in the data folder
    data_folder = os.path.join("data", folder_path)
    if os.path.exists(data_folder):
        folder_path = data_folder
    elif not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path} or data/{folder_path}")

    for filename in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in accepted_exts:
            full_path = os.path.join(folder_path, filename)
            image = cv2.imread(full_path)
            if image is None:
                print(f"[WARNING] Skipping unreadable image: {filename}")
            else:
                images.append(image)

    if not images:
        raise ValueError(f"No valid images found in {folder_path}")

    print(f"[INFO] Loaded {len(images)} images from '{folder_path}'")
    return images


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    images = load_images_from_folder(input("Enter folder name: "))[:18]
    num_images = 18
    cols = 3
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten() if num_images > 1 else [axes]

    for i, img in enumerate(images):
        if i >= len(axes):
            break

        # OpenCV loads images in BGR, convert to RGB for display
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(rgb_img)
        axes[i].set_title(f"Image {i+1}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
