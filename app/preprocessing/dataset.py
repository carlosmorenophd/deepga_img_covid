import os
from PIL import Image
import numpy as np


class DatasetImages():
    def __init__(self) -> None:
        self.dataset = None

    def __len__(self) -> tuple[int, int]:
        return self.dataset.shape

    def to_csv(self, save_path: str) -> None:
        np.savetxt(save_path, self.dataset, delimiter=',')

    def add_images_from_folder(
            self,
            main_folder: str,
            size: list[int, int],
            normalize: int = 1,
    ) -> None:
        list_main_folder = os.listdir(main_folder)
        sub_folders = [element for element in list_main_folder if os.path.isdir(
            os.path.join(main_folder, element))]
        for sub_folder in sub_folders:
            path_sub_folder = os.path.join(main_folder, sub_folder)
            images = [image for image in os.listdir(
                path_sub_folder) if image.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            for image in images:
                path_image = os.path.join(path_sub_folder, image)
                image_pillow = Image.open(path_image)
                array_image = np.array(image_pillow.resize(size=size).convert('L'))
                image_normalized = array_image / 255.0
                vector = np.append(image_normalized.ravel(), int(sub_folder))
                if self.dataset is None:
                    self.dataset = vector
                else:
                    self.dataset = np.vstack((self.dataset, vector))
