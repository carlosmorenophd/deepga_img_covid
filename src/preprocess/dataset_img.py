import cv2
import numpy as np
import os
import h5py
from torch.utils.data import Dataset
from PIL import Image


class FolderImageToDatasetH5:

    def __init__(
        self,
        new_pixel_img: tuple = (180, 180),
        base_path: str = "",
        is_debug: bool = False,
        suffix_save: str = "_build",
        prefix_save: str = "dataset_",
    ) -> None:
        self.is_debug = is_debug
        if self.is_debug:
            print(f"Path source {os.getcwd()}")
        self.root_folders = []
        self.new_pixel_img = new_pixel_img
        self.images = []
        self.labels = []
        self.base_path = base_path
        self.suffix_save = suffix_save
        self.prefix_save = prefix_save

    def convert_image_to_Dataset(self, image_path, label):
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.new_pixel_img)
        return img, label

    def clean_images(self):
        self.images = []
        self.labels = []
    
    def __len__(self):
        return len(self.labels)

    def convert_folder_to_dataset(self, root_folder: str, append_images: bool = False):
        path = root_folder
        if self.base_path != "":
            path = os.path.join(self.base_path, root_folder)
        self.root_folders.append(path)
        if self.is_debug:
            print (f"Path to get images {path}")
        images = []
        labels = []
        if not append_images:
            self.clean_images()
        list_main_folder = os.listdir(path)
        sub_folders = [
            element
            for element in list_main_folder
            if os.path.isdir(os.path.join(path, element))
            and not element.startswith(".")
        ]
        for sub_folder in sub_folders:
            path_sub_folder = os.path.join(path, sub_folder)
            list_images = [
                image
                for image in os.listdir(path_sub_folder)
                if image.endswith((".jpg", ".jpeg", ".png", ".gif"))
            ]
            for filename in list_images:
                image_path = os.path.join(path, sub_folder, filename)
                label = int(sub_folder)
                img_flat, label = self.convert_image_to_Dataset(image_path, label)
                images.append(img_flat)
                labels.append(label)
        self.images = np.array(images)
        self.labels = np.array(labels)

    def save_file(self, file_name: str = "", overwrite: bool = True):
        save_name = f"{self.prefix_save}{file_name}{self.suffix_save}{'.h5'}"
        if self.base_path != "":
            save = os.path.join(self.base_path, save_name)
        if self.is_debug:
            print (f" File to save h5 {save}")
        if overwrite and os.path.isfile(save):
            os.remove(save)
        with h5py.File(save, "w") as f:
            f.create_dataset("images", data=self.images)
            f.create_dataset("labels", data=self.labels)
        return save_name


class CustomDatasetH5(Dataset):
    def __init__(self, base_dir, h5_file, is_debug: bool = False, transform=None):
        self.is_debug = is_debug
        path = os.path.join(base_dir, h5_file)
        if self.is_debug:
            print(os.getcwd())
            print(path)
        self.h5_file = h5py.File(path, "r")
        self.transform = transform

    def __len__(self):
        return len(self.h5_file["images"])

    def __getitem__(self, idx):
        image = self.h5_file["images"][idx]
        label = self.h5_file["labels"][idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class FolderImagesCsv:
    def __init__(self) -> None:
        self.dataset = None

    def __len__(self) -> tuple[int, int]:
        return self.dataset.shape

    def to_csv(self, save_path: str) -> None:
        np.savetxt(save_path, self.dataset, delimiter=",")

    def add_images_from_folder(
        self,
        main_folder: str,
        size: list[int, int],
        normalize: int = 1,
    ) -> None:
        list_main_folder = os.listdir(main_folder)
        sub_folders = [
            element
            for element in list_main_folder
            if os.path.isdir(os.path.join(main_folder, element))
            and not element.startswith(".")
        ]
        for sub_folder in sub_folders:
            path_sub_folder = os.path.join(main_folder, sub_folder)
            images = [
                image
                for image in os.listdir(path_sub_folder)
                if image.endswith((".jpg", ".jpeg", ".png", ".gif"))
            ]
            for image in images:
                path_image = os.path.join(path_sub_folder, image)
                image_pillow = Image.open(path_image)
                array_image = np.array(image_pillow.resize(size=size).convert("L"))
                image_normalized = array_image / 255.0
                number = int(sub_folder)
                if number > -1:
                    vector = np.append(image_normalized.ravel(), number)
                if self.dataset is None:
                    self.dataset = vector
                else:
                    self.dataset = np.vstack((self.dataset, vector))
