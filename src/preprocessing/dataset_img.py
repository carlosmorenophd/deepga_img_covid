import cv2
import numpy as np
import os
import h5py
from torch.utils.data import Dataset


class Folder_Image_to_Dataset ():

    def __init__(self, new_pixel_img: tuple = (180, 180)) -> None:
        self.root_folders = []
        self.new_pixel_img = new_pixel_img
        self.images = []
        self.labels = []

    def convert_image_to_Dataset(self, image_path, label):
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.new_pixel_img)
        return img, label

    def clean_images(self):
        self.images = []
        self.labels = []

    def convert_folder_to_dataset(self, root_folder: str, append_images: bool = False):
        self.root_folders.append(root_folder)
        images = []
        labels = []
        if not append_images:
            self.clean_images()
        list_main_folder = os.listdir(root_folder)
        sub_folders = [element for element in list_main_folder if os.path.isdir(
            os.path.join(root_folder, element)) and not element.startswith('.')]
        for sub_folder in sub_folders:
            path_sub_folder = os.path.join(root_folder, sub_folder)
            list_images = [image for image in os.listdir(
                path_sub_folder) if image.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            for filename in list_images:
                image_path = os.path.join(
                    root_folder, sub_folder, filename)
                label = int(sub_folder)
                img_flat, label = self.convert_image_to_mnist(
                    image_path, label)
                images.append(img_flat)
                labels.append(label)
        self.images = np.array(images)
        self.labels = np.array(labels)

    def save_file(self, path_save: str = 'mnist_file'):
        save = f"{ path_save}{'.h5'}"
        with h5py.File(save, 'w') as f:
            f.create_dataset('images', data=self.images)
            f.create_dataset('labels', data=self.labels)


class CustomDataset(Dataset):
    def __init__(self, base_dir, h5_file, is_debug: bool = False, transform=None):
        self.is_debug = is_debug
        path = os.path.join(base_dir, h5_file)
        if(self.is_debug):
            print(os.getcwd())
            print(path)
        self.h5_file = h5py.File(path, 'r')
        self.transform = transform

    def __len__(self):
        return len(self.h5_file['images'])

    def __getitem__(self, idx):
        image = self.h5_file['images'][idx]
        label = self.h5_file['labels'][idx]
        if self.transform:
            image = self.transform(image)
        return image, label
