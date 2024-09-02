import cv2
import numpy as np
import os
import h5py


class Folder_Image_to_MNIST ():

    def __init__(self, new_pixel_img: tuple = (28, 28)) -> None:
        self.root_folders = []
        self.new_pixel_img = new_pixel_img
        self.images = []
        self.labels = []

    def convert_image_to_mnist(self, image_path, label):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.new_pixel_img)
        img = img / 255.0
        img_flat = img.flatten()
        return img_flat, label

    def clean_images(self):
        self.images = []
        self.labels = []

    def convert_folder_to_mnist(self, root_folder: str, append_images: bool = False):
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
