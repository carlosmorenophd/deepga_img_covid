from preprocessing.dataset_img import Folder_Image_to_MNIST

def run():
    print("Run")
    to_mnist =Folder_Image_to_MNIST()
    to_mnist.convert_folder_to_mnist(root_folder="data_img_test/Entrenamiento",)
    to_mnist.save_file(path_save="test_training_covid_img")

    to_mnist.convert_folder_to_mnist(root_folder="data_img_test/Test",)
    to_mnist.save_file(path_save="test_testing_covid_img")
    print("end")

if __name__ == '__main__':
    run()
