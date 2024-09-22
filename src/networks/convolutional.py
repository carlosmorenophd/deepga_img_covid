import torch.nn as nn
import torch.nn.functional as F  # adds some efficiency
import math


class ConvolutionalNetworkLayers(nn.Module):
    def __init__(
        self,
        number_pixel_image: int,
        number_of_class: int,
        drop_out_probability: float = 0.5,
        number_band_image: int = 3,
        convolution_filter: tuple = [10, 5],
        layers: tuple = [200, 100, 50],
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(number_band_image, convolution_filter[0], 5, 1)
        self.conv2 = nn.Conv2d(convolution_filter[0], convolution_filter[1], 3, 1)
        output_convolution_neurons = math.floor(
            (((number_pixel_image - 2) / 2) - 2) / 2
        )
        self.exit_convolution_neurons = (
            output_convolution_neurons
            * output_convolution_neurons
            * convolution_filter[1]
        )
        layer_list = []
        n_in = self.exit_convolution_neurons
        for i in layers:
            layer_list.append(nn.Linear(n_in, i))
            layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.BatchNorm1d(i))
            layer_list.append(nn.Dropout(drop_out_probability))
            n_in = i
        layer_list.append(nn.Linear(layers[-1], number_of_class))
        self.layers = nn.Sequential(*layer_list)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, self.exit_convolution_neurons)
        # Set up model layers
        X = self.layers(X)
        return F.log_softmax(X, dim=1)
