import torch.nn as nn
import torch
import torch.nn.functional as F


class AWF(nn.Module):
    def __init__(self, num_classes=100, num_tab=1):
        super(AWF, self).__init__()

        self.model = AWF_model(num_classes=num_classes, num_tab=num_tab)

        self.classifier = nn.Linear(in_features=32 * 45, out_features=num_classes)

    def forward(self, x):
        feat = self.model(x)

        x = self.classifier(feat)
        x = x.view(x.size(0), -1)
        return x


class AWF_model(nn.Module):
    def __init__(self, num_classes=100, num_tab=1):
        super(AWF_model, self).__init__()

        # Define the feature extraction part of the network using a sequential container
        self.feature_extraction = nn.Sequential(
            nn.Dropout(p=0.25),  # Dropout layer with a 25% dropout rate for regularization

            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5,
                      stride=1, padding='valid', bias=False),  # First convolutional layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.MaxPool1d(kernel_size=4, padding=0),  # First max pooling layer

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5,
                      stride=1, padding='valid', bias=False),  # Second convolutional layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.MaxPool1d(kernel_size=4, padding=0),  # Second max pooling layer

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5,
                      stride=1, padding='valid', bias=False),  # Third convolutional layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.MaxPool1d(kernel_size=4, padding=0),  # Third max pooling layer
        )

        # Define the classifier part of the network
        self.classifier = nn.Flatten()

    def forward(self, x):
        # Ensure the input tensor has the expected shape
        assert x.shape[-1] == 3000, f"Expected input with 3000 elements, got {x.shape[-1]}"

        # Pass the input through the feature extraction part
        x = self.feature_extraction(x)

        # Pass the output through the classifier
        x = self.classifier(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pool_size, pool_stride, dropout_p, activation):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2  # Calculate padding to keep the output size same as input size
        # Define a convolutional block consisting of two convolutional layers, each followed by batch normalization
        # and activation
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),  # Batch normalization layer
            activation(inplace=True),  # Activation function (e.g., ELU or ReLU)
            nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),  # Batch normalization layer
            activation(inplace=True),  # Activation function
            nn.MaxPool1d(pool_size, pool_stride, padding=0),  # Max pooling layer to downsample the input
            nn.Dropout(p=dropout_p)  # Dropout layer for regularization
        )

    def forward(self, x):
        # Pass the input through the convolutional block
        return self.block(x)


class DF(nn.Module):
    def __init__(self, num_classes, num_tab=1):
        super(DF, self).__init__()

        self.model = DF_model(num_classes=num_classes, num_tab=num_tab)

        filter_num = [32, 64, 128, 256]  # Number of filters for each block
        length_after_extraction = 18  # Length of the feature map after the feature extraction part

        # Define the classifier part of the network
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten the tensor to a vector
            nn.Linear(filter_num[3] * length_after_extraction, 512, bias=False),  # Fully connected layer
            nn.BatchNorm1d(512),  # Batch normalization layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.Dropout(p=0.7),  # Dropout layer for regularization
            nn.Linear(512, 512, bias=False),  # Fully connected layer
            nn.BatchNorm1d(512),  # Batch normalization layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.Dropout(p=0.5),  # Dropout layer for regularization
            nn.Linear(512, num_classes)  # Output layer
        )

    def forward(self, x):
        feat = self.model(x)

        x = self.classifier(feat)
        x = x.view(x.size(0), -1)
        return x


class DF_model(nn.Module):
    def __init__(self, num_classes, num_tab=1):
        super(DF_model, self).__init__()

        # Configuration parameters for the convolutional blocks
        filter_num = [32, 64, 128, 256]  # Number of filters for each block
        kernel_size = 8  # Kernel size for convolutional layers
        conv_stride_size = 1  # Stride size for convolutional layers
        pool_stride_size = 4  # Stride size for max pooling layers
        pool_size = 8  # Kernel size for max pooling layers
        length_after_extraction = 18  # Length of the feature map after the feature extraction part

        # Define the feature extraction part of the network using a sequential container with ConvBlock instances
        self.feature_extraction = nn.Sequential(
            ConvBlock(1, filter_num[0], kernel_size, conv_stride_size, pool_size, pool_stride_size, 0.1, nn.ELU),
            # Block 1
            ConvBlock(filter_num[0], filter_num[1], kernel_size, conv_stride_size, pool_size, pool_stride_size, 0.1,
                      nn.ReLU),  # Block 2
            ConvBlock(filter_num[1], filter_num[2], kernel_size, conv_stride_size, pool_size, pool_stride_size, 0.1,
                      nn.ReLU),  # Block 3
            ConvBlock(filter_num[2], filter_num[3], kernel_size, conv_stride_size, pool_size, pool_stride_size, 0.1,
                      nn.ReLU)  # Block 4
        )

    def forward(self, x):
        # Pass the input through the feature extraction part
        x = self.feature_extraction(x)

        return x


class DilatedBasic1D(nn.Module):
    """
    This class defines a basic 1D dilated convolutional block with two convolutional layers,
    batch normalization, ReLU activation, and an optional shortcut connection for residual learning.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilations=(1, 1)):
        super(DilatedBasic1D, self).__init__()
        # First convolutional layer with dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=dilations[0],
                               dilation=dilations[0], bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        # Second convolutional layer with dilation
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=dilations[1], dilation=dilations[1],
                               bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # Shortcut connection to match dimensions if necessary
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        """
        Defines the forward pass through the block.
        """
        # Apply first convolutional layer, batch norm, and ReLU activation
        out = F.relu(self.bn1(self.conv1(x)))
        # Apply second convolutional layer and batch norm
        out = self.bn2(self.conv2(out))
        # Add the shortcut connection
        out += self.shortcut(x)
        # Apply ReLU activation
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    This class defines an encoder network composed of an initial convolutional block followed by several dilated convolutional blocks.
    """

    def __init__(self):
        super(Encoder, self).__init__()
        # Initial convolutional block with padding, convolution, batch norm, ReLU, and max pooling
        self.init_convs = nn.Sequential(*[
            nn.ConstantPad1d(3, 0),
            nn.Conv1d(1, 64, 7, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1)
        ])
        # Sequential stack of DilatedBasic1D blocks
        self.convs = nn.Sequential(*[
            DilatedBasic1D(in_channels=64, out_channels=64, stride=1, dilations=[1, 2]),
            DilatedBasic1D(in_channels=64, out_channels=64, stride=1, dilations=[4, 8]),
            DilatedBasic1D(in_channels=64, out_channels=128, stride=2, dilations=[1, 2]),
            DilatedBasic1D(in_channels=128, out_channels=128, stride=1, dilations=[4, 8]),
            DilatedBasic1D(in_channels=128, out_channels=256, stride=2, dilations=[1, 2]),
            DilatedBasic1D(in_channels=256, out_channels=256, stride=1, dilations=[4, 8]),
            DilatedBasic1D(in_channels=256, out_channels=512, stride=2, dilations=[1, 2]),
            DilatedBasic1D(in_channels=512, out_channels=512, stride=1, dilations=[4, 8]),
        ])
        # Adaptive average pooling to reduce the output to a fixed size
        self.classifier = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Defines the forward pass through the encoder.
        """
        # Pass through initial convolutional block
        x = self.init_convs(x)
        # Pass through dilated convolutional blocks
        x = self.convs(x)
        # Apply adaptive average pooling
        x = self.classifier(x)
        # Flatten the output
        x = x.view(x.shape[0], -1)
        return x


class VarCNN(nn.Module):
    def __init__(self, num_classes, num_tab=1):
        super(VarCNN, self).__init__()
        self.model = VarCNN_model(num_classes=num_classes, num_tab=num_tab)

        # Classifier consisting of linear layers, batch norm, ReLU, and dropout
        self.classifier = nn.Sequential(*[
            nn.Linear(in_features=1024, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=num_classes)
        ])

    def forward(self, x):
        feat = self.model(x)

        # Pass through the classifier
        x = self.classifier(feat)
        return x


class VarCNN_model(nn.Module):
    """
    This class defines the overall VarCNN composed of two encoders (directional and temporal)
    and a classifier for final prediction.
    """

    def __init__(self, num_classes, num_tab=1):
        super(VarCNN_model, self).__init__()
        # Two separate encoders for directional and temporal data
        self.dir_encoder = Encoder()
        self.time_encoder = Encoder()


    def forward(self, x):
        """
        Defines the forward pass through the VarCNN.
        """
        # Separate input into directional and temporal components and pass through respective encoders
        x_dir = self.dir_encoder(x[:, 0:1, :])
        x_time = self.time_encoder(x[:, 1:, :])
        # Concatenate the outputs of the two encoders
        x = torch.concat((x_dir, x_time), dim=1)

        return x
