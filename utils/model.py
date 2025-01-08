import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming the ImageCaptioningModel is defined as in the previous response
class ImageCaptioningModel(nn.Module):
    def __init__(self, target_size, vocabulary_size, embedding_dim):
        super(ImageCaptioningModel, self).__init__()

        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten the output of the convolutional layers
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256 * (target_size[0] // 8) * (target_size[1] // 8), 256)

        # Embedding layer for the sequence input
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.dropout2 = nn.Dropout(0.2)
        self.lstm = nn.LSTM(embedding_dim, 256, batch_first=True)

        # Decoder layers
        self.fc2 = nn.Linear(256 + 256, 256)
        self.fc3 = nn.Linear(256, vocabulary_size)

    def forward(self, input_1, input_2):
        # Convolutional layers
        x1 = F.relu(self.conv1(input_1))
        x1 = self.pool1(x1)
        x1 = F.relu(self.conv2(x1))
        x1 = self.pool2(x1)
        x1 = F.relu(self.conv3(x1))
        x1 = self.pool3(x1)
        x1 = self.flatten(x1)

        # Fully connected layers
        x1 = self.dropout1(x1)
        x1 = F.relu(self.fc1(x1))

        # Embedding layer
        x2 = self.embedding(input_2)
        x2 = self.dropout2(x2)
        x2, _ = self.lstm(x2)
        x2 = x2[:, -1, :]  # Take the last output of the LSTM

        # Decoder layers
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x