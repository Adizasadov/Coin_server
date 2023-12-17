import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image
import time
import os

class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(51136, 50)
        self.fc2 = nn.Linear(50, 4)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = x.view(x.size(0), -1)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x,dim=1)

model1 = None
model2 = None

def init():
    global model1, model2 
    
    model1 = CNNet()
    model1.load_state_dict(torch.load('./models/model1_h15.pth'))
    model1.eval()
    
    print("Initialized model 1")
    
    model2 = CNNet()
    model2.load_state_dict(torch.load('./models/model2_h15.pth'))
    model2.eval()

    print("Initialized model 1")

def create_spectrogram(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform + 1e-9

    spectrogram_tensor = torchaudio.transforms.Spectrogram()(waveform)
    spectrogram_numpy = spectrogram_tensor.log2()[0,:,:].numpy()
    filter = spectrogram_numpy != np.NINF
    filtered_spectrogram = np.where(filter, spectrogram_numpy, sys.float_info.min) # replace remaining -inf with smallest float

    plt.figure()
    spec_name = f'audio/spec_img_{time.ctime(time.time())}.png'
    plt.imsave(spec_name, filtered_spectrogram, cmap='viridis')
    plt.close()
    
    return spec_name

def get_model_prediction(model, image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')

    # Define the transformation to be applied to the input image
    transform=transforms.Compose([
        transforms.Resize((201,81)),
        transforms.ToTensor()
    ])

    # Apply the transformation to the image
    input_tensor = transform(image)

    # Add a batch dimension to the tensor (as the model expects a batch of images)
    input_batch = input_tensor.unsqueeze(0)

    # Forward pass to get predictions
    with torch.no_grad():
        output = model(input_batch)

    # The 'output' variable now contains the model predictions for the given input image
    probabilities = torch.exp(output)

    # Find the index of the class with the highest probability (argmax)
    return torch.argmax(probabilities).item()


class_map1 = {
    0: 'chirp', 
    1: 'hiss', 
    2: 'meow', 
    3: 'purr'
}

class_map2 = {
    0: 'angry', 
    1: 'happy', 
    2: 'sad', 
    3: 'scared'
}

def classify(audio_path):
    image_path = create_spectrogram(audio_path)
    
    predicted_class_index_m1 = get_model_prediction(model1, image_path)
    
    if predicted_class_index_m1 == 2: # Model 1 predicted meow
        predicted_class_index_m2 = get_model_prediction(model2, image_path)
        os.remove(image_path)
        return f'{class_map2[predicted_class_index_m2]} (Meow)'

    os.remove(image_path)
    return class_map1[predicted_class_index_m1]

    # Optionally, you can also get the probability of the predicted class
    #predicted_class_probability = probabilities[predicted_class_index].item()

    #print("Probability:", predicted_class_probability)