import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image
import json
from collections import OrderedDict
from time import time

# Creates Argument Parser object named parser
print('--DEFAULT PAREMETERS--')
print('path to checkpoint: check.pth')
print('path to image: flowers/train/100/image_07894.jpg')
print('Category Path: cat_to_name.json')
print('Top K Classes: 5')
print('Use GPU: Yes')

print("To Use Default Parameters press [0] for custom parameters press [1]")
prediction_setup = input('Decision: ')

if prediction_setup == '0':
    my_checkpoint = 'check.pth'
    my_image = 'flowers/train/100/image_07894.jpg'
    cat_to_name = 'cat_to_name.json'
    top_k_classes = 5
    device = 'cuda'
else:
    my_checkpoint = input('Enter Path to Checkpoint File: ')
    my_image = input('Enter Path to image file: ')
    cat_to_name = input('Enter Path to Category File: ')
    top_k_classes = int(input('Enter Desired top K classes: '))
    device = input('Enter cpu or cuda:')
    

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    if checkpoint['transfer_model'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained = True)
    else:
        raise ValueError('Model arch error.')

    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['idx_to_class']
    
    
    return model

model = load_checkpoint(my_checkpoint)
model.to(device);


if torch.cuda.is_available():
    print("GPU is available")
    if device == 'cuda':
        print('using GPU')
        model.to(device)
    else:
        print('using CPU')
        model.to(device)
else:
    print("GPU is not available defaulting to CPU")
    device = 'cpu'
    model.to(device)

def process_image(image):
    '''Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a Numpy array'''

    # Open the image file
    im = Image.open(image)

    # Get the width and height of the image
    width, height = im.size
    aspect_ratio = width / height

    # Resize the image to 256 pixels on the shorter side
    if aspect_ratio > 1:
        im = im.resize((256, int(256 / aspect_ratio)))
    else:
        im = im.resize((int(256 * aspect_ratio), 256))

    # Crop the image to 244x244 pixels centered
    left = (im.width - 244) / 2
    top = (im.height - 244) / 2
    right = (im.width + 244) / 2
    bottom = (im.height + 244) / 2
    im = im.crop((left, top, right, bottom))

    # Convert the image to a numpy array
    np_image = np.array(im) / 255

    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Transpose the image
    np_image = np_image.transpose((2, 0, 1))

    # Return the processed image
    return np_image


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Open the image file and process it
    with torch.no_grad():
        image = process_image(image_path)
        # Convert the image to a tensor
        image = torch.from_numpy(image)
        # Add a batch dimension to the image
        image.unsqueeze_(0)
        # Convert the image to float type
        image = image.float()
        # Get the output of the model
        model.to('cpu')
        outputs = model(image)
        # Get the top k classes and their probabilities
        probs, classes = torch.exp(outputs).topk(topk)
        # Convert the classes and probs to lists
        return probs[0].tolist(), classes[0].add(1).tolist()


probs, classes = predict(my_image, model, topk=top_k_classes)


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
names = [cat_to_name.get(key) for key in classes]
print("Class name:")
print(names)

print("Class number:")
print(classes)
print("Probability (%):")
for idx, item in enumerate(probs):
    probs[idx] = round(item*100, 2)
print(probs)
