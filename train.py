import numpy as np
import torch
import json
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict


print('--DEFAULT PAREMETERS--')
print('Architecture: vgg16')
print('hidden units: 4096')
print('Learning Rate: 0.001')
print('Epochs: 1')
print('Use GPU: Yes')

print("To Use Default Parameters press [0] for custom parameters press [1]")
classifier_setup = input('Decision: ')

if classifier_setup == '0':
    my_architecture = 'vgg16'
    my_hidden_units = 4096
    my_learning_rate = 0.001
    my_epochs = 1
    device = 'cuda'
else:
    my_architecture = input('Enter architecture (options - vgg16 - vgg13 ): ')
    my_hidden_units = int(input('Enter no. of desired hidden units: '))
    my_learning_rate = float(input('Enter Desired Learning Rate: '))
    my_epochs = int(input('Enter Desired Epochs: '))
    device = input('Enter cpu or cuda:')
    
    


# ---------------------------SETTING UP DIRCTORIES / LOADING DATA---------------------------------

print('****Setting up Directories / Loading Data****')

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# DONE: Define your transforms for the training, validation, and testing sets

#transform for the validation and test datasets
transforms_valid_test = transforms.Compose([
                                    transforms.Resize(255),
                                    transforms.CenterCrop(244),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485,0.456,0.406),
                                    (0.229, 0.224, 0.225))
                                    ])
#transform for training datasets
transforms_train = transforms.Compose([
                        transforms.Resize(255),
                        transforms.CenterCrop(244),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485,0.456,0.406),
                        (0.229, 0.224, 0.225))
                        ])

# DONE: Load the datasets with ImageFolder

#loading the images
image_datasets = datasets.ImageFolder(data_dir, 
                                    transform = transforms_valid_test)
#loading the training images
train_datasets = datasets.ImageFolder(train_dir, 
                                    transform = transforms_train)
#loading the validation images
valid_datasets = datasets.ImageFolder(valid_dir, 
                                    transform = transforms_valid_test)
#loading the test images
test_datasets = datasets.ImageFolder(test_dir, 
                                    transform = transforms_valid_test)
# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders=torch.utils.data.DataLoader(image_datasets, batch_size=64,
                                        shuffle=True)
trainloaders=torch.utils.data.DataLoader(train_datasets, batch_size=64,
                                         shuffle=True)
validloaders=torch.utils.data.DataLoader(valid_datasets, batch_size=64,
                                         shuffle=True)
testloaders=torch.utils.data.DataLoader(test_datasets, batch_size=64,
                                        shuffle=True)

# ---------------------------LABEL MAPPING---------------------------------

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

    
    # ----------------------------Setting up Model--------------------------------

print("------ Setting Model Parameters ----------------")


if my_architecture == 'vgg16':
    model = models.vgg16(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, my_hidden_units)), # fully connected layer with 25088 inputs and 4096 outputs
                          ('relu', nn.ReLU()), # activation function
                          ('fc2', nn.Linear(my_hidden_units, 102)), # fully connected layer with 4096 inputs and 102 outputs
                          ('output', nn.LogSoftmax(dim=1)) # output layer with log-softmax activation
                          ]))
    model.classifier = classifier
    
elif my_architecture == 'vgg13':
    model = models.vgg13(pretrained = True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, my_hidden_units)), # fully connected layer with 25088 inputs and 4096 outputs
                          ('relu', nn.ReLU()), # activation function
                          ('fc2', nn.Linear(my_hidden_units, 102)), # fully connected layer with 4096 inputs and 102 outputs
                          ('output', nn.LogSoftmax(dim=1)) # output layer with log-softmax activation
                          ]))
    model.classifier = classifier
else:
    raise ValueError('Input Correct Architecture.')

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=my_learning_rate)

print("****** model arch: " + my_architecture)
print("------ model building finished -----------")
### ------------------------------------------------------------




### ------------------------------------------------------------
###                         training the model
### ------------------------------------------------------------

print("------ training the model ----------------")

steps  = 0
running_loss = 0
print_every  = 5
if torch.cuda.is_available():
    print("GPU is available")
    if device == 'cuda':
        print('using GPU')
        model.to(device)
else:
    print("GPU is not available defaulting to CPU")
    device = 'cpu'
    model.to(device)



for epoch in range(my_epochs):
    for inputs, labels in trainloaders:
        steps += 1
        
        # Move input and label tensors to the default device (GPU if available, otherwise CPU)
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        
        logps = model.forward(inputs) # Forward pass
        loss = criterion(logps, labels) # Compute loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() # Accumulate loss for printing
        
        # Print statistics every 'print_every' steps
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            
            # Set model to evaluation mode
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloaders:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    # Accumulate test loss
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            # Print statistics for the current epoch        
            print(f"Epoch {epoch+1}/{my_epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(validloaders):.3f}.. "
                  f"Test accuracy: {accuracy/len(validloaders):.3f}")
            # reset running loss
            running_loss = 0
            # set model back to train mode
            model.train()
            
# TODO: Do validation on the test set

correct,total = 0,0

# No gradient computation is needed in evaluation
with torch.no_grad():
    # Set the model to evaluation mode
    model.eval()
    for data in testloaders:
        images, labels = data
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images, labels = images.to(device), labels.to(device)
        outputs = model(images) # Forward pass
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0) # Find total number of images
        correct += (predicted == labels).sum().item() # Find number of correctly predicted images

# Print the final accuracy on test set        
print('Test Accuracy: %d%%' % (100 * correct / total))


#Save the checkpoint 
model.class_to_idx = train_datasets.class_to_idx


# Create a dictionary to store all the important information of the model
checkpoint = {'transfer_model': 'vgg16',
              'input_size': 25088,
              'output_size': 102,
              'features': model.features,
              'classifier': model.classifier,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'idx_to_class': {v: k for k, v in train_datasets.class_to_idx.items()}
             }

# Save the checkpoint dictionary to a file named 'check.pth'
torch.save(checkpoint, 'check.pth')

