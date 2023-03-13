# Starter code for Part 1 of the Small Data Solutions Project
# 

#Set up image data for train and test

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms 
from TrainModel import train_model
from TestModel import test_model
from torchvision import models


data_dir = 'imagedata-50/'
train_data = data_dir + 'train/'
val_data = data_dir + 'val/'
test_data = data_dir + 'test/'

# use this mean and sd from torchvision transform documentation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

#Set up Transforms (train, val, and test)

#<<<YOUR CODE HERE>>>
data_transforms = {
        "train": transforms.Compose(
            [transforms.RandomHorizontalFlip(),
                                      transforms.Resize(256),
                                      transforms.RandomCrop(224),
                                      transforms.RandomRotation(20),
                                      transforms.ColorJitter(brightness = 0.2, contrast=0.2),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)]
        ),
        "val": transforms.Compose(
            [transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)]
        ),
        "test": transforms.Compose(
            [transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)]
        ),
    }



#Set up DataLoaders (train, val, and test)
batch_size = 10
num_workers = 4

#<<<YOUR CODE HERE>>>

train_set = datasets.ImageFolder(train_data, transform=data_transforms['train'])
val_set = datasets.ImageFolder(train_data, transform=data_transforms['val'])
test_set = datasets.ImageFolder(train_data, transform=data_transforms['test'])



train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, 
                                           num_workers=num_workers)



#hint, create a variable that contains the class_names. You can get them from the ImageFolder

class_names = train_set.classes


# Using the VGG16 model for transfer learning 
# 1. Get trained model weights
model = models.vgg16(pretrained=True)

# 2. Freeze layers so they won't all be trained again with our data
for param in model.parameters():
    param.requires_grad = False

# 3. Replace top layer classifier with a classifer for our 3 categories
out_features = 3 # 3 classes 

features = list(model.classifier.children())[:-1]
features.extend([nn.Linear(model.classifier[6].in_features, out_features)]) 
model.classifier = nn.Sequential(*features)

# Train model with these hyperparameters
# 1. num_epochs 
# 2. criterion 
# 3. optimizer 
# 4. train_lr_scheduler 

#<<<YOUR CODE HERE>>>
num_epochs = 3 # Transfer learning maximally needs 10 epochs
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
train_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

# When you have all the parameters in place, uncomment these to use the functions imported above
def main():
   trained_model = train_model(model, criterion, optimizer, train_lr_scheduler, train_loader, val_loader, num_epochs=num_epochs)
   test_model(test_loader, trained_model, class_names)

if __name__ == '__main__':
    main()
    print("done")