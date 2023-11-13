import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm


# transform data into format needed for resnet models. Expect 224x224 3-channel image
transform = transforms.Compose([
     transforms.Resize((224, 224)),
     transforms.CenterCrop(224),
     transforms.ToTensor()
])

# standard ImageNet normalization
transform_normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]
)

# applies tranforms and other operations to a PIL image so it can be used in a resnet model
def make_input_image(input):
    image = transform(input)
    image = transform_normalize(image)
    image = image.unsqueeze(0)

    return image

def load_data(path, device):
    print("Data Loading Initialized")

    class_index = np.loadtxt(path + "/train_image_classes.txt").astype(np.int64)

    image_path = path + "/images"
    images = sorted(os.listdir(image_path))
    image_count = len(images)
        
    data = torch.empty((image_count, 1, 3, 224, 224))
    targets = torch.empty(image_count, dtype = torch.int64)
    image_numbers = torch.empty(image_count, dtype = torch.int64)

    images_used = 0
    

    for i in tqdm(range(image_count), desc = "Loading Images"):    
        img = Image.open(image_path + "/" + images[i])
        img = make_input_image(img)
        
        image_number = int((images[i]).split(".")[0])

        # put the images, targets, and image number into arrays
        data[images_used] = img
        targets[images_used] =  class_index[image_number]
        image_numbers[images_used] = image_number

        images_used += 1

    data = data.to(device)
    targets = targets.to(device)
    
    return data, targets, image_numbers
