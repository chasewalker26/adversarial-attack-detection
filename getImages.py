import torch
from torchvision import models
from torchvision import transforms
import numpy as np
import os
import argparse
import cv2
from PIL import Image

# standard ImageNet normalization
transform_normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]
)

# standard ImageNet normalization
invert_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

# transform data into format needed for resnet models. Expect 224x224 3-channel image
transform = transforms.Compose([
     transforms.Resize((224, 224)),
     transforms.CenterCrop(224),
     transforms.ToTensor()
])

# returns the class of an image 
def getClass(input, model, device):
    # calculate a prediction
    input = input.to(device)
    output = model(input)
    _, index = torch.max(output, 1)

    return index[0]

# applies transorms and other operations to a PIL image so it can be used in a resnet model
def make_input_image(input):
    image = transform_normalize(input)
    image = image.unsqueeze(0)

    return image

# checks if bouding box around the image subject is < some percentage of the total image
def checkObjectSize(path, image, cutoff):
    # load image
    img = cv2.imread(path + "/" + image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize and crop to the usual used for attribution
    img = cv2.resize(img, (224, 224))
    center = img.shape
    x = (center[1] / 2) - (224 / 2)
    y = (center[0] / 2) - (224 / 2)
    img = img[int(y) : int(y + 224), int(x) : int(x + 224)]

    # only rgb images can be classified
    if img.shape != (224, 224, 3):
        return False

    # Find bounding box
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    x, y, w, h = cv2.boundingRect(thresh)

    # percentage of the image that the object takes up
    percentage = 100 * (w * h) / 224**2

    if percentage < cutoff:
       return True
    else:
        return False

def find_and_save_data(path, image_count, cutoff, path_to_save, model, device):
    print("Data Loading Initialized")

    images_used = 0
    images_per_class = int(np.ceil(image_count / 1000))
    classes_used = [0] * 1000

    data = torch.empty((image_count, 1, 3, 224, 224))
    targets = torch.empty(image_count, dtype = torch.int64)
    # this holds the number of the selected imagenet image for file naming [1-50000]
    image_numbers = torch.empty(image_count, dtype = torch.int64)

    clsloc = open("class_maps/map_clsloc.txt", 'r')
    class_map = clsloc.readlines()
    class_code_map = [code.split(" ")[0] for code in class_map]

    classes = open("class_maps/imagenet_classes.txt", 'r')
    class_list = classes.readlines()

    if not os.path.exists(path_to_save + "/images"):
        os.makedirs(path_to_save + "/images")
        
    file = open(path_to_save + "/train_image_classes.txt", "w")

    # look at imagenet images in order
    for image in sorted(os.listdir(path)):
        if images_used == image_count:
            break

        # if pulling from imagenet training set
        image_class_code = image.split("_")[0]

        if checkObjectSize(path, image, cutoff) == False:
            continue

        pil_im = Image.open(path + "/" + image)
        img = transform(pil_im)

        # only rgb images can be classified
        if img.shape != (3, 224, 224):
            continue

        img = make_input_image(img)

        # use the class code to find the name in clsloc class map
        map_line = class_code_map.index(image_class_code)
        # an id, line number, and class name e.g sea_snake
        class_info = class_map[map_line]
        # get the class name, and replace the underscore(s) with space(s)
        class_name = (class_info.split(" ")[-1]).replace('_', ' ')
        # get the array number of the class name which is the class index
        class_index = class_list.index(class_name)

        # Track which classes have been used
        if classes_used[class_index] == images_per_class:
            continue
        else:
            classes_used[class_index] += 1       

        # check for correct classification
        classify = getClass(img, model, device)            
        if classify != class_index:
            classes_used[class_index] -= 1
            continue

        # save the image and its class
        pil_im.save(path_to_save + "/images/" + f'{(images_used):.0f}' + ".jpg")
        file.write(str(class_index) + "\n")

        images_used += 1
        
        if images_used % images_per_class == 0:
            print("Found " + str(images_used) + "/" + str(image_count) + " images")
            
    print("Found " + str(images_used) + "/" + str(image_count) + " images")
    print("Data Loading Finished")
    
    file.close()
    clsloc.close()
    classes.close()

    # remove any remaining indicies that were not filled with images
    data = data[0 : images_used]
    targets = targets[0 : images_used]
    image_numbers = image_numbers[0 : images_used]

    data = data.to(device)
    targets = targets.to(device)

    return 

def main(FLAGS):
    device = 'cuda:' + str(FLAGS.cuda_num) if torch.cuda.is_available() else 'cpu'
    model = models.resnet101(weights = "ResNet101_Weights.IMAGENET1K_V2")
    model = model.eval()
    model.to(device)

    # loads and transforms correctly classified RGB imagenet validation images that pass cutoff
    find_and_save_data(FLAGS.path_to_load, FLAGS.image_count, FLAGS.cutoff, FLAGS.folder_name, model, device)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Adversarial Classifier Training Data Generation Script.')
    parser.add_argument('--image_count',
            type = int, default = 50,
            help='How many attributions to make.')
    parser.add_argument('--cutoff',
            type = int, default = 50,
            help='Maximum area an object in an image can take when loading images.')
    parser.add_argument('--cuda_num',
            type = int, default = 0,
            help = 'The number of the GPU you want to use.')
    parser.add_argument('--path_to_load',
            type = str, default = "imagenet_train",
            help = 'The path for images to attack and attribute".')
    parser.add_argument('--folder_name',
            type = str, default = "train_data",
            help = 'The folder name to save image training data.')
    
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)
