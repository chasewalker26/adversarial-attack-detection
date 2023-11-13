import torch
from torchvision import models
from torchvision import transforms
from torchvision import utils

import numpy as np
import matplotlib.pyplot as plt
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap

import os
import argparse
import attributions as attribution

import data_loader
import attacker

import captum.attr as attrMethods
from modified_models import resnet

import warnings

from tqdm import tqdm

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

plt.rcParams['figure.dpi'] = 75
plt.rcParams['savefig.dpi'] = 75
plt.ioff()

def save_attacks(model, images, attacked_images, image_numbers, path_to_save, device):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    image_count = attacked_images.shape[0]
    images_used = 0

    good_images = torch.zeros((image_count,images.shape[1], images.shape[2], images.shape[3], images.shape[4]))
    good_attacked_images = torch.zeros((image_count, images.shape[2], images.shape[3], images.shape[4]))

    try:
        for i in tqdm(range(image_count), desc = "Saving attacks"):
            benign = images[i]
            attacked = attacked_images[i]

            # make sure the attack worked
            worked, _, _ = attack_worked(benign, attacked, model, device)
            if worked == False:
                continue
    
            utils.save_image(invert_normalize(attacked_images[i]), path_to_save + "/" + f'{image_numbers[images_used]:.0f}' + ".jpg")
            
            good_images[images_used] = benign
            good_attacked_images[images_used] = attacked

            images_used += 1
    except:
        pass

    return good_images[0: images_used], good_attacked_images[0: images_used], image_numbers[0 : images_used]

def save_attributions(benign_attr, attacked_attr, image_numbers, path_to_save):
    default_cmap = LinearSegmentedColormap.from_list('custom blue',  [(0, '#ffffff'), (0.25, '#0000ff'), (1, '#0000ff')], N = 256)

    image_count = benign_attr.shape[0]

    benign_path = path_to_save + "/benign/"
    attacked_path = path_to_save + "/attacked/"

    if not os.path.exists(benign_path):
        os.makedirs(benign_path)
    if not os.path.exists(attacked_path):
        os.makedirs(attacked_path)

    try:
        for i in tqdm(range(image_count), desc = "Saving attributions"):
            # save benign attr
            fig, _ = viz.visualize_image_attr(np.transpose(benign_attr[i].squeeze().cpu().detach().numpy(), (1,2,0)),
                method = 'heat_map',
                plt_fig_axis = plt.subplots(1, 1),
                cmap = default_cmap,
                sign = "absolute_value",
                use_pyplot = False)

            plt.figure(fig)
            fig.savefig(benign_path + f'{image_numbers[i]:.0f}' + ".jpg", bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig)

            # save attacked attr
            fig, _ = viz.visualize_image_attr(np.transpose(attacked_attr[i].squeeze().cpu().detach().numpy(), (1,2,0)),
                method = 'heat_map',
                plt_fig_axis = plt.subplots(1, 1),
                cmap = default_cmap,
                sign = "absolute_value",
                use_pyplot = False)

            plt.figure(fig)
            fig.savefig(attacked_path + f'{image_numbers[i]:.0f}' + ".jpg", bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig)
    except:
        pass
    
    return

def attack_worked(benign, attacked, model, device):
    benign_class, _ = attribution.getClass(benign.to(device), model, device)
    attacked_class, _ = attribution.getClass(attacked.to(device), model, device)

    if benign_class == attacked_class:
        return False, False, False
    else:
        return True, benign_class, attacked_class

def attribute_images(model, attr_method, images, attacked_images, image_numbers, device):
    image_count = len(attacked_images)
    images_used = 0

    modified_model = resnet.resnet101(weights = "ResNet101_Weights.IMAGENET1K_V2")
    modified_model = modified_model.eval()
    modified_model.to(device)
    GBP = attrMethods.GuidedBackprop(modified_model)
    
    # hold the attributions
    benign_attr = torch.zeros((image_count, images.shape[2], images.shape[3], images.shape[4]))
    attacked_attr = torch.zeros((image_count, images.shape[2], images.shape[3], images.shape[4]))

    try:    
        for i in tqdm(range(image_count), desc = "Generating " + attr_method + " attributions"):
            benign = images[i]
            attacked = attacked_images[i]
            
            # # make sure the attack worked
            worked, benign_class, attacked_class = attack_worked(benign, attacked, model, device)
            if worked == False:
                continue

            if attr_method == "IDG":
                benign_attr[images_used] = attribution.IDG1(invert_normalize(benign.squeeze()).to(device), model, 50, 50, 0, device, benign_class)
                attacked_attr[images_used] = attribution.IDG1(invert_normalize(attacked.squeeze()).to(device), model, 50, 50, 0, device, attacked_class)
            elif attr_method == "IG":
                benign_attr[images_used] = attribution.IGParallel(benign.to(device), model, 50, 50, 0, device, benign_class)
                attacked_attr[images_used] = attribution.IGParallel(attacked.to(device), model, 50, 50, 0, device, attacked_class)
            elif attr_method == "GBP":
                benign.requires_grad = True
                attacked.requires_grad = True

                attacked = attacked.unsqueeze(0)    

                warnings.filterwarnings("ignore", message="Setting backward hooks on ReLU activations.")

                benign_attr[images_used] = GBP.attribute(benign.to(device), target = benign_class)   
                attacked_attr[images_used] = GBP.attribute(attacked.to(device), target = attacked_class)           

            images_used += 1
    except:
        pass

    return benign_attr[0 : images_used], attacked_attr[0 : images_used], image_numbers[0 : images_used]

def main(FLAGS):
    device = 'cuda:' + str(FLAGS.cuda_num) if torch.cuda.is_available() else 'cpu'
    model = models.resnet101(weights = "ResNet101_Weights.IMAGENET1K_V2")
    model = model.eval()
    model.to(device)

    # loads and transforms correctly classified RGB imagenet validation images that pass cutoff
    # images, labels, image_numbers = data_loader.load_data(FLAGS.path_to_load, FLAGS.image_count, FLAGS.data_type, FLAGS.cutoff, model, device)
    images, labels, image_numbers = data_loader.load_data(FLAGS.path_to_load, device)

    attrs = ["IDG", "IG", "GBP"]

    # attack loaded images
    attacked_images = attacker.attack_images(model, FLAGS.attack, images, labels, len(images), FLAGS.batch_size, device)
    
    images, attacked_images, image_numbers = save_attacks(model, images, attacked_images, image_numbers, "attacked_" + FLAGS.attack, device)

    for i in range(len(attrs)):
        # gather attributions
        benign_attr, attacked_attr, image_numbers = attribute_images(model, attrs[i], images, attacked_images, image_numbers, device)
        
        save_attributions(benign_attr, attacked_attr, image_numbers, "classification_" + attrs[i] + "_" + FLAGS.attack)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Adversarial Classifier Training Data Generation Script.')
    parser.add_argument('--batch_size',
            type = int, default = 10,
            help='batch_size for the attack.')
    parser.add_argument('--cuda_num',
            type = int, default = 0,
            help = 'The number of the GPU you want to use.')
    parser.add_argument('--path_to_load',
            type = str, default = "train_data",
            help = 'The path to images and their classes txt file.')
    parser.add_argument('--attack',
            type = str, default = "Linf",
            help = 'The attack method to use: Linf, L2, patch')
    
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)
