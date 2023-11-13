# library that attacks a group of images with imperceptible noise
# https://github.com/Harry24k/adversarial-attacks-pytorch
import torchattacks
from patch_class import PatchAttacker
import torch
from tqdm import tqdm

'''
This file is used to attack a set of images.
'''

# create adversarially attacked images in batches
def attack_images(model, attack_type, images, labels, num_images, batch_size, device):
    torch.cuda.empty_cache()

    # setup attack object  
    if attack_type == "patch":
        mean_vec = [0.485, 0.456, 0.406]
        std_vec =  [0.229, 0.224, 0.225]
        attacker = PatchAttacker(model, mean_vec, std_vec, device, patch_size = 39, step_size=0.05, steps=250, image_size=224)

        attacked_images = torch.empty(num_images, 3, 224, 224)    

        for i in tqdm(range(num_images), desc = "Performing " + attack_type + " attack"):
            true_class = labels[i]
            ground_truth = torch.zeros(1, 1000)
            ground_truth[0, true_class] = 1

            adv_image, _ = attacker.perturb(images[i], ground_truth.to(device))

            attacked_images[i] = adv_image

        return attacked_images
    else:
        if attack_type == "Linf":
            atk = torchattacks.PGD(model, eps = 8/255, alpha = 2/255, steps = 50, random_start = True)
        elif attack_type == "L2":
            atk = torchattacks.PGDL2(model, eps = 1.0, alpha = 0.2, steps = 50, random_start = True) 

        atk.set_normalization_used(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        atk.set_mode_targeted_by_label(quiet = True) # do not show the message
        target_labels = (labels + 100) % 1000

        num_batches = int(num_images / batch_size)

        # empty array to capture each attacked image
        attacked_images = torch.empty(num_batches, batch_size, 3, 224, 224)

        labels = labels.to(torch.int64)

        # get a batch of attacked images
        for i in tqdm(range(num_batches), desc = "Performing " + attack_type + " attack"):
            torch.cuda.empty_cache()

            attacked_image_set = torch.empty(batch_size, 3, 224, 224)
            attacked_image_set = atk(images[i * batch_size : (i + 1) * batch_size].squeeze(), target_labels[i * batch_size : (i + 1) * batch_size])

            attacked_images[i] = attacked_image_set

        # Stack all of the images
        attacked_images = torch.reshape(attacked_images, (attacked_images.shape[0] * attacked_images.shape[1], 3, 224, 224))

        return attacked_images