# Adversarial Attack Detection via Attributions
The implementation of our MILCOM 2023 paper - "Advesarial Pixel and Patch Detection Using Attribution Analysis"

Example usage to train an IDG Linf attack detector
---

#### Generate the images to train on from a 2012 imagenet train dataset folder
  * `python3 getImages.py --image_count 20000 --path_to_load imagenet_train --folder_name training_images`
  * This dataset generation only needs to be done once for a given number of images

#### Generate (IDG, GBP, and IG) attributions for Linf pixel attacks. 
  * Only done once for any attack type.  
  * Makes 4 folders: attacked_Linf (images), classification_IDG_Linf, classification_GBP_Linf, classification_IG_Linf (attributions)
  * `python3 makeTrainingData.py --cuda_num 0 --batch_size 100 --path_to_load training_images --attack Linf`

#### Train the detector using one attribution type (IDG):
  * `python3 training.py --path_to_load classification_IDG_Linf/ --save_name IDG_Linf.pth.tar --epochs 50 --batch-size 1024 --gpu <gpu number 0-N or none if using multiprocessing>`
  
  * If training on IG or GBP:
    * `python3 code/adversarialTraining/training.py --path_to_load classification_IG_Linf/ --save_name IG_Linf.pth.tar`
    * `python3 code/adversarialTraining/training.py --path_to_load classification_GBP_Linf/ --save_name GBP_Linf.pth.tar`

#### Evaluate the detector
  * `python3 training.py --path_to_load classification_IDG_Linf/ --resume IDG_Linf.pth.tar`

Notebook
---
This repository provides an example notebook attackTest.ipynb
  * It showcases the reaction of IDG, GBP, and IG to the three attack types as discussed in the paper
