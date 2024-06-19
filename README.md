# DeepLabv3Plus-Pytorch + CCMBA
This the source code used in my Final Year Project - Robust Semantic SLAM for Autonomous Robot. It is forked from the [DeepLabv3Plus-Pytorch repository](https://github.com/VainF/DeepLabV3Plus-Pytorch), and integrates class centric motion blur augmentation (CCMBA) transformations provided in [this repository](https://github.com/aka-discover/CCMBA_CVPR23) based on [this research paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Aakanksha_Improving_Robustness_of_Semantic_Segmentation_to_Motion-Blur_Using_Class-Centric_Augmentation_CVPR_2023_paper.pdf)

## Code Overview
Building on the existing DeepLabV3+ code, this repository includes 900 blur kernels generated using Point Spread Functions, which are used to blur images from the PASCAL VOC dataset for training and testing. The dataset images are artificially blurred using functions and transformations provided in the CCMBA repository before semantic segmentation.

## Model Training and Testing
All the models trained were DeepLabV3+ with ResNet50 backbone, which was what the CCMBA study used. The model retrained on the CCMBA augmented dataset is the final product of this project. 2 baselines were used to benchmark the performance of this model. The first baseline is the model pretrained on the clean, original dataset, and the second is the same pre-trained model fine-tuned on the CCMBA augmented dataset.
To test the models, a set of varying datasets were used:  a clean dataset, which is the original untouched dataset, a mixed dataset, which is half clean images and half augmented images across all blur levels, and 3 other datsets each fully containing augmented images but only for a specific blur level. 

## The CCMBA Algorithm
The CCMBA algorithm applies class-centric blur to each image from the PASCAL VOC dataset. First, it takes in a sharp image and its ground truth class centric mask. It blurs each image with 50% probability, and if an image is to be blurred, the algorithm randomly selects 1 out of the 900 blur kernels to apply to the image. This involves randomly choosing a subset of classes within the image to blur, then creating a binary mask for them using thresholding, before applying the kernel selectively to the image where the binary mask is positive. This result is blended with the unblurred regions of the original sharp image to create an image with either linear blur, or non-linear blur on at least one object class.

