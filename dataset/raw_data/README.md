# EfficientNet_Mobile

Mobile friendly EfficientNet based on ImageNet with tensorflow keras

![Python version support](https://img.shields.io/badge/python-3.6-blue.svg)
![Tensorflow version support](https://img.shields.io/badge/tensorflow-2.3.0-red.svg)

:star: Star us on GitHub — it helps!!

Mobile friendly EfficientNet implementation for *[Here](https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html)*


## Install

You will need a machine with a GPU and CUDA installed.  
Then, you prepare runtime environment:

   ```shell
   pip install -r requirements.txt
   ```

## Training

### 0. Get 
### 1. Prepare your dataset

# Step-by-step


0. Clone this repository.

   ```shell
   $ git clone https://github.com/da2so/EfficientNet_Mobile.git
   $ cd EfficientNet_Mobile/dataset/
   ```

1. Download the "Training images (Task 1 & 2)" and "Validation images (all tasks)" from the [ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012) download page](http://image-net.org/download) in `dataset` directory.

   ```shell
   $ ls -l ./
   -rwxr-xr-x 1 jkjung jkjung 147897477120 Nov  7  2018 ILSVRC2012_img_train.tar
   -rwxr-xr-x 1 jkjung jkjung   6744924160 Nov  7  2018 ILSVRC2012_img_val.tar
   ```

2. Untar the "train" and "val" files.  For example, I put the untarred files at ${HOME}/data/ILSVRC2012/.

   ```shell
   $ mkdir -p ${HOME}/data/ILSVRC2012
   $ cd ${HOME}/data/ILSVRC2012
   $ mkdir train
   $ cd train
   $ tar xvf ${HOME}/Downloads/ILSVRC2012_img_train.tar
   $ find . -name "*.tar" | while read NAME ; do \
         mkdir -p "${NAME%.tar}"; \
         tar -xvf "${NAME}" -C "${NAME%.tar}"; \
         rm -f "${NAME}"; \
     done
   $ cd ..
   $ mkdir validation
   $ cd validation
   $ tar xvf ${HOME}/Downloads/ILSVRC2012_img_val.tar
   ```

3. Clone this repository.

   ```shell
   $ git clone https://github.com/da2so/EfficientNet_Mobile.git
   $ cd EfficientNet_Mobile
   ```

4. Pre-process the validation image files.  (The script would move the JPEG files into corresponding subfolders.)

   ```shell
   $ cd data
   $ python3 ./preprocess_imagenet_validation_data.py \
             ${HOME}/data/ILSVRC2012/validation \
             imagenet_2012_validation_synset_labels.txt
   ```

5. Build TFRecord files for "train" and "validation".  (This step could take a couple of hours, since there are 1,281,167 training images and 50,000 validation images in total.)

   ```shell
   $ mkdir ${HOME}/data/ILSVRC2012/tfrecords
   $ python3 build_imagenet_data.py \
             --output_directory ${HOME}/data/ILSVRC2012/tfrecords \
             --train_directory ${HOME}/data/ILSVRC2012/train \
             --validation_directory ${HOME}/data/ILSVRC2012/validation
   ```
Then 
    # ImageNet dataset structure should be like this
    dataset/
        -raw_data/
            -train/
                -n01440764/
                	-n01440764_*.JPEG
                -n01443537/
                	-n01443537_*.JPEG
                ...
            -val/
                -n01440764/
                	-n01440764_*.JPEG
                -n01443537/
                	-n01443537_*.JPEG
                ...
        -tfrecord/
    