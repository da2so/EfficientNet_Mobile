# EfficientNet_Mobile

Mobile friendly EfficientNet based on ImageNet dataset with tensorflow keras

![Python version support](https://img.shields.io/badge/python-3.6-blue.svg)
![Tensorflow version support](https://img.shields.io/badge/tensorflow-2.3.0-red.svg)

:star: Star us on GitHub — it helps!!

Mobile friendly EfficientNet is described in *[Here](https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html)*


## Install

You will need a machine with a GPU and CUDA installed.  
Then, you prepare runtime environment:

   ```shell
   pip install -r requirements.txt
   ```

## Use

### 1. Prepare your dataset

#### Step-by-step

0. Clone this repository.

   ```shell
   $ git clone https://github.com/da2so/EfficientNet_Mobile.git
   $ cd EfficientNet_Mobile/dataset/raw_data
   ```

1. Download the "Training images (Task 1 & 2)" and "Validation images (all tasks)" from the [ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012) download page](http://image-net.org/download) at `dataset/raw_data` directory.

   ```shell
   $ ls -l ./
   -rwxr-xr-x 1 shkang shkang 147897477120 Feb  14 14:55 ILSVRC2012_img_train.tar
   -rwxr-xr-x 1 shkang shkang   6744924160 Feb  14 15:58 ILSVRC2012_img_val.tar
   ```

2. Untar the "train" and "val" files. For example, I put the untarred files at `dataset/raw_data`.

   ```shell
   $ mkdir train
   $ cd train
   $ tar xvf ../ILSVRC2012_img_train.tar
   $ find . -name "*.tar" | while read NAME ; do \
         mkdir -p "${NAME%.tar}"; \
         tar -xvf "${NAME}" -C "${NAME%.tar}"; \
         rm -f "${NAME}"; \
     done
   $ cd ..
   $ mkdir validation
   $ cd validation
   $ tar xvf ../ILSVRC2012_img_val.tar
   ```

4. Pre-process the validation image files. (The script would move the JPEG files into corresponding subfolders.)

   ```shell
   $ cd ../../../data  # EfficientNet_Mobile/dataset/raw_data/validation -> EfficientNet_mobile/data
   $ python  ./process_val.py \
             ./dataset/raw_data/valiation/ \
             imagenet_2012_validation_synset_labels.txt
   ```

5. Build TFRecord files for "train" and "validation". (This step could take a couple of hours, since there are 1,281,167 training images and 50,000 validation images in total.)

   ```shell
   $ cd .. 
   $ mkdir /dataset/tfrecord/
   $ python  convert2tfrecord.py \
             --output_directory=./dataset/tfrecord/ \
             --train_directory=./dataset/raw_data/train/ \
             --validation_directory=./dataset/raw_data/val/
   ```


### 2. Train Mobile Friendly EfficientNet 

Train mobile friendly EfficientNet based on ImageNet dataset

   ```shell
   $ CUDA_VISIBLE DEVICES=0 python train.py \
                                   --imagenet_path=./dataset/tfrecord/ \
                                   --model_name=EfficientNetB0_M \
                                   --bs=16 \
                                   --epochs=100 \
                                   --lr=1e-2 \
                                   --save_dir=./result/ \
   ```


#### Train Using Multi-GPU

   ```shell
   $ CUDA_VISIBLE DEVICES=0,1,2,3 python train.py \
                                   --imagenet_path=./dataset/tfrecord/ \
                                   --model_name=EfficientNetB0_M \
                                   --bs=16 \
                                   --epochs=100 \
                                   --lr=1e-2 \
                                   --save_dir=./result/ \
   ```

The total number of batch size will be (16 x The number of GPUs) = 16 x 4 = 64.


Arguments:

- `imagenet_path` - ImageNet dataset path
- `model_name` - Name of Mobile friendly EfficientNet  
  - Choices = ['EfficientNetB0_M', 'EfficientNetB1_M', 'EfficientNetB2_M', 'EfficientNetB3_M']
- `bs` -  Batch size
- `epochs` -  The number of epochs
- `lr` - Learning rate
- `save_dir` - Save directory for EfficientNet model




