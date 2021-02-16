import argparse
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras.optimizers import Adam , SGD

from utils.dataset import get_dataset
from utils.model import load_model_arch 

class Trainer():
    def __init__(self, 
                imagenet_path, 
                model_name, 
                bs, 
                steps, 
                epochs, 
                lr,
                num_data_workers,
                save_dir, 
                metrics='accuracy'):
        
        self.strategy = tf.distribute.MirroredStrategy()
        self.num_devices = int(self.strategy.num_replicas_in_sync)
        print ('The number of devices: {}'.format(self.num_devices))

        with self.strategy.scope():

            self.model_name = model_name
            self.bs = bs * self.num_devices
            self.steps = steps
            self.epochs = epochs
            self.lr = lr

            self.ds_train = get_dataset(imagenet_path, 'train', bs, num_data_workers)
            self.ds_val = get_dataset(imagenet_path, 'validation', bs, num_data_workers)

            self.ds_train = self.strategy.experimental_distribute_dataset(self.ds_train)
            self.ds_val = self.strategy.experimental_distribute_dataset(self.ds_val)

            self.model = load_model_arch(self.model_name, 1000) #1000 is the number of class

            self.optimizer= SGD(lr=self.lr, momentum=0.9, decay=3e-5, nesterov=False)

            self.loss = tf.keras.losses.CategoricalCrossentropy()
            self.metrics = metrics

            self.save_dir = save_dir 
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
    
    def get_callback_list(self, early_stop = True , lr_reducer = True):

        callback_list=list()
        
        if early_stop == True:
            callback_list.append(tf.keras.callbacks.EarlyStopping(min_delta=0, patience=20, verbose=2, mode='auto'))
        if lr_reducer == True:
            callback_list.append(tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, cooldown=0, patience=10, min_lr=0.5e-6))
        

        callback_list.append(tf.keras.callbacks.ModelCheckpoint(
                                os.path.join(self.save_dir, self.model_name) + '-{epoch:03d}.h5',
                                monitor='val_loss',
                                save_best_only=False))
    
        return callback_list

    def run(self):

        self.model.compile(loss= self.loss, optimizer=self.optimizer, metrics=[self.metrics])
        callback_list = self.get_callback_list()

        self.model.fit( x = self.ds_train,
                        steps_per_epoch = 1281167 // self.bs, # 1281167 is the number of training data
                        validation_data = self.ds_val,
                        validation_steps = 50000 // self.bs, # 50000 is the number of validation data
                        callbacks = callback_list,
                        # The following doesn't seem to help in terms of speed.
                        # use_multiprocessing=True, workers=4,
                        epochs = self.epochs)



if __name__ == '__main__':
    
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='EfficientNet Tensorflow Keras based on ImageNet')

    # Arguments related to dataset
    parser.add_argument('--imagenet_path', type = str, default = './dataset/tfrecord/', help ='ImageNet dataset tfrecord path' )
    parser.add_argument('--model_name', type = str, default = 'EfficientNetB0_M', help ='EfficientNet model name' )

    # Arguments related to train config
    parser.add_argument('--bs', type = int, default = 16, help = 'Batch size')
    parser.add_argument('--steps', type=int, default = 1000, help = 'Number of steps per epoch.')
    parser.add_argument('--epochs', type = int, default = 100, help = 'Epoch number')
    parser.add_argument('--lr', type = float, default = 1e-2, help = 'Epoch number')

    parser.add_argument('--num_data_workers', type = int, default = 2, help = 'The number of workers')
    parser.add_argument('--save_dir', type = str, default = './result/', help = 'Save path for the object detection model')

    args = parser.parse_args()

    trainer_obj = Trainer(  imagenet_path       = args.imagenet_path,
                            model_name          = args.model_name,
                            bs                  = args.bs,
                            steps               = args.steps,
                            epochs              = args.epochs,
                            lr                  = args.lr,                    
                            num_data_workers    = args.num_data_workers,
                            save_dir            = args.save_dir
                        )
                            
    trainer_obj.run()

