# Dataset structure


   ```shell
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
            -train-00000-of-01024
            -train-00001-of-01024
            ...
            -validation-00000-of-00128
            -validation-00001-of-00128
            ...
   ```
