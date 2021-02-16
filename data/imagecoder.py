import tensorflow as tf

class ImageCoder(object):
    def png_to_jpeg(self,img):
        """ convert png to jpeg for an image"""
        uint_img=tf.io.decode_png(img,channels=3)
        img=tf.io.encode_jpeg(image=uint_img, format='rgb', quality=100)

        return img

    def cmyk_to_rgb(self,img):
        """ convert cmyk to jpeg for an image"""
        uint_img=tf.io.decode_jpeg(img,channels=0)
        img=tf.io.encode_jpeg(image=uint_img, format='rgb', quality=100)
        
        return img

    def decode_jpeg(self,img):
        img=tf.io.decode_jpeg(img,channels=3)
        assert len(img.shape) == 3
        assert img.shape[2] == 3

        return img
    
    def _is_png(self,filename):
        """Determine if a file contains a PNG format image."""

        return 'n02105855_2933.JPEG' in filename


    def _is_cmyk(self,filename):
        """Determine if file contains a CMYK JPEG format image."""

        blacklist = ['n01739381_1309.JPEG', 'n02077923_14822.JPEG',
                    'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
                    'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
                    'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
                    'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
                    'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
                    'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
                    'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
                    'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
                    'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
                    'n07583066_647.JPEG', 'n13037406_4650.JPEG']
        return filename.split('/')[-1] in blacklist