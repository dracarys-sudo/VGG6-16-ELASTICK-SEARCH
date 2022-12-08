from keras_preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np

class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract(self, img):
        """
        Extract a deep feature from an input image

        """
         # VGG ==> input_image(224x224 )
        img = img.resize((224, 224)) 

         # check  if  image is colored (RGB) 
        img = img.convert('RGB') 
        #Convert  Image to  array (h*w*c)
        x = image.img_to_array(img) 
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x) 
         # Subtracting avg values for each pixel
        feature = self.model.predict(x)[0]  
        # Normalization
        return feature / np.linalg.norm(feature)  