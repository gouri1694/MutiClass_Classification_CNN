import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
class flowertype:
    def __init__(self,filename):
        self.filename =filename


    def predictionflower(self):
        # load model
        # model = load_model('model.h5')
        model=tf.keras.models.load_model(r'C:\Users\DELL\Documents\CNN\CustomImageClassification\flowerRecognition\model.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (128,128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        # print(result[0][4],result)
        # return result
        if result[0][0] == 1:
            prediction = 'daisy'
            return [prediction]
        elif result[0][1] == 1:
            prediction = 'dandelion'
            return [prediction]
        elif result[0][2] == 1:
            prediction = 'rose'
            return [prediction]
        elif result[0][3] == 1:
            prediction = 'sunflower'
            return [prediction]
        else:
            prediction = 'tulip'
            return [prediction]


# filename = r"C:\Users\DELL\Documents\CNN\CustomImageClassification\flowerRecognition\tulip.jpg"
# classifier = flowertype(filename)
# # print(classifier)
# print(classifier.predictionflower())