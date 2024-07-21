import tensorflow as tf
import numpy as np

class Model():
    def __init__(self):
        self.name = 'Model'

    def __call__(self, x):
        return self.model(x)

    def model(self, input_shape):
        # Input layer
        inputs = tf.keras.Input(shape=input_shape)
        
        # 1st encoding layer
        s1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        s1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(s1)
        p1 = tf.keras.layers.MaxPool2D((2, 2))(s1)
        
        # 2nd encoding layer
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
        p2 = tf.keras.layers.MaxPool2D((2, 2))(c2)
        
        # 3rd encoding layer
        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
        p3 = tf.keras.layers.MaxPool2D((2, 2))(c3)
        
        # 4th encoding layer
        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
        p4 = tf.keras.layers.MaxPool2D((2, 2))(c4)
        
        # Bottleneck layer
        b = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
        b = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(b)
        
        # 1st decoding layer
        d1 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(b)
        d1 = tf.keras.layers.concatenate([d1, c4])
        d1 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(d1)
        d1 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(d1)
        
        # 2nd decoding layer
        d2 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(d1)
        d2 = tf.keras.layers.concatenate([d2, c3])
        d2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(d2)
        d2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(d2)
        
        # 3rd decoding layer
        d3 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(d2)
        d3 = tf.keras.layers.concatenate([d3, c2])
        d3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(d3)
        d3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(d3)
        
        # 4th decoding layer
        d4 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d3)
        d4 = tf.keras.layers.concatenate([d4, s1])
        d4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(d4)
        d4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(d4)
        
        # Output layer
        output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(d4)
        
        model = tf.keras.models.Model(inputs=[inputs], outputs=[output])
        return model

if __name__ == "__main__":
    model_instance = Model()
    unet_model = model_instance.model((128, 128, 1))
    unet_model.summary()
