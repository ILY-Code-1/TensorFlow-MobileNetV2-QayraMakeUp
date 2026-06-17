import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import custom_object_scope


def focal_loss(gamma=2.0):
    def _focal(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = K.pow(1.0 - y_pred, gamma) * y_true
        return K.sum(weight * cross_entropy, axis=-1)
    return _focal


with custom_object_scope({'_focal': focal_loss()}):
    model = tf.keras.models.load_model("skin_model_finetuned.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("skin_condition.tflite", "wb") as f:
    f.write(tflite_model)

print("skin_condition.tflite berhasil dibuat")
