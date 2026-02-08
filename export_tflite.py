import tensorflow as tf

model = tf.keras.models.load_model("skin_model_finetuned.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("skin_condition.tflite", "wb") as f:
    f.write(tflite_model)

print("skin_condition.tflite berhasil dibuat")
