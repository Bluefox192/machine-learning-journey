import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import time

# Cek device
print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# Load dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build model sederhana
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Latih model
start = time.time()
history = model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=1)
end = time.time()

# Evaluasi
loss, acc = model.evaluate(x_test, y_test, verbose=0)

print(f"\n✅ Training selesai dalam {end-start:.2f} detik")
print(f"🎯 Akurasi test: {acc:.4f}")
