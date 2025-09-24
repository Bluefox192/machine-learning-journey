import tensorflow as tf
import time

# Cek device yang tersedia
print("Available GPUs:", tf.config.list_physical_devices("GPU"))

# Dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Model sederhana
def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

def run_training(device):
    with tf.device(device):
        model = create_model()
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        start = time.time()
        history = model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=0)
        duration = time.time() - start
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        return duration, test_acc

# Jalankan di CPU
cpu_time, cpu_acc = run_training("/CPU:0")
print(f"🖥️ CPU - waktu: {cpu_time:.2f} detik, akurasi: {cpu_acc:.4f}")

# Jalankan di GPU (kalau ada)
if tf.config.list_physical_devices("GPU"):
    gpu_time, gpu_acc = run_training("/GPU:0")
    print(f"🎮 GPU - waktu: {gpu_time:.2f} detik, akurasi: {gpu_acc:.4f}")

