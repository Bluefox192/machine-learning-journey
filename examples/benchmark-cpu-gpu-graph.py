import tensorflow as tf
import time
import matplotlib.pyplot as plt

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
        model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=0)
        duration = time.time() - start
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        return duration, test_acc

# Simpan hasil
results = {}

# CPU
cpu_time, cpu_acc = run_training("/CPU:0")
results["CPU"] = (cpu_time, cpu_acc)
print(f"🖥️ CPU - waktu: {cpu_time:.2f} detik, akurasi: {cpu_acc:.4f}")

# GPU
if tf.config.list_physical_devices("GPU"):
    gpu_time, gpu_acc = run_training("/GPU:0")
    results["GPU"] = (gpu_time, gpu_acc)
    print(f"🎮 GPU - waktu: {gpu_time:.2f} detik, akurasi: {gpu_acc:.4f}")

# Plot hasil
labels = list(results.keys())
times = [results[d][0] for d in labels]
accs = [results[d][1] for d in labels]

plt.figure(figsize=(10,4))

# Grafik waktu
plt.subplot(1,2,1)
plt.bar(labels, times, color=['skyblue','orange'])
plt.ylabel("Waktu (detik)")
plt.title("Perbandingan Waktu Training")

# Grafik akurasi
plt.subplot(1,2,2)
plt.bar(labels, accs, color=['skyblue','orange'])
plt.ylabel("Akurasi")
plt.title("Perbandingan Akurasi")

plt.suptitle("Benchmark CPU vs GPU TensorFlow", fontsize=14)
plt.tight_layout()
plt.show()
