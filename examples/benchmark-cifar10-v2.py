import tensorflow as tf
import time
import matplotlib.pyplot as plt

# Pastikan TensorFlow detect GPU
print("Available GPUs:", tf.config.list_physical_devices("GPU"))

# Load dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Model CNN sederhana
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# Fungsi training + timing
def benchmark(device):
    with tf.device(device):
        model = create_model()
        start = time.time()
        history = model.fit(
            x_train, y_train,
            epochs=3,
            batch_size=512,
            validation_data=(x_test, y_test),
            verbose=0
        )
        elapsed = time.time() - start
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return elapsed, test_acc

# Benchmark CPU & GPU
results = {}
for device in ["/CPU:0", "/GPU:0"]:
    try:
        elapsed, acc = benchmark(device)
        print(f"{device} -> waktu: {elapsed:.2f} detik, akurasi: {acc:.4f}")
        results[device] = (elapsed, acc)
    except Exception as e:
        print(f"{device} tidak tersedia: {e}")

# Plot hasil
devices = list(results.keys())
times = [results[d][0] for d in devices]
accs = [results[d][1] for d in devices]

fig, ax1 = plt.subplots()

# Grafik waktu
ax1.set_xlabel("Device")
ax1.set_ylabel("Waktu (detik)", color="tab:red")
ax1.bar(devices, times, color="tab:red", alpha=0.6, label="Waktu (detik)")
ax1.tick_params(axis="y", labelcolor="tab:red")

# Grafik akurasi
ax2 = ax1.twinx()
ax2.set_ylabel("Akurasi", color="tab:blue")
ax2.plot(devices, accs, color="tab:blue", marker="o", label="Akurasi")
ax2.tick_params(axis="y", labelcolor="tab:blue")

plt.title("Benchmark CIFAR-10: CPU vs GPU")
plt.show()
