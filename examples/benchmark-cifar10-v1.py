import tensorflow as tf
import time
import matplotlib.pyplot as plt

# Load dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# CNN sederhana
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def run_training(device, epochs=5):
    with tf.device(device):
        model = build_model()
        start = time.time()
        history = model.fit(x_train, y_train,
                            epochs=epochs,
                            batch_size=128,
                            validation_data=(x_test, y_test),
                            verbose=0)
        duration = time.time() - start
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
    return history, duration, acc

# CPU run
cpu_hist, cpu_time, cpu_acc = run_training('/CPU:0')

# GPU run (cek dulu ketersediaan)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    gpu_hist, gpu_time, gpu_acc = run_training('/GPU:0')
else:
    gpu_hist, gpu_time, gpu_acc = None, None, None

# Plot grafik
plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
plt.plot(cpu_hist.history['loss'], label='CPU Loss')
if gpu_hist: plt.plot(gpu_hist.history['loss'], label='GPU Loss')
plt.title('Training Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

# Accuracy
plt.subplot(1,2,2)
plt.plot(cpu_hist.history['accuracy'], label='CPU Acc')
if gpu_hist: plt.plot(gpu_hist.history['accuracy'], label='GPU Acc')
plt.title('Training Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

plt.tight_layout()
plt.savefig("cifar10_benchmark.png")

print(f"🖥 CPU - waktu: {cpu_time:.2f} detik, akurasi: {cpu_acc:.4f}")
if gpu_hist:
    print(f"🎮 GPU - waktu: {gpu_time:.2f} detik, akurasi: {gpu_acc:.4f}")
else:
    print("⚠️ GPU tidak tersedia")
