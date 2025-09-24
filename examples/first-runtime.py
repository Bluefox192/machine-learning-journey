import tensorflow as tf
import time

# Buat matrix besar
a = tf.random.normal([3000, 3000])
b = tf.random.normal([3000, 3000])

# Cek device
print("Device yang dipakai:", a.device)

# Hitung perkalian matrix dan ukur waktu
start = time.time()
c = tf.matmul(a, b)
end = time.time()

print("Waktu eksekusi:", end - start, "detik")
