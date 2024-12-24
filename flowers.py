import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Veri yolu
dataset_path = "flower_photos"

print(f"Dataset Path Exists: {os.path.exists(dataset_path)}")


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Parametreler
img_height, img_width = 150, 150  # Görüntü boyutları
batch_size = 32
num_classes = 5  # Daisy, Dandelion, Roses, Sunflowers, Tulips
nb_epoch = 20

# Flowers Dataset yolunu belirtin (klasör yapısı: flower_photos/daisy, flower_photos/dandelion, ...)
data_dir = "flower_photos"  # Path to your dataset

# Veri artırma (Data Augmentation)
data_gen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalizasyon
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,  # Eğitim/Doğrulama veri ayrımı
)

# Eğitim ve doğrulama veri setlerini oluşturma
train_generator = data_gen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
)

val_generator = data_gen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
)

# Model oluşturma
model = Sequential([
    Input(shape=(img_height, img_width, 3)),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax"),
])

# Model özeti
model.summary()

# Modeli derleme
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Model eğitimi
history = model.fit(
    train_generator,
    epochs=nb_epoch,
    validation_data=val_generator,
    verbose=1,
)

# Eğitim ve doğrulama sonuçlarının görselleştirilmesi
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))

# Doğruluk grafiği
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Eğitim Doğruluğu')
plt.plot(val_acc, label='Doğrulama Doğruluğu')
plt.title('Doğruluk Grafiği')
plt.xlabel('Epochs')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid()

# Loss grafiği
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Eğitim Kaybı')
plt.plot(val_loss, label='Doğrulama Kaybı')
plt.title('Kayıp Grafiği')
plt.xlabel('Epochs')
plt.ylabel('Kayıp')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Tahmin işlemi
val_generator.reset()
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(val_generator.class_indices.keys()),
            yticklabels=list(val_generator.class_indices.keys()))
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değer')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
class_report = classification_report(
    y_true, y_pred_classes, target_names=list(val_generator.class_indices.keys())
)
print("Classification Report:\n", class_report)
