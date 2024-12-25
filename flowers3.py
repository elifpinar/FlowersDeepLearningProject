import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image

# Parametreler
img_height, img_width = 150, 150  # Görüntü boyutları
batch_size = 64
num_classes = 5  # Daisy, Dandelion, Roses, Sunflowers, Tulips
nb_epoch = 20


data_dir = "flower_photos"  
def load_balanced_data(data_dir, img_height, img_width):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    min_samples = min([len(os.listdir(os.path.join(data_dir, cls))) for cls in class_names])
    
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        class_images = glob.glob(class_dir + "/*.jpg")[:min_samples]  #Minimum örnek kadar al
        for image_path in class_images:
            img = Image.open(image_path).resize((img_width, img_height))
            img_array = np.array(img) / 255.0  #Normalizasyon
            images.append(img_array)
            labels.append(idx)

    images, labels = shuffle(np.array(images), np.array(labels))
    return images, to_categorical(labels, num_classes=len(class_names))

X, y = load_balanced_data(data_dir, img_height, img_width)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


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

#Model özeti
model.summary()

#Modeli derleme
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



early_stopping = EarlyStopping(
    monitor="val_loss", 
    patience=5,  #İyileşme olmadan geçecek epoch sayısı
    restore_best_weights=True  
)

#Model eğitimi
history = model.fit(
    X_train, y_train,
    epochs=nb_epoch,
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[early_stopping],  # Callbacks parametresine ekleme
)

#Eğitim ve doğrulama sonuçlarının görselleştirilmesi
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))

#Doğruluk grafiği
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Eğitim Doğruluğu')
plt.plot(val_acc, label='Doğrulama Doğruluğu')
plt.title('Doğruluk Grafiği')
plt.xlabel('Epochs')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid()

#Loss grafiği
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

#Tahmin 
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

#Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(os.listdir(data_dir)),
            yticklabels=sorted(os.listdir(data_dir)))
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değer')
plt.title('Confusion Matrix')
plt.show()


class_report = classification_report(
    y_true, y_pred_classes, target_names=sorted(os.listdir(data_dir))
)
print("Classification Report:\n", class_report)
