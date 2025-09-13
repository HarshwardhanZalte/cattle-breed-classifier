# train.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix
import itertools

# CONFIG
TRAIN_DIR = "Dataset"         
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 5
FINE_TUNE_AT = -20            

os.makedirs("artifacts", exist_ok=True)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    color_mode="rgb",
    shuffle=True
)

val_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    color_mode="rgb",
    shuffle=False
)

num_classes = train_data.num_classes
print(f"Found {train_data.samples} training images and {val_data.samples} validation images across {num_classes} classes.")

labels_path = os.path.join("artifacts", "labels.json")
with open(labels_path, "w") as f:
    json.dump(train_data.class_indices, f)
print(f"Saved label mapping to {labels_path}")

# Build model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# Callbacks
checkpoint = ModelCheckpoint(
    filepath=os.path.join("artifacts", "best_model.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)

# Initial training
history = model.fit(
    train_data,
    epochs=INITIAL_EPOCHS,
    validation_data=val_data,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# Fine-tune
base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

print("Starting fine-tuning...")
fine_history = model.fit(
    train_data,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history.epoch[-1] + 1 if len(history.epoch) else 0,
    validation_data=val_data,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# Save final model
final_model_path = os.path.join("artifacts", "cattle_breed_model.h5")
model.save(final_model_path)
print(f"Saved final model to {final_model_path}")

val_data.reset()
y_true = val_data.classes
y_prob = model.predict(val_data, verbose=1)
y_pred = np.argmax(y_prob, axis=1)
class_indices = train_data.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}

# classification report
report = classification_report(y_true, y_pred, target_names=[idx_to_class[i] for i in range(num_classes)])
with open(os.path.join("artifacts", "classification_report.txt"), "w") as f:
    f.write(report)
print("Saved classification report to artifacts/classification_report.txt")
print(report)

# confusion matrix plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, [idx_to_class[i] for i in range(num_classes)], rotation=45, ha='right')
plt.yticks(tick_marks, [idx_to_class[i] for i in range(num_classes)])
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig(os.path.join("artifacts", "confusion_matrix.png"))
print("Saved confusion matrix to artifacts/confusion_matrix.png")
