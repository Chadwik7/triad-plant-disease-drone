import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Define paths to your extracted folders
train_dir = r"C:\Users\rithv\Documents\decease detection\Plant_Disease_Dataset\train"
val_dir = r"C:\Users\rithv\Documents\decease detection\Plant_Disease_Dataset\valid"

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# 2. Load Training Dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# 3. Load Validation Dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print("Loaded Classes:", class_names)

# 4. Normalize pixel values (scale from 0-255 to 0.0-1.0)
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# 5. Build CNN Model Architecture
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train the Model (Start with 5-10 epochs for testing)
epochs = 5
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# 7. Save the trained model
model.save("plant_disease_model.keras")
print("Model training complete and saved as plant_disease_model.keras!")