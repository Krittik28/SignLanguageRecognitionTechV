import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

loaded_model = tf.keras.models.load_model('trained_model.h5')

test_data_dir = 'python/TechVariable/main_dataset/testing'

batch_size = 32
image_size = (200, 200)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

test_loss, test_accuracy = loaded_model.evaluate(test_generator)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

num_batches = len(test_generator)
y_true = []
y_pred = []

for i in range(num_batches):
    batch_data, batch_labels = test_generator[i]
    batch_predictions = loaded_model.predict(batch_data)
    
    batch_true_labels = np.argmax(batch_labels, axis=1)
    batch_pred_labels = np.argmax(batch_predictions, axis=1)
    
    y_true.extend(batch_true_labels)
    y_pred.extend(batch_pred_labels)

class_names = list(test_generator.class_indices.keys())
print("Class Names:", class_names)

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
