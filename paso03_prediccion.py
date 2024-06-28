import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import pandas as pd

# Cargar el modelo entrenado
model = load_model('monkey_species_classifier.h5')

# Cargar el mapeo de índices de clase a nombres de clase
class_indices_path = 'class_indices.json'

try:
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
        print("Class indices loaded:", class_indices)
except FileNotFoundError:
    print(f"Error: File {class_indices_path} not found.")
    exit(1)

# Invertir el diccionario para obtener {índice: etiqueta}
index_to_label = {str(v): k for k, v in class_indices.items()}
print("Index to label mapping:", index_to_label)

# Cargar las etiquetas y nombres comunes desde el archivo
columns = ["Label", "Common Name", "Train Images", "Validation Images"]
df = pd.read_csv("D:/UNA-PUNO/MACHINE LEARNING/monkeys/monkey_labels.txt", names=columns, skiprows=1)
df['Label'] = df['Label'].str.strip()
df['Common Name'] = df['Common Name'].str.strip()

# Crear un diccionario que mapea las etiquetas a los nombres comunes
label_to_common_name = df.set_index("Label")["Common Name"].to_dict()
print("Label to common name mapping:", label_to_common_name)

# Función para preprocesar la imagen
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: The image file {image_path} does not exist.")
        return None
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load image {image_path}.")
        return None
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Función para predecir la clase de la imagen
def predict_image(image_path):
    img = preprocess_image(image_path)
    if img is None:
        return "Error in image preprocessing."
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    print("Predicted class index:", predicted_class)
    label = index_to_label[str(predicted_class[0])]
    common_name = label_to_common_name[label]
    return common_name

# Ejemplo de uso
image_path = 'image.jpg'
print(f'Predicted Class: {predict_image(image_path)}')
