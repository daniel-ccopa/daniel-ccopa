import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import json
import os
import pandas as pd

# Cargar el modelo entrenado
model = load_model('monkey_species_classifier.h5')

# Cargar el mapeo de índices de clase a nombres de clase
class_indices_path = 'class_indices.json'

try:
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
except FileNotFoundError:
    st.error(f"Error: File {class_indices_path} not found.")
    st.stop()

# Invertir el diccionario para obtener {índice: etiqueta}
index_to_label = {str(v): k for k, v in class_indices.items()}

# Cargar las etiquetas y nombres comunes desde el archivo
columns = ["Label", "Common Name", "Train Images", "Validation Images"]
df = pd.read_csv("D:/UNA-PUNO/MACHINE LEARNING/monkeys/monkey_labels.txt", names=columns, skiprows=1)
df['Label'] = df['Label'].str.strip()
df['Common Name'] = df['Common Name'].str.strip()

# Crear un diccionario que mapea las etiquetas a los nombres comunes
label_to_common_name = df.set_index("Label")["Common Name"].to_dict()

# Diccionario con descripciones de cada tipo de mono y su imagen correspondiente
monkey_descriptions = {
    'aullador_de_manto': {
        'description': 'El aullador de manto, nativo de América del Sur, destaca por su distintivo pelaje y su llamado resonante que atraviesa la selva.',
        'image_path': 'images/aullador_de_manto.jpg'
    },
    'mono_patas': {
        'description': 'Originario de África, el mono patas se distingue por sus largas extremidades, adaptadas para la vida en las copas de los árboles de la selva tropical.',
        'image_path': 'images/mono_patas.jpg'
    },
    'uakari_calvo': {
        'description': 'Este primate amazónico es reconocido por su rostro rojo intenso y su escasa cobertura de pelo, habitando en las profundidades de la selva tropical.',
        'image_path': 'images/uakari_calvo.jpg'
    },
    'macaco_japones': {
        'description': 'Conocido por su inteligencia y comportamiento social, el macaco japonés es una especie emblemática que se encuentra en las islas de Japón.',
        'image_path': 'images/macaco_japones.jpg'
    },
    'titi_pigmeo': {
        'description': 'Pequeño pero ágil, el tití pigmeo es un habitante de la selva amazónica conocido por su diminuto tamaño y su estilo de vida arbóreo.',
        'image_path': 'images/titi_pigmeo.jpg'
    },
    'capuchino_de_cabeza_blanca': {
        'description': 'Este mono, caracterizado por el mechón blanco en su cabeza, es una especie activa y sociable que se encuentra en América Central y del Sur.',
        'image_path': 'images/capuchino_de_cabeza_blanca.jpg'
    },
    'tití_plateado': {
        'description': 'Con su brillante pelaje plateado, el tití plateado es un habitante de los bosques tropicales que se desplaza ágilmente entre las ramas en busca de alimento.',
        'image_path': 'images/tití_plateado.jpg'
    },
    'mono_ardilla_comun': {
        'description': 'Con su agilidad y habilidad para moverse entre las ramas como una ardilla, este mono es común en las selvas de América del Sur.',
        'image_path': 'images/mono_ardilla_comun.jpg'
    },
    'mono_nocturno_de_cabeza_negra': {
        'description': 'Adaptado a la vida nocturna, este mono se distingue por su pelaje oscuro y sus grandes ojos, ideales para la visión nocturna en la densa vegetación de la selva.',
        'image_path': 'images/mono_nocturno_de_cabeza_negra.jpg'
    },
    'langur_de_nilgiri': {
        'description': 'Endémico de las montañas Nilgiri en el sur de la India, este mono se caracteriza por su pelaje oscuro y su hábitat montañoso entre densos bosques tropicales.',
        'image_path': 'images/langur_de_nilgiri.jpg'
    }
}

# Función para preprocesar la imagen
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        st.error(f"Error: The image file {image_path} does not exist.")
        return None
    img = cv2.imread(image_path)
    if img is None:
        st.error(f"Error: Failed to load image {image_path}.")
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
    label = index_to_label[str(predicted_class[0])]
    common_name = label_to_common_name[label]
    return common_name

# Título y barra lateral de la aplicación
st.title("K'usillunaka IA")
st.sidebar.image("images/logo.jpeg", use_column_width=True)
st.sidebar.subheader("Descripción")
st.sidebar.write("Esta aplicación clasifica imágenes de monos en diferentes especies utilizando un modelo de aprendizaje automático.")
st.sidebar.subheader("UNA-PUNO - FINESI")
st.sidebar.write("Correo del desarrollador: danielccopa76@.com")
st.sidebar.write("Teléfono: +519163301154")

# Subir una imagen
uploaded_file = st.file_uploader("Elige una imagen...", type="jpg")

if uploaded_file is not None:
    # Leer y mostrar la imagen
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Guardar temporalmente la imagen subida
    temp_file_path = 'temp_image.jpg'
    image.save(temp_file_path)

    # Preprocesar y predecir la imagen
    predicted_class = predict_image(temp_file_path)
    st.write(f'Especie predecida: {predicted_class.upper()}')

# Descripciones de cada tipo de mono con imágenes de ejemplo
st.header("Descripciones de Monos")
cols = st.columns(2)

for i, (monkey, data) in enumerate(monkey_descriptions.items()):
    col = cols[i % 2]
    with col:
        st.image(data['image_path'], caption=monkey.replace("_", " ").capitalize(), width=150)
        st.write(data['description'])
        st.write("---")  # Línea divisoria
