import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Definir rutas para las imágenes
ruta = {
    'persona1': r'C:\\Users\\usuario\\Downloads\\imagenesFacial\\persona1',
    'persona2': r'C:\\Users\\usuario\\Downloads\\imagenesFacial\\persona2',
    'persona3': r'C:\\Users\\usuario\\Downloads\\imagenesFacial\\persona3'
}

# Función para cargar imágenes de una ruta
def cargar_imagenes(ruta):
    imagenes = []
    for nombre_archivo in os.listdir(ruta):
        ruta_imagen = os.path.join(ruta, nombre_archivo)
        imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        if imagen is not None:
            imagen = cv2.resize(imagen, (128, 128))  # Redimensionar
            imagenes.append(imagen)
    return imagenes

# Procesar imágenes y extraer características LBP
def procesar_y_extraer_caracteristicas(imagenes):
    caracteristicas = []
    for imagen in imagenes:
        lbp = local_binary_pattern(imagen, 8, 1, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
        hist = hist / hist.sum()  # Normalizar
        caracteristicas.append(hist)
    return np.array(caracteristicas)

# Cargar datos y etiquetas
datos, etiquetas = [], []
for etiqueta, dir_ruta in ruta.items():
    imagenes = cargar_imagenes(dir_ruta)
    datos.extend(imagenes)
    etiquetas.extend([etiqueta] * len(imagenes))

# Extraer características
caracteristicas_lbp = procesar_y_extraer_caracteristicas(datos)

# Dividir en entrenamiento y prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    caracteristicas_lbp, etiquetas, test_size=0.3, random_state=42
)

# Entrenar modelo KNN
modelo_knn = KNeighborsClassifier(n_neighbors=3)
modelo_knn.fit(X_entrenamiento, y_entrenamiento)

# Conectar con la cámara IP
camara = cv2.VideoCapture('http:192.168.1.33:8080/video')  # Reemplaza la direccion ip
print("Presiona 'a' para finalizar")

# Reconocimiento facial en tiempo real
while camara.isOpened():
    ret, cuadro = camara.read()
    if not ret:
        break

    gris = cv2.cvtColor(cuadro, cv2.COLOR_BGR2GRAY)
    gris_redimensionado = cv2.resize(gris, (128, 128))
    
    # Extraer características
    caracteristicas = procesar_y_extraer_caracteristicas([gris_redimensionado])
    caracteristicas = np.array(caracteristicas).reshape(1, -1)

    # Predicción
    prediccion = modelo_knn.predict(caracteristicas)

    # Mostrar resultado
    cv2.putText(cuadro, prediccion[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Reconocimiento Facial en Tiempo Real', cuadro)

    # Finalizar con 'a'
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

camara.release()
cv2.destroyAllWindows()

# Evaluar modelo
y_pred = modelo_knn.predict(X_prueba)
cm = confusion_matrix(y_prueba, y_pred)

# Mostrar matriz de confusión
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=modelo_knn.classes_, yticklabels=modelo_knn.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
