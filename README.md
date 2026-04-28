# Aircraft Model Recognition API

Proyecto de clasificación de modelos de aeronaves mediante Inteligencia Artificial y visión artificial.  
La aplicación permite enviar una imagen de un avión a una API desarrollada en Python y recibir como respuesta el modelo de aeronave predicho junto con un nivel de confianza.

## Descripción del proyecto

Este proyecto consiste en un prototipo funcional capaz de clasificar imágenes de aviones utilizando un modelo entrenado con Ultralytics YOLO en modo clasificación.

La primera versión del sistema reconoce las siguientes clases:

- A320
- A380
- Boeing_737
- Boeing_747
- Boeing_777

El objetivo principal es demostrar cómo se puede construir una solución completa de visión artificial que incluya preparación de dataset, entrenamiento de modelo, integración en una API y prueba mediante una interfaz web automática.

## Tecnologías utilizadas

El proyecto utiliza las siguientes tecnologías:

- Python: lenguaje principal del proyecto.
- FastAPI: framework utilizado para crear la API.
- Uvicorn: servidor ASGI utilizado para ejecutar la API.
- Ultralytics YOLO: herramienta utilizada para entrenar y ejecutar el modelo de clasificación.
- PyTorch: motor de deep learning utilizado por YOLO.
- OpenCV: lectura y procesamiento de imágenes.
- NumPy: manipulación de imágenes como matrices de datos.
- Pillow: apoyo en el tratamiento de imágenes.
- python-multipart: recepción de archivos subidos mediante formularios HTTP.

## Estructura del proyecto

La estructura principal del proyecto es la siguiente:

```text
aircraft-api/
├── app/
│   ├── main.py
│   ├── aircraft_classifier.py
│   └── image_reader.py
│
├── datasets/
│   ├── aircraft_detection/
│   └── aircraft_models/
│       ├── train/
│       │   ├── A320/
│       │   ├── A380/
│       │   ├── Boeing_737/
│       │   ├── Boeing_747/
│       │   └── Boeing_777/
│       │
│       ├── val/
│       │   ├── A320/
│       │   ├── A380/
│       │   ├── Boeing_737/
│       │   ├── Boeing_747/
│       │   └── Boeing_777/
│       │
│       └── test/
│           ├── A320/
│           ├── A380/
│           ├── Boeing_737/
│           ├── Boeing_747/
│           └── Boeing_777/
│
├── models/
│   └── best.pt
│
├── training/
│   └── train_aircraft_model.py
│
├── runs/
├── requirements.txt
└── README.md
```

## Requisitos previos

Antes de ejecutar el proyecto es necesario tener instalado:

- Python 3.10 o superior.
- pip.
- Conexión a internet para instalar dependencias.
- Dataset de imágenes de aeronaves descargado previamente.

Aunque el proyecto puede ejecutarse en CPU, se recomienda disponer de una GPU compatible si se desea reducir el tiempo de entrenamiento.

## Instalación del entorno

Desde la carpeta raíz del proyecto, crear un entorno virtual:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

Después, instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Dependencias

El archivo `requirements.txt` debe contener:

```text
ultralytics
torch
torchvision
opencv-python
fastapi
uvicorn[standard]
python-multipart
numpy
pillow
```

## Preparación del dataset

El usuario debe descargar previamente el dataset de imágenes de aeronaves y colocarlo en la ruta indicada por el proyecto.

Después, se debe ejecutar el script de preparación del dataset para organizar las imágenes en el formato que necesita Ultralytics YOLO.

El objetivo es obtener la siguiente estructura:

```text
datasets/aircraft_models/
├── train/
│   ├── A320/
│   ├── A380/
│   ├── Boeing_737/
│   ├── Boeing_747/
│   └── Boeing_777/
│
├── val/
│   ├── A320/
│   ├── A380/
│   ├── Boeing_737/
│   ├── Boeing_747/
│   └── Boeing_777/
│
└── test/
    ├── A320/
    ├── A380/
    ├── Boeing_737/
    ├── Boeing_747/
    └── Boeing_777/
```

Cada carpeta representa una clase. Por ejemplo, las imágenes de aviones A320 deben estar dentro de la carpeta `A320`, y las imágenes de Boeing 737 deben estar dentro de `Boeing_737`.

## Entrenamiento del modelo

Una vez preparado el dataset, se puede entrenar el modelo ejecutando:

```bash
python training/train_aircraft_model.py
```

Durante el entrenamiento, YOLO utilizará las imágenes de `train` para aprender y las imágenes de `val` para validar el rendimiento.

Al finalizar, se generará un modelo entrenado en una ruta similar a:

```text
runs/classify/aircraft_model_classifier/weights/best.pt
```

Este archivo representa la mejor versión del modelo obtenida durante el entrenamiento.

## Integración del modelo en la API

Para que la API utilice el modelo entrenado, copiar el archivo:

```text
runs/classify/aircraft_model_classifier/weights/best.pt
```

a la carpeta:

```text
models/best.pt
```

La API cargará el modelo desde esa ruta al iniciarse.

## Prueba del modelo desde consola

Antes de arrancar la API, se recomienda probar el modelo con una imagen del conjunto `test`.

Ejemplo de prueba esperada:

```text
Predicción: A320
Confianza: 0.91
```

Esta prueba permite comprobar que el modelo se carga correctamente y que devuelve una clase válida.

## Ejecución de la API

Para arrancar la API, ejecutar desde la raíz del proyecto:

```bash
uvicorn app.main:app --reload
```

La API quedará disponible en:

```text
http://localhost:8000
```

La documentación automática de FastAPI estará disponible en:

```text
http://localhost:8000/docs
```

## Uso de la API

El endpoint principal es:

```text
POST /analyze
```

Este endpoint permite subir una imagen de un avión y obtener una predicción.

Pasos para probarlo desde el navegador:

1. Abrir `http://localhost:8000/docs`.
2. Buscar el endpoint `POST /analyze`.
3. Pulsar en `Try it out`.
4. Subir una imagen de avión.
5. Ejecutar la petición.
6. Revisar la respuesta generada.

## Ejemplo de respuesta correcta

```json
{
  "aircraft_detected": true,
  "prediction": {
    "model": "A320",
    "confidence": 0.91
  }
}
```

## Formatos admitidos

La API acepta imágenes en los siguientes formatos:

- JPG
- PNG
- WEBP

Si se sube un archivo con un formato no permitido, la API devuelve un error controlado.

Ejemplo:

```json
{
  "detail": "Solo se permiten imágenes JPG, PNG o WEBP"
}
```

## Procedimiento de actualización del modelo

Para actualizar el modelo de clasificación:

1. Añadir nuevas imágenes al dataset.
2. Mantener la estructura `train`, `val` y `test`.
3. Ejecutar de nuevo el entrenamiento.
4. Localizar el nuevo archivo `best.pt`.
5. Copiarlo a `models/best.pt`.
6. Reiniciar la API.

Es recomendable conservar una copia del modelo anterior antes de sustituirlo, por si la nueva versión ofrece peores resultados.

## Posibles mejoras futuras

Algunas mejoras que podrían añadirse en futuras versiones son:

- Añadir más modelos de avión.
- Clasificar también la aerolínea.
- Analizar vídeos además de imágenes.
- Añadir detección previa del avión dentro de la imagen.
- Guardar predicciones en una base de datos.
- Crear una interfaz web o móvil.
- Desplegar la API en un servidor cloud.
- Añadir autenticación de usuarios.

## Limitaciones actuales

La versión actual del proyecto tiene las siguientes limitaciones:

- Solo reconoce las clases entrenadas.
- No identifica aerolíneas.
- No analiza vídeos.
- No detecta automáticamente matrículas.
- La precisión depende directamente de la calidad del dataset.
- Si se sube una imagen muy diferente a las utilizadas en el entrenamiento, el modelo puede fallar.

## Licencia y uso del dataset

El dataset utilizado debe respetar las condiciones de uso indicadas por su fuente original.  
En caso de utilizar FGVC-Aircraft, debe tenerse en cuenta que sus imágenes están destinadas a fines académicos y de investigación no comercial.

## Autor

Proyecto desarrollado como prototipo académico de Inteligencia Artificial y Big Data, centrado en la clasificación automática de aeronaves mediante visión artificial.
