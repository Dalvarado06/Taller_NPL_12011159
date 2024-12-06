# Proyecto de Taller de Procesamiento de Lenguaje Natural (NLP)

Este proyecto tiene como propósito mejorar el código existente en un taller de Procesamiento de Lenguaje Natural (NLP) y proveer una versión más robusta y optimizada. A través de este proyecto, se busca realizar mejoras significativas en las técnicas de procesamiento de texto, vectorización y clasificación de datos, utilizando herramientas y librerías de Python.

## Estructura del Proyecto

El proyecto se organiza en dos carpetas principales:

- **Improvement Code**: Contiene las versiones mejoradas del código original del taller. Aquí se aplican las optimizaciones y mejoras para hacer el código más robusto y eficiente.
- **Legacy Code**: Contiene el código original del taller, antes de aplicar las mejoras. Es el punto de partida para comparar el rendimiento y las modificaciones.

## Archivos Generados

Todos los archivos generados durante la ejecución del proyecto, como los modelos entrenados y los resultados de las predicciones, se almacenan en el directorio raíz del proyecto, en la ruta `..`. Esta ruta es utilizada para gestionar todos los archivos de salida y resultados generados por el proyecto.

## Librerías Utilizadas

Este proyecto hace uso de las siguientes librerías esenciales de Python para el procesamiento y análisis de datos:

- **Scikit-learn**: Para las técnicas de modelado y evaluación, como clasificación y métricas de rendimiento.
- **Pickle**: Para guardar y cargar modelos entrenados, así como para la serialización de objetos en Python.
- **Pandas**: Para el manejo de datos estructurados y el análisis de los mismos, como la carga y manipulación de datasets.
- **NLTK (Natural Language Toolkit)**: Para el procesamiento de texto y la manipulación de datos de lenguaje natural, como tokenización y eliminación de stopwords.
- **BeautifulSoup (bs4)**: Para la limpieza y extracción de datos de documentos HTML, si es necesario.
- **SciPy**: Para el análisis y la optimización matemática, utilizado en algunas técnicas de preprocesamiento y evaluación.

## Instrucciones de Uso

1. Clona el repositorio en tu máquina local o en tu entorno de desarrollo.
2. Asegúrate de tener las librerías necesarias instaladas. Puedes instalarlas usando el siguiente comando:

   ```bash
   pip install -r requirements.txt
