# FarmaCheck IA — primera publicación web

Prototipo web académico para capturar o subir fotografías de empaques farmacéuticos desde navegador.

## Esta primera versión permite
- Abrir la cámara del celular desde una página HTTPS.
- Subir una imagen existente.
- Mostrar la fotografía recibida.
- Revisar nitidez e iluminación con OpenCV.
- Mostrar claramente que OCR y CNN todavía están pendientes de integración en esta primera publicación.

## Publicar en Streamlit Community Cloud

1. Crea un repositorio público en GitHub, por ejemplo `FarmaCheck-IA`.
2. Sube al directorio raíz del repositorio estos archivos:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
3. Entra a https://share.streamlit.io e inicia sesión con GitHub.
4. Selecciona **Create app** → **Yup, I have an app**.
5. Selecciona el repositorio `FarmaCheck-IA`, la rama `main` y el archivo principal `app.py`.
6. En **Advanced settings**, selecciona Python 3.12.
7. Presiona **Deploy**.

Streamlit Community Cloud generará una dirección HTTPS bajo `streamlit.app`. Abre esa dirección en Safari o Chrome del celular y permite el uso de la cámara.

## Desarrollo por etapas

Esta primera publicación es para probar la cámara y el flujo de imagen. La CNN se entrena en Google Colab y el OCR se integrará después. No incluyas el dataset, credenciales, imágenes privadas ni el archivo del modelo en este primer repositorio.

## Ejecución local opcional

```bash
pip install -r requirements.txt
streamlit run app.py
```

La ejecución local requiere una computadora; los usuarios finales solo necesitarán el enlace web publicado.

## Alcance y advertencia

El sistema no certifica autenticidad, no analiza la composición química y no sustituye a DIGEMID. Un resultado sin alertas no demuestra que el medicamento sea original.
