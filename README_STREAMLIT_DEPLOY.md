# Streamlit companion app for Alzheimer's MRI classification

This package adds a Streamlit inference interface alongside the existing Flask project. It reuses the trained Keras model files and implements image upload, model selection when the Custom CNN file is present, prediction confidence scores, and per-class score visualization.

## Important distinction

`streamlit_app.py` is a new demo interface. It does **not** port the existing Flask registration/login, SQLite-backed patient history, Doctor Dashboard, Admin Panel, or PDF report route. The Streamlit demo does not persist uploaded images or prediction history. The existing `app.py` and its templates remain the original Flask application.

## Files to add to the GitHub repository root

- `streamlit_app.py`
- Replace the existing `requirements.txt` with the provided merged `requirements.txt` (it retains the dependencies used by Flask and adds Streamlit/Pillow/Pandas).

Do not delete `app.py`, `templates/`, `static/`, `.gitattributes`, or the existing `.keras` model files.

## Add files through GitHub's website

1. Open https://github.com/yaswanth6378/alzheimers-detection
2. Click **Add file → Upload files**.
3. Upload `streamlit_app.py` and `README_STREAMLIT_DEPLOY.md`.
4. Upload the new `requirements.txt`, confirming that it replaces the existing file rather than creating a second differently named requirements file.
5. Commit the changes to the `main` branch.

## Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Click **Create app** and choose the repository `yaswanth6378/alzheimers-detection`, branch `main`.
3. Set the main file path to `streamlit_app.py`.
4. Choose Python 3.12 in advanced settings if offered, matching the project's reported development environment and TensorFlow 2.21 compatibility.
5. Click **Deploy** and inspect the app logs if the build fails.

## Model files

The current Flask project tracks `.keras` files using Git LFS. Streamlit Community Cloud supports Git LFS, but confirm the deployed checkout contains the real model files rather than a small Git LFS pointer file. The app also has the same Google Drive fallback ID used by the existing Flask app for the ResNet50 model. The Custom CNN option only appears if `custom_cnn_alzheimer.keras` is present as a complete file.

If the Google Drive file is not public to users with the link, the ResNet50 download fallback will fail; adjust sharing permissions or store the complete model with Git LFS. The TensorFlow wheel is large and model loading may exceed Community Cloud's memory allowance. If that happens, try deploying only one model / a smaller model or use a host with more RAM.

## Local testing

In a Python 3.12 environment:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The model preprocessing mirrors the existing Flask code: RGB conversion, resize to 224x224, and division by 255. A displayed confidence score is a model output for a single image, not overall evaluation accuracy. This app is for education/research and must not be used for clinical diagnosis.
