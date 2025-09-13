# 🐄 Cattle Breed Identification

A deep learning project that identifies cattle breeds from images using a Convolutional Neural Network (MobileNetV2) and provides useful breed information.
The project also includes a Streamlit frontend where you can upload an image and get instant predictions.


## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/HarshwardhanZalte/cattle-breed-classifier.git
   cd cattle-breed-classifier
   ```
2. Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate   # On Linux/Mac
    .venv\Scripts\activate      # On Windows
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Train Model
    ```bash
    python train.py
    ```
5. Start the Streamlit app
    ```bash
    streamlit run app.py
    ```

## 🐂 Supported Breeds

- Ayrshire
- Gir
- Holstein Friesian
- Brown Swiss
- Nagpuri
- Nagori
- Jaffrabadi

## 📸 Demo
![App Screenshot](demo_screenshot.png)
