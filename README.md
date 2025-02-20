# 🌐 Detect AI vs Human-Generated Images

A machine learning-based project for classifying images as either AI-generated or human-generated. The model uses state-of-the-art image classification techniques to determine the authenticity of images.

## 📌 Features

✅ **Image Classification** – Detects whether an image is AI-generated or human-generated.  
✅ **Deep Learning Model** – Uses CNNs and transfer learning for classification.  
✅ **Kaggle Competition** – Part of the WAI X Shutterstock Kaggle challenge.  
✅ **Cloud Deployment** – Deployable as a web API, Android app (TensorFlow Lite), or TensorFlow.js for web applications.

## 🖼️ Dataset

The dataset contains authentic images sampled from the Shutterstock platform paired with their AI-generated counterparts. It includes 79,950 training images and 19,986 test images.

- **Classes**: 
  - 0: Human-generated
  - 1: AI-generated  

📌 Dataset Source: [Kaggle Challenge Dataset](https://www.kaggle.com/competitions/detect-ai-vs-human-generated-images/data)

## 🛠️ Tech Stack

- **Python 3.11**
- **TensorFlow/Keras** (for model training)
- **Flask/FastAPI** (for API deployment)
- **OpenCV/PIL** (for image preprocessing)
- **NumPy, Pandas, Matplotlib** (for data analysis)
- **TensorFlow Lite (TFLite)** (for mobile app deployment)
- **Google Cloud/AWS** (for cloud hosting)

## 🚀 Installation & Setup

1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/yourusername/Detect-AI-vs-Human-Generated-Images.git
cd Detect-AI-vs-Human-Generated-Images
2️⃣ Create Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate  # For Window
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Train the Model (Optional)
If you want to retrain the model, run:
python train.py
This will save the model as image_classifier_model.h5.

📡 Deployment Options

🌐 Web API (FastAPI)
Run the API locally:
uvicorn app:app --host 0.0.0.0 --port 8000
Send a test request:
curl -X POST -F "file=@test_image.jpg" http://127.0.0.1:8000/predict/
📱 Android App (TensorFlow Lite)
Convert the model to TFLite format:
tflite_convert --saved_model_dir=image_classifier_model/ --output_file=image_classifier_model.tflite
Integrate it into an Android app using ML Kit.

☁️ Cloud Deployment

Google Cloud Run (for scalable API hosting)
AWS Lambda + API Gateway (serverless API)
Firebase Hosting (for web app)
📌 Example Usage
Using Python:
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("image_classifier_model.h5")

def predict_image(image_path):
    image = Image.open(image_path).resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape((1, 224, 224, 3))
    prediction = model.predict(img_array)
    return prediction

print(predict_image("test_image.jpg"))
📷 Screenshots
Predictions on Various Images

Healthy Human vs AI-generated images
🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repo
Create a new branch (feature-new-idea)
Commit your changes
Submit a Pull Request
📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

💡 Acknowledgments
Women in AI and Shutterstock for organizing this challenge.
Kaggle for providing the platform and dataset.
Pre-trained models and research papers on AI image generation and classification.
📩 Contact
For questions or suggestions, reach out:
📧 Email: abenezeralz659@gmail.com
GitHub: Abena-3565
This format gives a clear and structured overview of the project, its features, tech stack, setup, usage, and deployment options. Let me know if you'd like any further adjustments!
