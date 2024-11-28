import numpy as np
from PIL import Image
from  joblib import load

def predict_image(image_path):
    model = load("saved_model/digit_classifier_model.joblib")
    image = Image.open(image_path).convert("L")
    image = np.array(image.resize((28, 28)))
    image = image.reshape(1, -1)/255.0

    prediction = model.predict(image)
    return prediction[0]


print("predicting...")

