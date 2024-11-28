# Import necessary functions and libraries
from data_preprocess import data_preprocessing
from model import train_model  
from prediction import predict_image  
from evaluation import evaluate_model 
from joblib import load  
from sklearn import datasets
import numpy as np

# Step 1: Load Data
mnist = datasets.fetch_openml('mnist_784', version=1)
X, y = np.array(mnist.data), np.array(mnist.target).astype(int)


# Step 2: Preprocess Data
X_train, X_test, y_train, y_test = data_preprocessing(X, y)

# Step 3: Train Model (Optional if the model is already trained and saved)
# Uncomment the following line to train a new model:
model = train_model(X_train, y_train)

# Step 4: Load Trained Model
model = load("saved_model/digit_classifier_model.joblib")  # Load existing model

# Step 5: Evaluate Model
evaluate_model(model, X_test, y_test)  # Evaluate the performance on test data

# Step 6: Predict on Custom Image
image_path = "Image/test_images/sample_digit_1.png"  # Example test image path
predicted_label = predict_image(image_path)  # Predict digit for the custom image
print(f"Predicted Label for the Image: {predicted_label}")
