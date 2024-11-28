from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split


# Downloading MNIST dataset
mnist = datasets.fetch_openml('mnist_784', version=1)
X = np.array(mnist.data)  # Features (pixel values)
y = np.array(mnist.target).astype(int)  # Labels (0-9 digits)

#Performing datapreprocessing
def data_preprocessing(X, y):
    X_flat = X.reshape(X.shape[0], -1)
    X_flat = X_flat/255.0
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
print("data preprocess successfully...")

# from sklearn.datasets import fetch_openml
# import numpy as np
from PIL import Image

# # Load MNIST data
# mnist = fetch_openml('mnist_784', version=1)
# X, y = np.array(mnist.data), np.array(mnist.target).astype(int)

# Select a sample image (e.g., first test image)
sample_image = X[8].reshape(28, 28) * 255  # Scale to [0, 255]
sample_label = y[8]

# Save image
img = Image.fromarray(sample_image.astype(np.uint8))
img.save(f"sample_digit_{sample_label}.png")
print(f"Saved sample digit {sample_label} as an image.")
