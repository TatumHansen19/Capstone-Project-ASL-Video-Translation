import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# File path for gesture data
gesture_data_path = r"C:\Users\tatum\OneDrive\Spring-2025\STG 452\MVPUpdated (1)\CSV Files\AMC.csv"

# Check if the file exists
if not os.path.exists(gesture_data_path):
    raise FileNotFoundError(f"The file {gesture_data_path} does not exist. Please provide a valid path to the gesture data.")

# Load the CSV data
data = pd.read_csv(gesture_data_path)

# Check if the dataset is empty
if data.empty:
    raise ValueError("The dataset is empty. Please collect gesture data before training the model.")

# Debugging: Print dataset head and shape
print("Dataset loaded successfully.")
print(data.head())  # Print the first few rows of the dataset
print(f"Dataset shape: {data.shape}")  # Print the dataset's shape

# Split data into features and labels
X = data.iloc[:, :-1].values  # Landmarks (features)
y = data.iloc[:, -1].values   # Gesture labels

# Encode labels to numeric
le = LabelEncoder()
y = le.fit_transform(y)

# Ensure there are enough samples to split
if len(X) < 2:
    raise ValueError("Not enough data to split into training and test sets. Please collect more gesture data.")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Determine the number of input features and classes
input_dim = X_train.shape[1]  # Number of features (columns in X)
num_classes = len(le.classes_)  # Number of unique gestures

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(input_dim,)),  # Adjust dynamically based on input dimensions
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer adjusts based on class count
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Starting model training...")
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model for later use
model_path = r"C:\Users\tatum\OneDrive\Spring-2025\STG 452\MVPUpdated (1)\Models\Updated.h5"
model.save(model_path)
print(f"Model training completed and saved to '{model_path}'.")
