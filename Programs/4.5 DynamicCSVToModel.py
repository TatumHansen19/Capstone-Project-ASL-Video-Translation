import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define sequence length (number of frames per sequence)
SEQUENCE_LENGTH = 20

# Load the motion data for J (stored separately from static signs)
data = pd.read_csv("J_motion.csv")  # Make sure this contains J's motion sequences

# Extract features and labels
X = data.iloc[:, :-1].values  # Hand landmark features
y = data.iloc[:, -1].values   # Labels

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Ensure we have enough frames to form sequences
num_samples = len(X) // SEQUENCE_LENGTH  # Total usable sequences

# Reshape data into sequences of 20 frames each
X = X[:num_samples * SEQUENCE_LENGTH]  # Trim excess data
y = y[:num_samples * SEQUENCE_LENGTH]  # Trim excess labels

# Reshape X to (num_samples, SEQUENCE_LENGTH, 63) for LSTM input
X = X.reshape(num_samples, SEQUENCE_LENGTH, 63)

# Only take one label per sequence (e.g., first frame in each 20-frame sequence)
y = y[::SEQUENCE_LENGTH]  

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")  # Should be (num_samples, 20, 63)
print(f"y_train shape: {y_train.shape}")  # Should match num_samples

# Define LSTM model for J's motion recognition
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 63)),  # 20 frames, 63 features
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# Save the trained model
model.save("DynamicJZModel.h5")
print("Dynamic J model trained and saved successfully.")
