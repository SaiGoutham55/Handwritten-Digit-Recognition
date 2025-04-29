import tkinter as tk
from tkinter import Canvas
import numpy as np
import cv2
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the dataset
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a simple neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, verbose=1)

class DigitRecognizer:
    def _init_(self, root):
        self.root = root
        self.root.title("Draw a Digit")
        
        self.canvas = Canvas(self.root, width=280, height=280, bg='white')
        self.canvas.pack()
        
        self.button_predict = tk.Button(self.root, text="Predict", command=self.predict_digit)
        self.button_predict.pack()
        
        self.button_clear = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()
        
        self.image = Image.new("L", (280, 280), 255)  # Create a blank white image
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind("<B1-Motion>", self.draw_lines)
    
    def draw_lines(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x+10, y+10, fill='black', width=10)
        self.draw.ellipse([x, y, x+10, y+10], fill='black')
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
    
    def predict_digit(self):
        # Resize to 28x28 (Remove Image.ANTIALIAS)
        img_resized = self.image.resize((28, 28))

        # Convert to numpy array
        img_array = np.array(img_resized)

        # Invert colors (MNIST expects white digits on black background)
        img_array = 255 - img_array

        # Normalize the image
        img_array = img_array / 255.0

        # Reshape to fit the model input (1, 28, 28)
        img_array = img_array.reshape(1, 28, 28)

        # Predict the digit
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        # Display the image and prediction
        plt.imshow(img_resized, cmap='gray')
        plt.title(f"Predicted: {predicted_digit}")
        plt.show()
        
if _name_ == "_main_":
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop()
