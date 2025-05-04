import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

# Load your pre-trained model
model = tf.keras.models.load_model('shape_recognizer.keras')
class_names = ['circle', 'rectangle', 'triangle']

# Create canvas and image for drawing
canvas_width = 280
canvas_height = 280
image = Image.new("L", (canvas_width, canvas_height), color=255)  # white background
draw = ImageDraw.Draw(image)

# Tkinter window setup
root = tk.Tk()
root.title("Draw a shape")
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

def paint(event):
    x1, y1 = (event.x - 4), (event.y - 4)
    x2, y2 = (event.x + 4), (event.y + 4)
    canvas.create_oval(x1, y1, x2, y2, fill='black')
    draw.ellipse([x1, y1, x2, y2], fill=0)

canvas.bind("<B1-Motion>", paint)

def predict_digit():
    # Resize to 28x28
    img = image.resize((28, 28))
    img = np.expand_dims(img, axis=(0, -1))  # shape (1, 28, 28, 1)

    # Predict
    pred = model.predict(img)
    class_id = np.argmax(pred, axis=1)[0]
    label.config(text=f"Predicted: {class_names[class_id]}")

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill=255)
    label.config(text="Draw a shape")

# Buttons
btn_predict = tk.Button(root, text="Predict", command=predict_digit)
btn_predict.pack()

btn_clear = tk.Button(root, text="Clear", command=clear_canvas)
btn_clear.pack()

label = tk.Label(root, text="Draw a shape", font=("Helvetica", 16))
label.pack()

root.mainloop()