import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from skimage.transform import resize


class PneumoniaDetectionGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Pneumonia Detection GUI")
        
        self.model = None

        self.create_widgets()

    def create_widgets(self):
        # Load Model button
        self.load_model_button = tk.Button(self.master, text="Load Model", command=self.load_model)
        self.load_model_button.pack()

        # File selection button
        self.load_image_button = tk.Button(self.master, text="Load Image", command=self.load_image)
        self.load_image_button.pack()

        # Image display
        self.image_label = tk.Label(self.master)
        self.image_label.pack()

        # Predict button
        self.predict_button = tk.Button(self.master, text="Predict", command=self.predict)
        self.predict_button.pack()

        # Prediction output
        self.prediction_label = tk.Label(self.master, text="")
        self.prediction_label.pack()

    def load_model(self):
        file_path = filedialog.askopenfilename(title="Select Model File", filetypes=(("HDF5 files", "*.h5"), ("All files", "*.*")))
        if file_path:
            try:
                self.model = tf.keras.models.load_model(file_path)
                messagebox.showinfo("Info", "Model loaded successfully!")
            except:
                messagebox.showerror("Error", "Failed to load model!")

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select Image File", filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
        if file_path:
            self.image = Image.open(file_path)
            # Convert image to RGB if it has fewer than three color channels
            self.image = self.image.convert("RGB")
            self.image = self.image.resize((227, 227))  # Resize image to match model input size
            self.photo = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=self.photo)


    def preprocess_image(self):
        if self.image:
            image_array = np.array(self.image)
            # Convert grayscale images to RGB
            if len(image_array.shape) == 2:
                image_array = np.stack((image_array,) * 3, axis=-1)
            elif image_array.shape[-1] == 1:
                # Convert single-channel images to RGB
                image_array = np.repeat(image_array, 3, axis=-1)
            # Resize image to match the expected input shape of your model
            resized_image = resize(image_array, (227, 227))  # Assuming the model expects input of size 227x227x3
            # Add extra dimension to match the expected input shape (batch dimension)
            resized_image = np.expand_dims(resized_image, axis=0)
            # Normalize pixel values to [0, 1]
            normalized_image = resized_image / 255.0
            return normalized_image
        else:
            return None


    def predict(self):
        if not self.model:
            messagebox.showerror("Error", "Please load a model!")
            return

        preprocessed_image = self.preprocess_image()

        if preprocessed_image is None:
            messagebox.showerror("Error", "Please load an image!")
            return

        # Perform prediction using the loaded model
        if self.model:
            # Assuming self.model.predict() returns a single output value
            prediction_value = self.model.predict(preprocessed_image)[0]
            
            # Define a threshold for considering a positive prediction
            threshold = 0.5

            if prediction_value[0] > threshold:  # Accessing the first element of prediction_value
                prediction = "Positive (COVID-19)"
            else:
                prediction = "Negative (Not COVID-19)"

            self.prediction_label.config(text=f"Prediction: {prediction}")
        else:
            messagebox.showerror("Error", "Please load a model!")




def main():
    root = tk.Tk()
    app = PneumoniaDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
