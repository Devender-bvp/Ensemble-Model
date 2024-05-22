    if self.image:
            image_array = np.array(self.image)
            # Convert grayscale images to RGB
            if len(image_array.shape) == 2:
                image_array = np.stack((image_array,) * 3, axis=-1)
            elif image_array.shape[-1] == 1:
                # Convert single-channel images to RGB
                image_array = np.repeat(image_array, 3, axis=-1)
            # Resize image to match the expected input shape of your model
            resized_image = resize(image_array, (227, 227, 3))  # Assuming the model expects input of size 227x227x3
            # Normalize pixel values to [0, 1]
            normalized_image = resized_image / 255.0
            return normalized_image
        else:
            return None