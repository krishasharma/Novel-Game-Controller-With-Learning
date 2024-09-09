# save_basic_model.py

from src.models.basic_model import BasicModel

# Initialize the input shape and number of categories
input_shape = (150, 150, 3)  # Example input shape (height, width, channels)
categories_count = 3  # Example number of categories

# Create an instance of the BasicModel
basic_model = BasicModel(input_shape, categories_count)

# Save the model to a file

basic_model.save_model('basic_model.keras')  # This will save the model as basic_model.keras