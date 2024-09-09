# test_transferred_model.py

from src.models.transfered_model import TransferedModel

# Define the input shape and the number of categories
input_shape = (150, 150, 3)  # Example input shape (height, width, channels)
categories_count = 3  # Example number of categories

# Initialize the TransferedModel
transferred_model = TransferedModel(input_shape, categories_count)

# Print the model summary
transferred_model.print_summary()

# Optionally, evaluate the model with test data
# Assuming you have a test_dataset loaded:
# transferred_model.evaluate(test_dataset)