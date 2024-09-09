# test_random_model.py

from src.models.random_model import RandomModel

# Define the input shape and the number of categories
input_shape = (150, 150, 3)  # Example input shape (height, width, channels)
categories_count = 3  # Example number of categories

# Initialize the RandomModel
random_model = RandomModel(input_shape, categories_count)

# Print the model summary
random_model.print_summary()

# Optionally, evaluate the model with test data
# Assuming you have a test_dataset loaded:
# random_model.evaluate(test_dataset)