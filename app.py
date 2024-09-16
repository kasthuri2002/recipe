from flask import Flask, render_template, request, redirect
import os
import pandas as pd
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__, template_folder='templates')

# Define the directory for model and CSV files
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the recipe model
model_file_path = os.path.join(base_dir, r"C:\Users\PPM\OneDrive\anu\OneDrive\Documents\flask_app\recipemodel .h5")
if not os.path.exists(model_file_path):
    raise FileNotFoundError(f"The file {model_file_path} does not exist.")
model = load_model(model_file_path)

# Load the ingredients data
csv_file_path = os.path.join(base_dir, r"C:\Users\PPM\OneDrive\anu\OneDrive\Documents\flask_app\ingrediant.csv")
if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"The file {csv_file_path} does not exist.")
ingredients = pd.read_csv(csv_file_path, encoding='latin1')

# Ensure the directory for uploaded images exists
uploads_dir = os.path.join(base_dir, 'static', 'uploads')
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image to 150x150 pixels
    image = image.resize((150, 150))
    # Convert image to array and normalize pixel values
    image = np.array(image) / 255.0
    return image

# Define a function to predict the recipe from an image
def predict_recipe(image_path):
    # Load the image
    image = Image.open(image_path)
    if image is None:
        raise ValueError("Unable to read the image file.")
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Make the prediction
    predictions = model.predict(np.expand_dims(preprocessed_image, axis=0))
    
    # Get the index of the highest prediction
    predicted_recipe_index = np.argmax(predictions)
    
    # Return the name of the recipe
    return ingredients.iloc[predicted_recipe_index]['Dish']

# Define the route for the home page
@app.route('/')
def home():
    return redirect('/upload')

# Define the route for the upload page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            # Get the uploaded image
            file = request.files['image']
            
            # Save the image to the local machine
            file.save(os.path.join(uploads_dir, file.filename))
            
            # Get the path of the uploaded image
            image_path = os.path.join(uploads_dir, file.filename)
            
            # Predict the recipe from the image
            predicted_recipe = predict_recipe(image_path)
            
            # Redirect to the result page with the predicted recipe as a parameter
            return redirect(f'/result/{predicted_recipe}')
        
        except Exception as e:
            return str(e)  # Return the error message if any error occurs

    return render_template('upload.html')

# Define the route for the result page
@app.route('/result/<recipe>')
def result(recipe):
    # Get the recipe details from the CSV file based on the predicted recipe
    recipe_details = ingredients[ingredients['Dish'] == recipe].to_dict('records')
    
    if not recipe_details:
        return "Recipe details not found."
    
    # Return the result template with the recipe details
    return render_template('result.html', recipe=recipe_details[0])

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
