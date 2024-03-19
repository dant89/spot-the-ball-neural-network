Spot-the-Ball Neural Network
============================

This Python project utilises neural networks to train a model that can 
predict the location of balls in sports photographs, specifically 
scenarios where the ball is not visible.

The goal? To effectively predict the balls position in images where it 
has been digitally removed.

## Environment Setup

Follow these steps to create and activate a virtual environment:

1. python -m venv venv
2. source venv/bin/activate
3. pip install -r requirements.txt

## How To

### Populate Training Images

**Collect Images** (collect_images.py): Use to fetch 'spot the ball' images 
from a specified source. Images are stored temporarily on the local disk in two 
directories:

- `images_raw/{id}.png`
- `images_training/{id}--{x_coord}-{y_coord}.png` for training images, where filenames include the correct ball coordinates.

### Generate Trained Model

**Generate a Trained Model** (trainer.py): Execute to train your model using the 
collected images. The script outputs a model file that will be used for making 
predictions.

### Predict Spot The Ball Positions

With a trained model, you can predict the ball's location in new images.

**Training Predictor** (training_predictor.py): This script processes images 
from the images_training directory, which contain known ball positions. It 
overlays both the actual and predicted ball locations on the images, marking
them with crosses of different colors. It's particularly useful for 
evaluating the model's accuracy.

**General Predictor** (predictor.py): Use this script to predict the ball's 
location in images without known coordinates. Predictions are marked with a
cross on the output images.

### Improvements

This neural network serves as a starting point, there is ample room for 
further optimisation. Enhancing the models training process and predictions
is key to improving the accuracy. 
