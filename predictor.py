import torch
from PIL import Image, ImageDraw
import os
from utils import SpotTheBallUtils


def predict_and_draw(image_path, model, utils):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Preprocess and predict
    transformed_image = utils.preprocess_image(image_path)

    # Add batch dimension
    transformed_image = transformed_image.unsqueeze(0)
    with torch.no_grad():
        output = model(transformed_image)
        predicted_coordinates = output[0].numpy()

    x_pred, y_pred = utils.scale_coordinates(predicted_coordinates[0], predicted_coordinates[1])

    # Extract the correct coordinates from filename
    basename = os.path.basename(image_path)
    _, x_true, y_true = [int(part) for part in basename.split('--')[1].split('-')]

    utils.draw_cross(draw, (x_true, y_true), color="green")
    utils.draw_cross(draw, (x_pred, y_pred), color="red")

    utils.draw_legend(image)

    return image, x, y


utils = SpotTheBallUtils('spot_the_ball_model.pth')
model = utils.load_model()

images_training_folder = "images_raw/"
predicted_images_folder = "images_raw/predicted/"

if not os.path.exists(predicted_images_folder):
    os.makedirs(predicted_images_folder)

for filename in os.listdir(images_training_folder):
    if filename.endswith('.png'):
        print(f"Processing {filename}")
        image_path = os.path.join(images_training_folder, filename)
        processed_image, x, y = predict_and_draw(image_path, model, utils)
        save_path = os.path.join(predicted_images_folder, f"{filename.split('.')[0]}--{x}-{y}.png")
        processed_image.save(save_path, format='PNG')
        print(f"Processed and saved: {save_path}")
