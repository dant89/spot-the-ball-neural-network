import os
import re
import torch
import utils
from PIL import Image, ImageDraw


def process_and_predict(images_folder, predicted_folder, model,
                        spot_the_ball_utils):
    for filename in os.listdir(images_folder):
        if filename.endswith('.png'):
            print(f"Processing {filename}")

            # Extract correct coordinates and base name from filename
            match = re.match(r'(\d+)--(\d+)-(\d+).png', filename)
            if not match:
                print(f"Filename {filename} does not match pattern.")
                continue

            base_name, correct_x, correct_y = match.groups()
            correct_x, correct_y = int(correct_x), int(correct_y)

            image_path = os.path.join(images_folder, filename)
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)

            # Draw green cross for correct coordinates
            utils.draw_cross(draw, (correct_x, correct_y), color="green")

            # Predict new coordinates
            processed_image = spot_the_ball_utils.preprocess_image(image_path)
            processed_image = processed_image.unsqueeze(0)
            original_image_size = image.size

            with torch.no_grad():
                output = model(processed_image)
                predicted_coordinates = output[0].numpy()

            # Due to image normalisation, the guessed coordinates need to
            # be de-normalised
            new_x, new_y = utils.scale_coordinates(predicted_coordinates[0],
                                                   predicted_coordinates[1],
                                                   original_image_size)

            # Draw red cross for predicted coordinates
            utils.draw_cross(draw, (new_x, new_y), color="red")

            utils.draw_legend(image)

            # Save the modified image
            save_path = os.path.join(predicted_folder,
                                     f"{base_name}--"
                                     f"{correct_x}-{correct_y}--"
                                     f"{new_x}-{new_y}.png")
            image.save(save_path)
            print(f"Processed and saved: {save_path}")


model_path = "spot_the_ball_model.pth"
images_training_folder = "images_training/"
predicted_images_folder = "images_raw/prediction/"

# Ensure the predicted images directory exists
if not os.path.exists(predicted_images_folder):
    os.makedirs(predicted_images_folder)

spot_the_ball_utils = utils.SpotTheBallUtils(model_path)
model = spot_the_ball_utils.load_model()

process_and_predict(images_training_folder, predicted_images_folder, model,
                    spot_the_ball_utils)
