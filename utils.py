import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from neural_network import Net


def scale_coordinates(normalized_x, normalized_y,
                      original_image_size):
    original_width, original_height = original_image_size
    scaled_x = int(normalized_x * original_width)
    scaled_y = int(normalized_y * original_height)
    return scaled_x, scaled_y


def draw_cross(draw, center, color="red", size=50, thickness=20):
    x, y = center
    line1 = [(x - size, y - size), (x + size, y + size)]
    line2 = [(x + size, y - size), (x - size, y + size)]
    draw.line(line1, fill=color, width=thickness)
    draw.line(line2, fill=color, width=thickness)


def draw_legend(image, font_size=40, padding=20):
    draw = ImageDraw.Draw(image)

    try:
        font_bold = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            font_size)
    except Exception as e:
        font_bold = ImageFont.load_default(font_size)
        print(f"Using default font due to error: {e}")

    try:
        font_regular = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", font_size)
    except Exception as e:
        font_regular = ImageFont.load_default(font_size)
        print(f"Using default font due to error: {e}")

    segments = [
        {"text": "X", "font": font_bold, "color": "red"},
        {"text": " = Predicted,", "font": font_regular, "color": "black"},
        {"text": "X", "font": font_bold, "color": "green"},
        {"text": " = Actual", "font": font_regular, "color": "black"},
    ]

    x_pos = padding
    y_pos = padding

    for segment in segments:
        text = segment["text"]
        font = segment["font"]
        color = segment["color"]

        draw.text((x_pos, y_pos), text, font=font, fill=color)
        x_pos += font.getmask(text).getbbox()[2]
        if text == segments[1]['text']:
            x_pos += 20


class SpotTheBallUtils:
    def __init__(self, model_path, max_width=4000, max_height=4000):
        self.model_path = model_path
        self.max_width = max_width
        self.max_height = max_height

        # Basic normalisation of images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def load_model(self):
        model = Net()
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)

    def unscale_coordinates(self, x, y):
        return x / self.max_width, y / self.max_height
