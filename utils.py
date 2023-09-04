import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def resize_image(image: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
    """Resize an image with maintaining the aspect ratio."""

    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image, None, None

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def generate_vertices(size: float = 0.5) -> list[float, float, float, float, float, float]:
    """Generate random locations of vertices."""
    region_x = np.random.rand()
    region_y = np.random.rand()

    return [np.clip(region_x + np.random.rand() * size * 2 - size, 0, 1) if i % 2 == 0
            else np.clip(region_y + np.random.rand() * size * 2 - size, 0, 1)
            for i in range(6)]


def generate_color() -> list[float, float, float, float]:
    """Generate random RGBA color."""
    return [np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand() * 0.25]


def max_possible_resolution(img: np.ndarray, max_pixels: int) -> tuple[int, int]:
    """Get maximum resolution keeping the original aspect ratio
    and the number of pixels smaller than a given threshold."""
    height, width, _ = img.shape
    aspect_ratio = width / height

    if height * width <= max_pixels:
        return height, width

    if aspect_ratio <= 1:
        new_width = int(np.sqrt(max_pixels * aspect_ratio))
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = int(np.sqrt(max_pixels / aspect_ratio))
        new_width = int(new_height * aspect_ratio)

    return new_height, new_width


def triangle_fields(genome: np.ndarray, img_shape: tuple) -> np.ndarray:
    """Calculate triangle fields on a specific image."""
    height = img_shape[0]
    width = img_shape[1]

    vertices = genome[:, :6].copy()
    vertices[:, ::2] *= width
    vertices[:, 1::2] *= height

    field_func = lambda v: np.abs(0.5 * (v[0] * (v[3] - v[5]) + v[2] * (v[5] - v[1]) + v[4] * (v[1] - v[3])))
    fields = np.apply_along_axis(func1d=field_func, axis=1, arr=vertices)

    return fields


def save_evolution_progress_as_gif(
        img_history: list[tuple[int, np.ndarray]],
        original_img: np.ndarray,
        duration: float,
        output_path: str
):
    """Save the evolution progress as a GIF animation."""
    text_position = (10, 10)
    text_color = (255, 255, 255)
    font = ImageFont.truetype('arial.ttf', 16)
    frames = []

    for iteration, img_array in img_history:
        if original_img.shape[0] != img_array.shape[0]:
            img_array = resize_image(img_array, height=original_img.shape[0])

        img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img)
        draw.text(text_position, f'iteration: {iteration}', fill=text_color, font=font)
        frames.append(Image.fromarray(np.hstack((original_img, np.array(img)))))

    frames[0].save(output_path, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)
