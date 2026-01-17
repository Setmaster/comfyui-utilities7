"""
Image Composite Grid Node
Creates a grid of images with optional text labels below each image.
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def tensor2pil(tensor):
    """Convert a ComfyUI image tensor to PIL Image"""
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)


def pil2tensor(pil_img):
    """Convert PIL Image to ComfyUI tensor"""
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)


class ImageCompositeGrid:
    """
    ComfyUI node that creates a grid of images with optional text labels.
    Supports dynamic number of image inputs controlled by image_amount widget.
    """

    MAX_IMAGES = 16

    def __init__(self):
        self.font_size = 20
        self.padding = 10

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_amount": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": cls.MAX_IMAGES,
                    "step": 1,
                    "tooltip": "Number of image input slots to show"
                }),
                "columns": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Number of columns in the grid"
                }),
                "rows": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Number of rows in the grid"
                }),
                "labels": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "One label per line (empty line = no label)",
                    "tooltip": "Text labels for each image. One per line, matching image order."
                }),
                "text_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "Text color in hex format (e.g. #000000)"
                }),
                "bg_color": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": "Background color in hex format (e.g. #FFFFFF)"
                }),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "utilities7/image"

    def get_font(self):
        """Get a font for text rendering"""
        try:
            if os.name == "nt":
                return ImageFont.truetype("arial.ttf", self.font_size)
            elif os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
                return ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    self.font_size,
                )
            else:
                return ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()

    def create_text_panel(self, width, text, bg_color, text_color):
        """Create a text panel with centered text"""
        font = self.get_font()
        temp_img = Image.new("RGB", (width, self.font_size * 2), bg_color)
        temp_draw = ImageDraw.Draw(temp_img)
        text_bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        final_height = max(int(text_height * 1.5), self.font_size * 2)
        panel = Image.new("RGB", (width, final_height), bg_color)
        draw = ImageDraw.Draw(panel)
        x = (width - text_width) // 2
        y = (final_height - text_height) // 2
        draw.text((x, y), text, font=font, fill=text_color)
        return panel

    def generate(
        self,
        image_amount: int,
        columns: int,
        rows: int,
        labels: str,
        text_color: str = "#000000",
        bg_color: str = "#FFFFFF",
        **kwargs
    ):
        """Generate the composite grid image"""
        # Parse labels (one per line)
        label_list = labels.split("\n") if labels.strip() else []

        # Collect images from kwargs
        images = []
        for i in range(1, self.MAX_IMAGES + 1):
            img_key = f"image_{i}"
            if img_key in kwargs and kwargs[img_key] is not None:
                images.append(kwargs[img_key])

        if not images:
            print("[ImageCompositeGrid] Warning: No images provided")
            return (torch.zeros((1, 64, 64, 3)),)

        # Determine actual grid size based on provided images
        num_images = len(images)
        actual_slots = columns * rows

        # Convert tensors to PIL images (take first frame from each batch)
        pil_images = []
        for tensor in images:
            if tensor is not None and hasattr(tensor, "shape") and tensor.shape[0] > 0:
                pil_images.append(tensor2pil(tensor))

        if not pil_images:
            print("[ImageCompositeGrid] Warning: No valid images to process")
            return (torch.zeros((1, 64, 64, 3)),)

        # Find the largest dimensions to use as cell size
        max_width = max(img.width for img in pil_images)
        max_height = max(img.height for img in pil_images)

        # Calculate text panel height
        has_any_label = any(
            i < len(label_list) and label_list[i].strip()
            for i in range(len(pil_images))
        )
        text_panel_height = (self.font_size * 2 + self.padding) if has_any_label else 0

        # Calculate cell dimensions
        cell_width = max_width
        cell_height = max_height + text_panel_height

        # Calculate total canvas size
        total_width = columns * cell_width + (columns + 1) * self.padding
        total_height = rows * cell_height + (rows + 1) * self.padding

        # Create the result canvas
        result = Image.new("RGB", (total_width, total_height), bg_color)

        # Place images in grid
        for idx, pil_img in enumerate(pil_images):
            if idx >= actual_slots:
                break

            row = idx // columns
            col = idx % columns

            # Calculate position
            x = self.padding + col * (cell_width + self.padding)
            y = self.padding + row * (cell_height + self.padding)

            # Resize image to fit cell while maintaining aspect ratio
            img_ratio = pil_img.width / pil_img.height
            cell_ratio = cell_width / max_height

            if img_ratio > cell_ratio:
                # Image is wider - fit to width
                new_width = cell_width
                new_height = int(cell_width / img_ratio)
            else:
                # Image is taller - fit to height
                new_height = max_height
                new_width = int(max_height * img_ratio)

            resized = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Center the image in its cell
            img_x = x + (cell_width - new_width) // 2
            img_y = y + (max_height - new_height) // 2

            result.paste(resized, (img_x, img_y))

            # Add text label if exists
            if idx < len(label_list) and label_list[idx].strip():
                text_panel = self.create_text_panel(
                    cell_width,
                    label_list[idx].strip(),
                    bg_color,
                    text_color
                )
                text_y = y + max_height
                result.paste(text_panel, (x, text_y))

        return (pil2tensor(result),)


NODE_CLASS_MAPPINGS = {
    "ImageCompositeGrid": ImageCompositeGrid
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCompositeGrid": "Image Composite Grid"
}
