"""
Image Composite Grid Labels Node
Helper node that combines 16 text fields into a multiline string for use with Image Composite Grid.
"""


class ImageCompositeGridLabels:
    """
    ComfyUI node that combines multiple text fields into a multiline string.
    Each text field becomes one line in the output, matching Image Composite Grid's label format.
    """

    MAX_LABELS = 16

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {},
            "optional": {}
        }
        # Add all 16 text fields as required with empty defaults
        for i in range(1, cls.MAX_LABELS + 1):
            inputs["required"][f"text_{i}"] = ("STRING", {"default": ""})
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("labels",)
    FUNCTION = "combine"
    CATEGORY = "utilities7/image"

    def combine(self, **kwargs):
        """Combine all text fields into a multiline string"""
        lines = []
        for i in range(1, self.MAX_LABELS + 1):
            text = kwargs.get(f"text_{i}", "") or ""
            lines.append(text)

        # Join all lines, preserving empty lines
        result = "\n".join(lines)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "ImageCompositeGridLabels": ImageCompositeGridLabels
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCompositeGridLabels": "Image Composite Grid Labels"
}
