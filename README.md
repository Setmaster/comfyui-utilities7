# ComfyUI Utilities

A collection of utility nodes for ComfyUI.

## Installation

Clone or download this repository into your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Setmaster/comfyui-utilities7.git
```

Restart ComfyUI to load the nodes.

## Nodes

### Image Composite Grid

Creates a grid layout of multiple images with optional text labels below each image.

**Inputs:**
- `image_amount` (1-16): Number of image input slots
- `columns`: Number of columns in the grid
- `rows`: Number of rows in the grid
- `labels`: Multiline text field - one label per line (empty line = no label for that image)
- `text_color`: Hex color for label text (e.g., #000000)
- `bg_color`: Hex color for background (e.g., #FFFFFF)
- `image_1` to `image_N`: Dynamic image inputs based on `image_amount`

**Output:**
- `image`: The composited grid image

### Image Composite Grid Labels

Helper node that combines 16 individual text fields into a multiline string. Useful for programmatically generating labels from other nodes.

**Inputs:**
- `text_1` to `text_16`: Individual text fields (can be typed or connected)

**Output:**
- `labels`: Multiline string ready to connect to Image Composite Grid's labels input

## Category

All nodes appear under **utilities7/image** in the node menu.

## License

MIT
