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
- `font_size` (8-72, default 20): Font size for text labels
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

### Compose Video

Combines an image sequence into a video file with optional audio. Based on ComfyUI-VideoHelperSuite's Video Combine node.

**Inputs:**
- `images`: Image sequence to combine into video
- `frame_rate`: Frames per second (1-120, default 8)
- `loop_count`: Number of additional loops (0 = play once)
- `filename_prefix`: Output filename prefix (default: "video/compose_video", saves to video subfolder)
- `format`: Output format (image/gif, image/webp, video/h264-mp4, video/h265-mp4, video/webm, etc.)
- `pingpong`: Play forward then reverse for seamless loop effect
- `save_output`: Save to output folder (true) or temp folder (false)
- `audio` (optional): Audio to mix into the video

**Format-specific options** (appear based on selected format):
- `audio_output`: Controls audio output - "with audio" (default), "without audio", or "both". Only for formats supporting audio. If no audio connected, outputs video without audio regardless.
- `save_metadata`: Save a PNG with embedded workflow metadata alongside the output (default: true). Useful for formats that can't reliably embed metadata.
- `crf`, `pix_fmt`, `trim_to_audio`, etc.: Codec-specific options vary by format.

**Features:**
- Video preview displayed on node after execution
- Right-click context menu: Open, Save, Copy path, Pause/Resume, Hide/Show, Mute/Unmute
- Audio unmutes on hover
- Supports multiple video formats via ffmpeg

**Requirements:**
- ffmpeg (via imageio-ffmpeg, system PATH, or local binary)
- gifski (optional, for high-quality GIF output)

### Constrain Video

Constrains videos by maximum dimensions and/or file size. Useful for meeting platform upload limits (Discord, Twitter, etc.).

**Inputs (all optional - provide one source):**
- `video`: Native ComfyUI VIDEO input (from Load Video node)
- `video_path`: Path to video file (from Compose Video output or manual input)
- `images`: Image sequence (from VHS Load Video or other loaders)
- `frame_rate`: Frame rate for image sequence input (default: 30)
- `audio`: Audio for image sequence input

**Constraint options:**
- `max_width`: Maximum width in pixels (0 = no constraint)
- `max_height`: Maximum height in pixels (0 = no constraint)
- `max_size_mb`: Maximum file size in MB (0 = no constraint). Uses two-pass encoding.
- `trim_start`: Seconds to trim from the start (0 = no trim)
- `trim_end`: Seconds to trim from the end (0 = no trim)
- `remove_audio`: Remove audio track from output (default: false)
- `filename_prefix`: Output filename prefix (default: "video/constrained")

**Output:**
- `video_path`: Path to the constrained video file

**Features:**
- Accepts multiple input types for flexibility
- Dimension constraint preserves aspect ratio
- Two-pass encoding for accurate file size targeting
- Can be chained after Compose Video via the video_path output

## Categories

- **utilities7/image**: Image Composite Grid, Image Composite Grid Labels
- **utilities7/video**: Compose Video, Constrain Video

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
