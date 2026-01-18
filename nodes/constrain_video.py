"""
Constrain Video Node
Constrains videos by maximum dimensions and/or file size.
Accepts native ComfyUI VIDEO type, file path string, or image sequence.
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path

import torch
import numpy as np
from PIL import Image

import folder_paths
from comfy.utils import ProgressBar

# Import utilities from compose_video
from .compose_video import (
    ffmpeg_path,
    tensor_to_bytes,
    ENCODE_ARGS,
)


def get_video_info(video_path: str) -> dict:
    """Get video duration, dimensions, and audio info using ffprobe."""
    if not os.path.exists(video_path):
        return None

    # Get video stream info
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    if not data.get("streams"):
        return None

    stream = data["streams"][0]

    # Get duration from format
    duration_cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", video_path
    ]
    duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
    try:
        duration_data = json.loads(duration_result.stdout)
    except json.JSONDecodeError:
        duration_data = {}

    # Check for audio stream
    audio_cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "a", video_path
    ]
    audio_result = subprocess.run(audio_cmd, capture_output=True, text=True)
    try:
        audio_data = json.loads(audio_result.stdout)
        has_audio = bool(audio_data.get("streams"))
    except json.JSONDecodeError:
        has_audio = False

    return {
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "duration": float(duration_data.get("format", {}).get("duration", 0)),
        "has_audio": has_audio,
    }


def constrain_video_dimensions(
    input_path: str,
    output_path: str,
    max_width: int,
    max_height: int,
    remove_audio: bool = False,
) -> bool:
    """
    Scale video to fit within max dimensions while preserving aspect ratio.
    Returns True if scaling was applied, False if already within limits.
    """
    info = get_video_info(input_path)
    if not info:
        return False

    width = info["width"]
    height = info["height"]

    # Check if scaling is needed
    if width <= max_width and height <= max_height:
        return False

    # Calculate scale to fit within bounds
    scale_w = max_width / width if max_width > 0 else 1
    scale_h = max_height / height if max_height > 0 else 1
    scale = min(scale_w, scale_h)

    new_width = int(width * scale)
    new_height = int(height * scale)
    # Ensure even dimensions for video codecs
    new_width = new_width - (new_width % 2)
    new_height = new_height - (new_height % 2)

    cmd = [
        ffmpeg_path, "-y", "-i", input_path,
        "-vf", f"scale={new_width}:{new_height}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
    ]

    if remove_audio or not info["has_audio"]:
        cmd.append("-an")
    else:
        cmd.extend(["-c:a", "copy"])

    cmd.append(output_path)

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg error during scaling: {e.stderr.decode(*ENCODE_ARGS)}")


def constrain_video_size(
    input_path: str,
    output_path: str,
    max_size_mb: float,
    remove_audio: bool = False,
) -> bool:
    """
    Constrain video file size using two-pass encoding.
    Returns True if re-encoding was applied, False if already under limit.
    """
    target_size_bytes = max_size_mb * 1024 * 1024
    current_size = os.path.getsize(input_path)

    if current_size <= target_size_bytes:
        return False

    info = get_video_info(input_path)
    if not info:
        return False

    duration = info["duration"]
    if duration <= 0:
        raise Exception("Could not determine video duration")

    # Calculate target bitrate
    target_size_bits = target_size_bytes * 8 * 0.95  # 5% margin

    include_audio = info["has_audio"] and not remove_audio
    if include_audio:
        audio_bitrate = 96  # kbps
        audio_bits = audio_bitrate * 1000 * duration
        video_bits = target_size_bits - audio_bits
    else:
        video_bits = target_size_bits

    video_bitrate = int(video_bits / duration / 1000)  # kbps

    if video_bitrate < 50:
        video_bitrate = 50
        print(f"Warning: Very low bitrate ({video_bitrate}k) - quality will be reduced")

    # Two-pass encoding
    passlog_prefix = os.path.join(tempfile.gettempdir(), "ffmpeg2pass")
    null_output = "/dev/null" if sys.platform != "win32" else "NUL"

    # Pass 1
    pass1_cmd = [
        ffmpeg_path, "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "medium",
        "-b:v", f"{video_bitrate}k",
        "-pass", "1", "-passlogfile", passlog_prefix,
        "-an", "-f", "null", null_output
    ]

    try:
        subprocess.run(pass1_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg pass 1 error: {e.stderr.decode(*ENCODE_ARGS)}")

    # Pass 2
    pass2_cmd = [
        ffmpeg_path, "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "medium",
        "-b:v", f"{video_bitrate}k",
        "-pass", "2", "-passlogfile", passlog_prefix,
    ]

    if include_audio:
        pass2_cmd.extend(["-c:a", "aac", "-b:a", f"{audio_bitrate}k"])
    else:
        pass2_cmd.append("-an")

    pass2_cmd.append(output_path)

    try:
        subprocess.run(pass2_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg pass 2 error: {e.stderr.decode(*ENCODE_ARGS)}")

    # Cleanup passlog files
    for f in Path(tempfile.gettempdir()).glob("ffmpeg2pass*.log*"):
        try:
            f.unlink()
        except:
            pass

    # Check if we hit the target
    final_size = os.path.getsize(output_path)
    if final_size > target_size_bytes:
        # Retry with lower bitrate
        reduction = (target_size_bytes / final_size) * 0.95
        new_bitrate = max(50, int(video_bitrate * reduction))

        pass1_cmd[pass1_cmd.index(f"{video_bitrate}k")] = f"{new_bitrate}k"
        pass2_cmd[pass2_cmd.index(f"{video_bitrate}k")] = f"{new_bitrate}k"

        subprocess.run(pass1_cmd, check=True, capture_output=True)
        subprocess.run(pass2_cmd, check=True, capture_output=True)

        for f in Path(tempfile.gettempdir()).glob("ffmpeg2pass*.log*"):
            try:
                f.unlink()
            except:
                pass

    return True


def remove_audio_from_video(input_path: str, output_path: str) -> bool:
    """Remove audio track from video. Returns True if audio was removed."""
    info = get_video_info(input_path)
    if not info or not info["has_audio"]:
        return False

    cmd = [
        ffmpeg_path, "-y", "-i", input_path,
        "-c:v", "copy", "-an", output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg error removing audio: {e.stderr.decode(*ENCODE_ARGS)}")


def trim_video(
    input_path: str,
    output_path: str,
    trim_start: float = 0.0,
    trim_end: float = 0.0,
) -> bool:
    """
    Trim video to a time range.
    trim_start: start timestamp in seconds (0 = from beginning)
    trim_end: end timestamp in seconds (0 = to the end)
    Returns True if trimming was applied, False if no trimming needed.
    """
    if trim_start <= 0 and trim_end <= 0:
        return False

    info = get_video_info(input_path)
    if not info:
        return False

    duration = info["duration"]

    # Calculate actual start and end points
    start_time = trim_start if trim_start > 0 else 0
    end_time = trim_end if trim_end > 0 else duration

    # Validate
    if start_time >= end_time:
        raise Exception(f"Invalid trim: start ({start_time}s) >= end ({end_time}s). Video duration: {duration}s")

    if start_time >= duration:
        raise Exception(f"trim_start ({trim_start}s) exceeds video duration ({duration}s)")

    if end_time > duration:
        end_time = duration  # Clamp to video duration

    new_duration = end_time - start_time

    cmd = [
        ffmpeg_path, "-y",
        "-ss", str(start_time),
        "-i", input_path,
        "-t", str(new_duration),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
    ]

    if info["has_audio"]:
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    else:
        cmd.append("-an")

    cmd.append(output_path)

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg error during trimming: {e.stderr.decode(*ENCODE_ARGS)}")


def encode_images_to_video(
    images: torch.Tensor,
    output_path: str,
    frame_rate: float,
    audio: dict = None,
) -> str:
    """Encode image tensor batch to a temporary video file."""
    import threading

    if ffmpeg_path is None:
        raise Exception("ffmpeg not found")

    num_frames = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]

    # Ensure even dimensions
    if width % 2 == 1:
        width -= 1
    if height % 2 == 1:
        height -= 1

    cmd = [
        ffmpeg_path, "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(frame_rate),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",
        output_path,
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Read stderr in a background thread to prevent deadlock
    stderr_output = []
    def read_stderr():
        stderr_output.append(proc.stderr.read())
    stderr_thread = threading.Thread(target=read_stderr)
    stderr_thread.start()

    pbar = ProgressBar(num_frames)
    try:
        for i in range(num_frames):
            frame = images[i]
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = frame[:height, :width, :]
            frame_bytes = tensor_to_bytes(frame).tobytes()
            proc.stdin.write(frame_bytes)
            pbar.update(1)
    except BrokenPipeError:
        pass  # Process died, we'll catch error below
    finally:
        proc.stdin.close()

    proc.wait()
    stderr_thread.join()

    if proc.returncode != 0:
        error = stderr_output[0].decode(*ENCODE_ARGS) if stderr_output else "Unknown error"
        raise Exception(f"FFmpeg encoding error: {error}")

    # Add audio if provided
    if audio is not None:
        temp_video = output_path + ".tmp.mp4"
        os.rename(output_path, temp_video)

        channels = audio['waveform'].size(1)
        audio_data = audio['waveform'].squeeze(0).transpose(0, 1).numpy().tobytes()

        mux_cmd = [
            ffmpeg_path, "-y",
            "-i", temp_video,
            "-ar", str(audio['sample_rate']),
            "-ac", str(channels),
            "-f", "f32le", "-i", "-",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_path
        ]

        try:
            subprocess.run(mux_cmd, input=audio_data, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            # If muxing fails, keep the video without audio
            os.rename(temp_video, output_path)
        else:
            os.unlink(temp_video)

    return output_path


class ConstrainVideo:
    """
    Constrains videos by maximum dimensions and/or file size.
    Accepts multiple input types for flexibility.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "video": ("VIDEO", {
                    "tooltip": "Native ComfyUI VIDEO input (from Load Video node)"
                }),
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to video file (from Compose Video or manual input)"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Image sequence to constrain (from VHS or other loaders)"
                }),
                "frame_rate": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 1.0,
                    "tooltip": "Frame rate for image sequence input (ignored for video/video_path)"
                }),
                "audio": ("AUDIO", {
                    "tooltip": "Audio for image sequence input (ignored for video/video_path)"
                }),
                "max_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Maximum width in pixels. 0 = no constraint."
                }),
                "max_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Maximum height in pixels. 0 = no constraint."
                }),
                "max_size_mb": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 0.1,
                    "tooltip": "Maximum file size in MB. 0 = no constraint. Uses two-pass encoding."
                }),
                "trim_start": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "Start timestamp in seconds. 0 = start from beginning."
                }),
                "trim_end": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "End timestamp in seconds. 0 = go to the end."
                }),
                "remove_audio": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Remove audio track from the output video."
                }),
                "filename_prefix": ("STRING", {
                    "default": "video/constrained",
                    "tooltip": "Prefix for output filename."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_TOOLTIPS = ("Path to the constrained video file.",)
    OUTPUT_NODE = True
    FUNCTION = "constrain"
    CATEGORY = "utilities7/video"

    def constrain(
        self,
        video=None,
        video_path: str = "",
        images=None,
        frame_rate: float = 30.0,
        audio=None,
        max_width: int = 0,
        max_height: int = 0,
        max_size_mb: float = 0.0,
        trim_start: float = 0.0,
        trim_end: float = 0.0,
        remove_audio: bool = False,
        filename_prefix: str = "video/constrained",
    ):
        if ffmpeg_path is None:
            raise Exception(
                "ffmpeg is required for Constrain Video and could not be found.\n"
                "Install imageio-ffmpeg with pip or add ffmpeg to system PATH."
            )

        # Determine input source (priority: video > video_path > images)
        source_path = None
        temp_source = False

        if video is not None:
            # Native ComfyUI VIDEO type
            try:
                stream_source = video.get_stream_source()
                if isinstance(stream_source, str):
                    source_path = stream_source
                else:
                    # BytesIO - save to temp file
                    temp_file = tempfile.NamedTemporaryFile(
                        suffix=".mp4", delete=False
                    )
                    temp_file.write(stream_source.read())
                    temp_file.close()
                    source_path = temp_file.name
                    temp_source = True
            except Exception as e:
                raise Exception(f"Could not read VIDEO input: {e}")

        elif video_path and video_path.strip():
            source_path = video_path.strip()
            if not os.path.exists(source_path):
                raise Exception(f"Video file not found: {source_path}")

        elif images is not None:
            # Encode images to temp video
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".mp4", delete=False
            )
            temp_file.close()
            source_path = encode_images_to_video(
                images, temp_file.name, frame_rate, audio
            )
            temp_source = True

        else:
            raise Exception(
                "No input provided. Connect either:\n"
                "- video (from Load Video)\n"
                "- video_path (from Compose Video)\n"
                "- images (from VHS or other loaders)"
            )

        # Check if any constraints are set
        has_dimension_constraint = max_width > 0 or max_height > 0
        has_size_constraint = max_size_mb > 0
        has_trim = trim_start > 0 or trim_end > 0

        if not has_dimension_constraint and not has_size_constraint and not has_trim and not remove_audio:
            # No constraints - just return original path
            if temp_source:
                # Move temp file to output
                output_dir = folder_paths.get_output_directory()
                full_output_folder, filename, _, subfolder, _ = \
                    folder_paths.get_save_image_path(filename_prefix, output_dir)

                output_file = os.path.join(full_output_folder, f"{filename}_00001.mp4")
                shutil.copy2(source_path, output_file)
                os.unlink(source_path)
                return (output_file,)
            return (source_path,)

        # Set up output path
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = \
            folder_paths.get_save_image_path(filename_prefix, output_dir)

        # Find next counter
        import re
        max_counter = 0
        matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\..+", re.IGNORECASE)
        for existing_file in os.listdir(full_output_folder):
            match = matcher.fullmatch(existing_file)
            if match:
                file_counter = int(match.group(1))
                if file_counter > max_counter:
                    max_counter = file_counter
        counter = max_counter + 1

        output_path = os.path.join(full_output_folder, f"{filename}_{counter:05}.mp4")

        # Track current working file
        current_path = source_path
        needs_cleanup = []

        try:
            # Apply trimming first (reduces data for subsequent operations)
            if has_trim:
                temp_trimmed = tempfile.NamedTemporaryFile(
                    suffix=".mp4", delete=False
                ).name

                if trim_video(current_path, temp_trimmed, trim_start, trim_end):
                    if current_path != source_path or temp_source:
                        needs_cleanup.append(current_path)
                    current_path = temp_trimmed
                else:
                    os.unlink(temp_trimmed)

            # Apply dimension constraints (simpler, may reduce file size)
            if has_dimension_constraint:
                temp_scaled = tempfile.NamedTemporaryFile(
                    suffix=".mp4", delete=False
                ).name

                if constrain_video_dimensions(
                    current_path, temp_scaled,
                    max_width if max_width > 0 else 99999,
                    max_height if max_height > 0 else 99999,
                    remove_audio
                ):
                    if current_path != source_path or temp_source:
                        needs_cleanup.append(current_path)
                    current_path = temp_scaled
                else:
                    os.unlink(temp_scaled)

            # Apply size constraint (two-pass encoding)
            if has_size_constraint:
                temp_sized = tempfile.NamedTemporaryFile(
                    suffix=".mp4", delete=False
                ).name

                if constrain_video_size(
                    current_path, temp_sized,
                    max_size_mb, remove_audio
                ):
                    if current_path != source_path or temp_source:
                        needs_cleanup.append(current_path)
                    current_path = temp_sized
                else:
                    os.unlink(temp_sized)

            # Remove audio if requested and not already done
            if remove_audio and not has_dimension_constraint and not has_size_constraint:
                temp_noaudio = tempfile.NamedTemporaryFile(
                    suffix=".mp4", delete=False
                ).name

                if remove_audio_from_video(current_path, temp_noaudio):
                    if current_path != source_path or temp_source:
                        needs_cleanup.append(current_path)
                    current_path = temp_noaudio
                else:
                    os.unlink(temp_noaudio)

            # Copy/move to final output
            if current_path != source_path or temp_source:
                shutil.copy2(current_path, output_path)
                needs_cleanup.append(current_path)
            else:
                # No changes were made, copy original
                shutil.copy2(source_path, output_path)

        finally:
            # Cleanup temp files
            if temp_source and source_path not in needs_cleanup:
                needs_cleanup.append(source_path)

            for f in needs_cleanup:
                try:
                    if os.path.exists(f):
                        os.unlink(f)
                except:
                    pass

        return (output_path,)


# Node Registration
NODE_CLASS_MAPPINGS = {
    "ConstrainVideo": ConstrainVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConstrainVideo": "Constrain Video"
}
