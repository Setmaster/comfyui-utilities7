"""
Compose Video Node
Creates video files from image sequences with optional audio.
Based on ComfyUI-VideoHelperSuite's Video Combine node (GPL-3.0).
Simplified version without meta_batch, vae pin, and filenames output.
"""

import os
import sys
import json
import subprocess
import numpy as np
import re
import datetime
import shutil
import functools
import time
from typing import List
import torch
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo
from pathlib import Path
from string import Template
import itertools

import folder_paths
from comfy.utils import ProgressBar

# ============================================================================
# Utility constants and functions (from VHS utils.py)
# ============================================================================

BIGMIN = -(2**53-1)
BIGMAX = (2**53-1)
ENCODE_ARGS = ("utf-8", 'backslashreplace')


def ffmpeg_suitability(path):
    """Score ffmpeg installations by feature support."""
    try:
        version = subprocess.run([path, "-version"], check=True,
                                 capture_output=True).stdout.decode(*ENCODE_ARGS)
    except:
        return 0
    score = 0
    simple_criterion = [("libvpx", 20), ("264", 10), ("265", 3),
                        ("svtav1", 5), ("libopus", 1)]
    for criterion in simple_criterion:
        if version.find(criterion[0]) >= 0:
            score += criterion[1]
    copyright_index = version.find('2000-2')
    if copyright_index >= 0:
        copyright_year = version[copyright_index+6:copyright_index+9]
        if copyright_year.isnumeric():
            score += int(copyright_year)
    return score


def _get_ffmpeg_path():
    """Find the best available ffmpeg installation."""
    if "VHS_FORCE_FFMPEG_PATH" in os.environ:
        return os.environ.get("VHS_FORCE_FFMPEG_PATH")
    
    ffmpeg_paths = []
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        imageio_ffmpeg_path = get_ffmpeg_exe()
        ffmpeg_paths.append(imageio_ffmpeg_path)
    except:
        pass
    
    if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
        return imageio_ffmpeg_path if ffmpeg_paths else None
    
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg is not None:
        ffmpeg_paths.append(system_ffmpeg)
    if os.path.isfile("ffmpeg"):
        ffmpeg_paths.append(os.path.abspath("ffmpeg"))
    if os.path.isfile("ffmpeg.exe"):
        ffmpeg_paths.append(os.path.abspath("ffmpeg.exe"))
    
    if len(ffmpeg_paths) == 0:
        return None
    elif len(ffmpeg_paths) == 1:
        return ffmpeg_paths[0]
    else:
        return max(ffmpeg_paths, key=ffmpeg_suitability)


def _get_gifski_path():
    """Find gifski installation if available."""
    gifski_path = os.environ.get("VHS_GIFSKI", None)
    if gifski_path is None:
        gifski_path = os.environ.get("JOV_GIFSKI", None)
        if gifski_path is None:
            gifski_path = shutil.which("gifski")
    return gifski_path


ffmpeg_path = _get_ffmpeg_path()
gifski_path = _get_gifski_path()


def cached(duration):
    """Decorator to cache function results for a duration in seconds."""
    def dec(f):
        cached_ret = None
        cache_time = 0
        def cached_func():
            nonlocal cache_time, cached_ret
            if time.time() > cache_time + duration or cached_ret is None:
                cache_time = time.time()
                cached_ret = f()
            return cached_ret
        return cached_func
    return dec


def merge_filter_args(args, ftype="-vf"):
    """Merge multiple filter arguments into one."""
    try:
        start_index = args.index(ftype) + 1
        index = start_index
        while True:
            index = args.index(ftype, index)
            args[start_index] += ',' + args[index+1]
            args.pop(index)
            args.pop(index)
    except ValueError:
        pass


def get_audio(file, start_time=0, duration=0):
    """Extract audio from a file using ffmpeg."""
    args = [ffmpeg_path, "-i", file]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    if duration > 0:
        args += ["-t", str(duration)]
    try:
        res = subprocess.run(args + ["-f", "f32le", "-"],
                             capture_output=True, check=True)
        audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
        match = re.search(', (\\d+) Hz, (\\w+), ', res.stderr.decode(*ENCODE_ARGS))
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to extract audio from {file}:\n" + e.stderr.decode(*ENCODE_ARGS))
    if match:
        ar = int(match.group(1))
        ac = {"mono": 1, "stereo": 2}[match.group(2)]
    else:
        ar = 44100
        ac = 2
    audio = audio.reshape((-1, ac)).transpose(0, 1).unsqueeze(0)
    return {'waveform': audio, 'sample_rate': ar}


# ============================================================================
# Video format handling
# ============================================================================

# Register video formats folder path
if 'utilities7_video_formats' not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["utilities7_video_formats"] = ((), {".json"})
if len(folder_paths.folder_names_and_paths['utilities7_video_formats'][1]) == 0:
    folder_paths.folder_names_and_paths["utilities7_video_formats"][1].add(".json")

base_formats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "video_formats")


def flatten_list(l):
    """Flatten nested lists."""
    ret = []
    for e in l:
        if isinstance(e, list):
            ret.extend(e)
        else:
            ret.append(e)
    return ret


def iterate_format(video_format, for_widgets=True):
    """Provides an iterator over widgets or arguments in a video format."""
    def indirector(cont, index):
        if isinstance(cont[index], list) and (not for_widgets
          or len(cont[index]) > 1 and not isinstance(cont[index][1], dict)):
            inp = yield cont[index]
            if inp is not None:
                cont[index] = inp
                yield
    for k in video_format:
        if k == "extra_widgets":
            if for_widgets:
                yield from video_format["extra_widgets"]
        elif k.endswith("_pass"):
            for i in range(len(video_format[k])):
                yield from indirector(video_format[k], i)
            if not for_widgets:
                video_format[k] = flatten_list(video_format[k])
        else:
            yield from indirector(video_format, k)


@cached(5)
def get_video_formats():
    """Get available video formats from JSON files."""
    format_files = {}

    # Check custom formats folder
    for format_name in folder_paths.get_filename_list("utilities7_video_formats"):
        format_files[format_name] = folder_paths.get_full_path("utilities7_video_formats", format_name)

    # Check built-in formats
    if os.path.isdir(base_formats_dir):
        for item in os.scandir(base_formats_dir):
            if not item.is_file() or not item.name.endswith('.json'):
                continue
            format_files[item.name[:-5]] = item.path

    formats = []
    format_widgets = {}

    # Audio output widget for formats that support audio
    audio_output_widget = ['audio_output', ["with audio", "without audio", "both"], {
        'default': "with audio",
        'tooltip': "Output options when audio is connected: 'with audio' outputs only the video with audio, 'without audio' outputs only the silent video, 'both' outputs both versions."
    }]

    for format_name, path in format_files.items():
        with open(path, 'r') as stream:
            video_format = json.load(stream)
        if "gifski_pass" in video_format and gifski_path is None:
            continue
        widgets = list(iterate_format(video_format))
        formats.append("video/" + format_name)

        # Add audio_output widget for formats that support audio (not gifski)
        if "gifski_pass" not in video_format:
            widgets = [audio_output_widget] + widgets

        if len(widgets) > 0:
            format_widgets["video/" + format_name] = widgets
    return formats, format_widgets


def apply_format_widgets(format_name, kwargs):
    """Apply format-specific widget values to the video format config."""
    if os.path.exists(os.path.join(base_formats_dir, format_name + ".json")):
        video_format_path = os.path.join(base_formats_dir, format_name + ".json")
    else:
        video_format_path = folder_paths.get_full_path("utilities7_video_formats", format_name)
    
    with open(video_format_path, 'r') as stream:
        video_format = json.load(stream)
    
    for w in iterate_format(video_format):
        if w[0] not in kwargs:
            if len(w) > 2 and 'default' in w[2]:
                default = w[2]['default']
            else:
                if type(w[1]) is list:
                    default = w[1][0]
                else:
                    default = {"BOOLEAN": False, "INT": 0, "FLOAT": 0, "STRING": ""}[w[1]]
            kwargs[w[0]] = default
    
    wit = iterate_format(video_format, False)
    for w in wit:
        while isinstance(w, list):
            if len(w) == 1:
                w = [Template(x).substitute(**kwargs) for x in w[0]]
                break
            elif isinstance(w[1], dict):
                w = w[1][str(kwargs[w[0]])]
            elif len(w) > 3:
                w = Template(w[3]).substitute(val=kwargs[w[0]])
            else:
                w = str(kwargs[w[0]])
        wit.send(w)
    return video_format


# ============================================================================
# Image/tensor conversion utilities
# ============================================================================

def tensor_to_int(tensor, bits):
    """Convert tensor values to integer representation."""
    tensor = tensor.cpu().numpy() * (2**bits - 1) + 0.5
    return np.clip(tensor, 0, (2**bits - 1))


def tensor_to_shorts(tensor):
    """Convert tensor to 16-bit integers."""
    return tensor_to_int(tensor, 16).astype(np.uint16)


def tensor_to_bytes(tensor):
    """Convert tensor to 8-bit integers."""
    return tensor_to_int(tensor, 8).astype(np.uint8)


# ============================================================================
# FFmpeg/Gifski process generators
# ============================================================================

def ffmpeg_process(args, video_format, video_metadata, file_path, env):
    """Generator that pipes frames to ffmpeg."""
    res = None
    frame_data = yield
    total_frames_output = 0
    
    if video_format.get('save_metadata', 'False') != 'False':
        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
        metadata_path = os.path.join(folder_paths.get_temp_directory(), "metadata.txt")
        
        def escape_ffmpeg_metadata(key, value):
            value = str(value)
            value = value.replace("\\", "\\\\")
            value = value.replace(";", "\\;")
            value = value.replace("#", "\\#")
            value = value.replace("=", "\\=")
            value = value.replace("\n", "\\\n")
            return f"{key}={value}"
        
        with open(metadata_path, "w") as f:
            f.write(";FFMETADATA1\n")
            if "prompt" in video_metadata:
                f.write(escape_ffmpeg_metadata("prompt", json.dumps(video_metadata["prompt"])) + "\n")
            if "workflow" in video_metadata:
                f.write(escape_ffmpeg_metadata("workflow", json.dumps(video_metadata["workflow"])) + "\n")
            for k, v in video_metadata.items():
                if k not in ["prompt", "workflow"]:
                    f.write(escape_ffmpeg_metadata(k, json.dumps(v)) + "\n")
        
        m_args = args[:1] + ["-i", metadata_path] + args[1:] + ["-metadata", "creation_time=now", "-movflags", "use_metadata_tags"]
        with subprocess.Popen(m_args + [file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    frame_data = yield
                    total_frames_output += 1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError as e:
                err = proc.stderr.read()
                if os.path.exists(file_path):
                    raise Exception("An error occurred in the ffmpeg subprocess:\n" + err.decode(*ENCODE_ARGS))
                print(err.decode(*ENCODE_ARGS), end="", file=sys.stderr)
    
    if res != b'':
        with subprocess.Popen(args + [file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    frame_data = yield
                    total_frames_output += 1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError as e:
                res = proc.stderr.read()
                raise Exception("An error occurred in the ffmpeg subprocess:\n" + res.decode(*ENCODE_ARGS))
    
    yield total_frames_output
    if len(res) > 0:
        print(res.decode(*ENCODE_ARGS), end="", file=sys.stderr)


def gifski_process(args, dimensions, frame_rate, video_format, file_path, env):
    """Generator that pipes frames through ffmpeg to gifski."""
    frame_data = yield
    with subprocess.Popen(args + video_format['main_pass'] + ['-f', 'yuv4mpegpipe', '-'],
                          stderr=subprocess.PIPE, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, env=env) as procff:
        with subprocess.Popen([gifski_path] + video_format['gifski_pass']
                              + ['-W', f'{dimensions[0]}', '-H', f'{dimensions[1]}']
                              + ['-r', f'{frame_rate}']
                              + ['-q', '-o', file_path, '-'], stderr=subprocess.PIPE,
                              stdin=procff.stdout, stdout=subprocess.PIPE,
                              env=env) as procgs:
            try:
                while frame_data is not None:
                    procff.stdin.write(frame_data)
                    frame_data = yield
                procff.stdin.flush()
                procff.stdin.close()
                resff = procff.stderr.read()
                resgs = procgs.stderr.read()
                outgs = procgs.stdout.read()
            except BrokenPipeError as e:
                procff.stdin.close()
                resff = procff.stderr.read()
                resgs = procgs.stderr.read()
                raise Exception("An error occurred while creating gifski output\n"
                        + "Make sure you are using gifski --version >=1.32.0\nffmpeg: "
                        + resff.decode(*ENCODE_ARGS) + '\ngifski: ' + resgs.decode(*ENCODE_ARGS))
    if len(resff) > 0:
        print(resff.decode(*ENCODE_ARGS), end="", file=sys.stderr)
    if len(resgs) > 0:
        print(resgs.decode(*ENCODE_ARGS), end="", file=sys.stderr)
    if len(outgs) > 0:
        print(outgs.decode(*ENCODE_ARGS))


def to_pingpong(inp):
    """Create a pingpong (forward then reverse) sequence."""
    if not hasattr(inp, "__getitem__"):
        inp = list(inp)
    yield from inp
    for i in range(len(inp) - 2, 0, -1):
        yield inp[i]


# ============================================================================
# Compose Video Node
# ============================================================================

class ComposeVideo:
    """
    ComfyUI node that combines an image sequence into a video file.
    Supports various output formats, audio mixing, and loop/pingpong options.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        ffmpeg_formats, format_widgets = get_video_formats()
        format_widgets["image/webp"] = [['lossless', "BOOLEAN", {'default': True, 'tooltip': "Lossless compression. True = perfect quality but larger files, False = lossy compression."}]]
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "The image sequence to combine into a video"
                }),
                "frame_rate": ("FLOAT", {
                    "default": 8,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "tooltip": "Frames per second for the output video. Connect to Video Info's loaded_fps output when using audio to maintain sync."
                }),
                "loop_count": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of additional times the video should loop. 0 means play once. High values with long sequences may cause performance issues."
                }),
                "filename_prefix": ("STRING", {
                    "default": "video/compose_video",
                    "tooltip": "Prefix for output filename. Can include subfolders (e.g., 'video/myprefix')."
                }),
                "format": (["image/gif", "image/webp"] + ffmpeg_formats, {
                    'formats': format_widgets,
                    "tooltip": "Output format. 'image/' formats use PIL, 'video/' formats use ffmpeg."
                }),
                "pingpong": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Play video forward then reverse to create a seamless loop effect."
                }),
                "save_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If True, save to output folder. If False, save to temp folder."
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Optional audio to mix into the video. Ensure frame_rate matches the source for proper sync."
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "utilities7/video"
    FUNCTION = "compose_video"

    def compose_video(
        self,
        images,
        frame_rate: float,
        loop_count: int,
        filename_prefix: str = "video/compose_video",
        format: str = "image/gif",
        pingpong: bool = False,
        save_output: bool = True,
        prompt=None,
        extra_pnginfo=None,
        audio=None,
        **kwargs
    ):
        # Get audio_output from kwargs (dynamic widget) with default
        audio_output = kwargs.pop('audio_output', 'with audio')
        if images is None:
            return {"ui": {"gifs": []}, "result": ()}
        
        if isinstance(images, torch.Tensor) and images.size(0) == 0:
            return {"ui": {"gifs": []}, "result": ()}
        
        num_frames = len(images)
        pbar = ProgressBar(num_frames)
        
        first_image = images[0]
        images = iter(images)
        
        # Get output directory
        output_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)
        
        output_files = []
        
        # Prepare metadata
        metadata = PngInfo()
        video_metadata = {}
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
            video_metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                video_metadata[x] = extra_pnginfo[x]
            extra_options = extra_pnginfo.get('workflow', {}).get('extra', {})
        else:
            extra_options = {}
        metadata.add_text("CreationTime", datetime.datetime.now().isoformat(" ")[:19])
        
        # Find next available counter
        max_counter = 0
        matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
        for existing_file in os.listdir(full_output_folder):
            match = matcher.fullmatch(existing_file)
            if match:
                file_counter = int(match.group(1))
                if file_counter > max_counter:
                    max_counter = file_counter
        counter = max_counter + 1

        format_type, format_ext = format.split("/")
        first_image_file = None

        def save_metadata_png():
            """Save first frame as PNG to preserve workflow metadata."""
            nonlocal first_image_file
            if extra_options.get('VHS_MetadataImage', True) == False:
                return
            first_image_file = f"{filename}_{counter:05}.png"
            png_path = os.path.join(full_output_folder, first_image_file)
            Image.fromarray(tensor_to_bytes(first_image)).save(
                png_path,
                pnginfo=metadata,
                compress_level=4,
            )
            output_files.append(png_path)

        if format_type == "image":
            # Save metadata PNG for image formats
            save_metadata_png()
            # Use PIL for image formats (gif, webp)
            image_kwargs = {}
            if format_ext == "gif":
                image_kwargs['disposal'] = 2
            if format_ext == "webp":
                exif = Image.Exif()
                exif[ExifTags.IFD.Exif] = {36867: datetime.datetime.now().isoformat(" ")[:19]}
                image_kwargs['exif'] = exif
                image_kwargs['lossless'] = kwargs.get("lossless", True)
            
            file = f"{filename}_{counter:05}.{format_ext}"
            file_path = os.path.join(full_output_folder, file)
            
            if pingpong:
                images = to_pingpong(images)
            
            def frames_gen(images):
                for i in images:
                    pbar.update(1)
                    yield Image.fromarray(tensor_to_bytes(i))
            
            frames = frames_gen(images)
            next(frames).save(
                file_path,
                format=format_ext.upper(),
                save_all=True,
                append_images=frames,
                duration=round(1000 / frame_rate),
                loop=loop_count,
                compress_level=4,
                **image_kwargs
            )
            output_files.append(file_path)
        else:
            # Use ffmpeg for video formats
            if ffmpeg_path is None:
                raise ProcessLookupError(
                    f"ffmpeg is required for video outputs and could not be found.\n"
                    f"Install imageio-ffmpeg with pip, place ffmpeg in {os.path.abspath('')}, "
                    f"or install ffmpeg and add it to the system path."
                )
            
            has_alpha = first_image.shape[-1] == 4
            kwargs["has_alpha"] = has_alpha
            video_format = apply_format_widgets(format_ext, kwargs)

            # Save metadata PNG for video formats when save_metadata is enabled
            if video_format.get('save_metadata', 'False') != 'False':
                save_metadata_png()

            dim_alignment = video_format.get("dim_alignment", 2)
            
            # Handle dimension alignment padding
            if (first_image.shape[1] % dim_alignment) or (first_image.shape[0] % dim_alignment):
                to_pad = (-first_image.shape[1] % dim_alignment,
                          -first_image.shape[0] % dim_alignment)
                padding = (to_pad[0]//2, to_pad[0] - to_pad[0]//2,
                           to_pad[1]//2, to_pad[1] - to_pad[1]//2)
                padfunc = torch.nn.ReplicationPad2d(padding)
                
                def pad(image):
                    image = image.permute((2, 0, 1))
                    padded = padfunc(image.to(dtype=torch.float32))
                    return padded.permute((1, 2, 0))
                
                images = map(pad, images)
                dimensions = (-first_image.shape[1] % dim_alignment + first_image.shape[1],
                              -first_image.shape[0] % dim_alignment + first_image.shape[0])
            else:
                dimensions = (first_image.shape[1], first_image.shape[0])
            
            if pingpong:
                images = to_pingpong(images)
                if num_frames > 2:
                    num_frames += num_frames - 2
                    pbar.total = num_frames
            
            if loop_count > 0:
                loop_args = ["-vf", "loop=loop=" + str(loop_count) + ":size=" + str(num_frames)]
            else:
                loop_args = []
            
            # Set pixel format based on color depth
            if video_format.get('input_color_depth', '8bit') == '16bit':
                images = map(tensor_to_shorts, images)
                i_pix_fmt = 'rgba64' if has_alpha else 'rgb48'
            else:
                images = map(tensor_to_bytes, images)
                i_pix_fmt = 'rgba' if has_alpha else 'rgb24'
            
            file = f"{filename}_{counter:05}.{video_format['extension']}"
            file_path = os.path.join(full_output_folder, file)
            
            bitrate_arg = []
            bitrate = video_format.get('bitrate')
            if bitrate is not None:
                bitrate_arg = ["-b:v", str(bitrate) + "M" if video_format.get('megabit') == 'True' else str(bitrate) + "K"]
            
            args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", i_pix_fmt,
                    "-color_range", "pc", "-colorspace", "rgb", "-color_primaries", "bt709",
                    "-color_trc", video_format.get("fake_trc", "iec61966-2-1"),
                    "-s", f"{dimensions[0]}x{dimensions[1]}", "-r", str(frame_rate), "-i", "-"] + loop_args
            
            images = map(lambda x: x.tobytes(), images)
            env = os.environ.copy()
            if "environment" in video_format:
                env.update(video_format["environment"])
            
            # Handle pre_pass if needed
            if "pre_pass" in video_format:
                images = [b''.join(images)]
                os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
                in_args_len = args.index("-i") + 2
                pre_pass_args = args[:in_args_len] + video_format['pre_pass']
                merge_filter_args(pre_pass_args)
                try:
                    subprocess.run(pre_pass_args, input=images[0], env=env,
                                   capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occurred in the ffmpeg prepass:\n" + e.stderr.decode(*ENCODE_ARGS))
            
            if "inputs_main_pass" in video_format:
                in_args_len = args.index("-i") + 2
                args = args[:in_args_len] + video_format['inputs_main_pass'] + args[in_args_len:]
            
            # Create output process
            if 'gifski_pass' in video_format:
                format = 'image/gif'
                output_process = gifski_process(args, dimensions, frame_rate, video_format, file_path, env)
                audio = None
            else:
                args += video_format['main_pass'] + bitrate_arg
                merge_filter_args(args)
                output_process = ffmpeg_process(args, video_format, video_metadata, file_path, env)
            
            output_process.send(None)
            
            for image in images:
                pbar.update(1)
                output_process.send(image)
            
            try:
                total_frames_output = output_process.send(None)
                output_process.send(None)
            except StopIteration:
                pass
            
            output_files.append(file_path)

            # Handle audio based on audio_output setting
            # If no audio connected, always just output video without audio
            a_waveform = None
            if audio is not None:
                try:
                    a_waveform = audio['waveform']
                except:
                    pass

            # Only process audio if we have audio data AND user wants audio output
            should_create_audio_version = (
                a_waveform is not None and
                audio_output in ["with audio", "both"]
            )

            if should_create_audio_version:
                output_file_with_audio = f"{filename}_{counter:05}-audio.{video_format['extension']}"
                output_file_with_audio_path = os.path.join(full_output_folder, output_file_with_audio)

                if "audio_pass" not in video_format:
                    video_format["audio_pass"] = ["-c:a", "libopus"]

                channels = audio['waveform'].size(1)
                min_audio_dur = total_frames_output / frame_rate + 1

                if video_format.get('trim_to_audio', 'False') != 'False':
                    apad = []
                else:
                    apad = ["-af", "apad=whole_dur=" + str(min_audio_dur)]

                mux_args = [ffmpeg_path, "-v", "error", "-n", "-i", file_path,
                            "-ar", str(audio['sample_rate']), "-ac", str(channels),
                            "-f", "f32le", "-i", "-", "-c:v", "copy"] \
                            + video_format["audio_pass"] \
                            + apad + ["-shortest", output_file_with_audio_path]

                audio_data = audio['waveform'].squeeze(0).transpose(0, 1).numpy().tobytes()
                merge_filter_args(mux_args, '-af')

                try:
                    res = subprocess.run(mux_args, input=audio_data,
                                         env=env, capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occurred in the ffmpeg subprocess:\n" + e.stderr.decode(*ENCODE_ARGS))

                if res.stderr:
                    print(res.stderr.decode(*ENCODE_ARGS), end="", file=sys.stderr)

                output_files.append(output_file_with_audio_path)

                # If user only wants "with audio", delete the non-audio version
                if audio_output == "with audio":
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    output_files.remove(file_path)

                # Preview the audio version
                file = output_file_with_audio
        
        # Clean up intermediate files if requested
        if extra_options.get('VHS_KeepIntermediate', True) == False:
            for intermediate in output_files[1:-1]:
                if os.path.exists(intermediate):
                    os.remove(intermediate)
        
        preview = {
            "filename": file,
            "subfolder": subfolder,
            "type": "output" if save_output else "temp",
            "format": format,
            "frame_rate": frame_rate,
            "workflow": first_image_file,
            "fullpath": output_files[-1],
        }
        
        return {"ui": {"gifs": [preview]}, "result": ()}


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ComposeVideo": ComposeVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComposeVideo": "Compose Video"
}
