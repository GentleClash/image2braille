from PIL.ImageFile import ImageFile
import argparse, cv2
from typing import Any, List, Literal
import io
import numpy as np
from functools import lru_cache
from numba import njit, prange
import random
import tempfile
import os
from PIL import Image, ImageChops
from numpy._typing._array_like import NDArray
from weasyprint import HTML
import subprocess
import shutil

class KernelDitherer:
    def __init__(self, origin, numerators, denominator=1) -> None:
        self.origin = origin
        self.numerators = numerators
        self.denominator = denominator

    def weights(self) -> List[List[float]]:
        weights: List[List[float]] = []
        origin_x, origin_y = self.origin
        for y in range(len(self.numerators)):
            for x in range(len(self.numerators[y])):
                if self.numerators[y][x] == 0:
                    continue
                weights.append([
                    x - origin_x,
                    y - origin_y,
                    self.numerators[y][x] / self.denominator
                ])
        return weights

    def dither(self, image_array, threshold) -> np.ndarray:
        height, width = image_array.shape
        output = np.zeros((height, width), dtype=np.uint8)
        weights_list: List[List[float]] = self.weights()
        working_array = image_array.copy().astype(np.float32)
        weights = np.array(weights_list, dtype=np.float32) if weights_list else np.zeros((0, 3), dtype=np.float32)
        output = _apply_dither(working_array, output, weights, threshold, height, width)
        return output
    
@njit(parallel=True)
def _apply_dither(working_array, output, weights, threshold, height, width):        
    for y in prange(height):
        for x in range(width):
            old_pixel = working_array[y, x]
            new_pixel: Literal[255] | Literal[0] = 255 if old_pixel > threshold else 0
            output[y, x] = new_pixel
            
            error = old_pixel - new_pixel
            
            for weight_x, weight_y, weight in weights:
                new_x = x + int(weight_x)
                new_y = y + int(weight_y)
                
                if (0 <= new_x < width and 
                    0 <= new_y < height):
                    working_array[new_y, new_x] += error * weight
                    
    return output

class BrailleAsciiConverter:
    def __init__(self) -> None:
        self.ASCII_X_DOTS = 2
        self.ASCII_Y_DOTS = 4
        
        self.ditherers: dict[str, KernelDitherer] = {
            'threshold': KernelDitherer([0, 0], [], 1),
            'floyd_steinberg': KernelDitherer([1, 0], [
                [0, 0, 7],
                [3, 5, 1]
            ], 16),
            'stucki': KernelDitherer([2, 0], [
                [0, 0, 0, 8, 4],
                [2, 4, 8, 4, 2],
                [1, 2, 4, 2, 1]
            ], 42),
            'atkinson': KernelDitherer([1, 0], [
                [0, 0, 1, 1],
                [1, 1, 1, 0],
                [0, 1, 0, 0]
            ], 8),
            'jarvis_judice_ninke': KernelDitherer([1, 0], [
                [0, 0, 0, 7, 5],
                [3, 5, 7, 5, 3],
                [1, 3, 5, 3, 1]
            ], 48),

            'stucki_extended': KernelDitherer([2, 0], [
                [0, 0, 0, 8, 4],
                [2, 4, 8, 4, 2],
                [1, 2, 4, 2, 1]
            ], 42),
            'burkes': KernelDitherer([2, 0], [
                [0, 0, 0, 8, 4],
                [2, 4, 8, 4, 2]
            ], 32),

            'sierra3': KernelDitherer([1, 0], [
                [0, 0, 5, 3],
                [2, 4, 5, 4, 2],
                [0, 2, 3, 2, 0]
            ], 32),

            'sierra2': KernelDitherer([1, 0], [
                [0, 0, 4, 3],
                [1, 2, 3, 2, 1]
            ], 16),

            'sierra_2_4a': KernelDitherer([1, 0], [
                [0, 0, 2],
                [1, 1, 0]
            ], 4),

            'stevenson_arce': KernelDitherer([3, 0], [
                [0, 0, 0, 0, 32, 0],
                [12, 0, 26, 0, 30, 0, 16],
                [0, 12, 0, 26, 0, 12, 0],
                [5, 0, 12, 0, 12, 0, 5]
            ], 200)
        }

    @lru_cache(maxsize=40)
    def convert_to_braille(self, image_path, ascii_width=100, ditherer_name='floyd_steinberg', 
                          threshold=127, invert=False, color=False) -> str: 
        
        #color_image = Image.open(image_path)
        #image = color_image.convert('L')
        if isinstance(image_path, str):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        else:
            try:
                image_array = np.frombuffer(image_path.read(), np.uint8) #From buffer
                image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
                color_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            except Exception as e:
                raise ValueError('Invalid image file')

        aspect_ratio = image.shape[0] / image.shape[1]
        ascii_height = int(ascii_width * self.ASCII_X_DOTS * aspect_ratio / self.ASCII_Y_DOTS)
        
        # Resizing both images
        width: int = ascii_width * self.ASCII_X_DOTS
        height: int = ascii_height * self.ASCII_Y_DOTS
        image: cv2.Mat | np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]] = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
        color_image = cv2.resize(color_image, (width, height), interpolation=cv2.INTER_LANCZOS4)
        
        image_array = np.array(image)
    
        # Applying dithering
        ditherer = self.ditherers[ditherer_name]
        dithered = ditherer.dither(image_array, threshold)
        
        ascii_art = []
        target_value: Literal[255] | Literal[0] = 255 if invert else 0
        
        for y in range(0, height, self.ASCII_Y_DOTS):
            line: list = []
            for x in range(0, width, self.ASCII_X_DOTS):
                if color:
                    b, g, r = color_image[y, x] 
                    
                    if color=='fg':
                        color_code=f'<font color="#{r:02x}{g:02x}{b:02x}">'
                        color_reset='</font>'

                    elif color=='bg':
                        color_code=f'<font style="background-color:#{r:02x}{g:02x}{b:02x}">'
                        color_reset='</font>'
                    
                    elif color=='all':
                        color_code=f'<font style="color:#{r:02x}{g:02x}{b:02x};background-color:#{r:02x}{g:02x}{b:02x}">'
                        color_reset='</font>'

                    else:
                        color_code = f"\x1b[38;2;{r};{g};{b}m"
                    line.append(color_code)
                
                code = 0x2800
                
                if y + 3 < height and x + 1 < width and dithered[y + 3, x + 1] == target_value:
                    code |= 1 << 7
                if y + 3 < height and x < width and dithered[y + 3, x] == target_value:
                    code |= 1 << 6
                if y + 2 < height and x + 1 < width and dithered[y + 2, x + 1] == target_value:
                    code |= 1 << 5
                if y + 1 < height and x + 1 < width and dithered[y + 1, x + 1] == target_value:
                    code |= 1 << 4
                if y < height and x + 1 < width and dithered[y, x + 1] == target_value:
                    code |= 1 << 3
                if y + 2 < height and x < width and dithered[y + 2, x] == target_value:
                    code |= 1 << 2
                if y + 1 < height and x < width and dithered[y + 1, x] == target_value:
                    code |= 1 << 1
                if y < height and x < width and dithered[y, x] == target_value:
                    code |= 1 << 0
                
                line.append(chr(code))
            
            if color:
                if color=='fg' or color=='bg' or color=='all':
                    line.append(color_reset)
                else:
                    line.append("\x1b[0m")  

            ascii_art.append(''.join(line))
       
        if color in ['fg', 'bg', 'all']:
            return '<br>'.join(ascii_art)
        else:
            return '\n'.join(ascii_art)

    def invertDotArt(self, dotArt: str) -> str:
        dotArtInversionMap: dict[str, str] = {"⠀":"⣿","⠁":"⣾","⠂":"⣽","⠃":"⣼","⠄":"⣻","⠅":"⣺","⠆":"⣹","⠇":"⣸","⠈":"⣷","⠉":"⣶","⠊":"⣵","⠋":"⣴","⠌":"⣳","⠍":"⣲","⠎":"⣱","⠏":"⣰","⠐":"⣯","⠑":"⣮","⠒":"⣭","⠓":"⣬","⠔":"⣫","⠕":"⣪","⠖":"⣩","⠗":"⣨","⠘":"⣧","⠙":"⣦","⠚":"⣥","⠛":"⣤","⠜":"⣣","⠝":"⣢","⠞":"⣡","⠟":"⣠","⠠":"⣟","⠡":"⣞","⠢":"⣝","⠣":"⣜","⠤":"⣛","⠥":"⣚","⠦":"⣙","⠧":"⣘","⠨":"⣗","⠩":"⣖","⠪":"⣕","⠫":"⣔","⠬":"⣓","⠭":"⣒","⠮":"⣑","⠯":"⣐","⠰":"⣏","⠱":"⣎","⠲":"⣍","⠳":"⣌","⠴":"⣋","⠵":"⣊","⠶":"⣉","⠷":"⣈","⠸":"⣇","⠹":"⣆","⠺":"⣅","⠻":"⣄","⠼":"⣃","⠽":"⣂","⠾":"⣁","⠿":"⣀","⡀":"⢿","⡁":"⢾","⡂":"⢽","⡃":"⢼","⡄":"⢻","⡅":"⢺","⡆":"⢹","⡇":"⢸","⡈":"⢷","⡉":"⢶","⡊":"⢵","⡋":"⢴","⡌":"⢳","⡍":"⢲","⡎":"⢱","⡏":"⢰","⡐":"⢯","⡑":"⢮","⡒":"⢭","⡓":"⢬","⡔":"⢫","⡕":"⢪","⡖":"⢩","⡗":"⢨","⡘":"⢧","⡙":"⢦","⡚":"⢥","⡛":"⢤","⡜":"⢣","⡝":"⢢","⡞":"⢡","⡟":"⢠","⡠":"⢟","⡡":"⢞","⡢":"⢝","⡣":"⢜","⡤":"⢛","⡥":"⢚","⡦":"⢙","⡧":"⢘","⡨":"⢗","⡩":"⢖","⡪":"⢕","⡫":"⢔","⡬":"⢓","⡭":"⢒","⡮":"⢑","⡯":"⢐","⡰":"⢏","⡱":"⢎","⡲":"⢍","⡳":"⢌","⡴":"⢋","⡵":"⢊","⡶":"⢉","⡷":"⢈","⡸":"⢇","⡹":"⢆","⡺":"⢅","⡻":"⢄","⡼":"⢃","⡽":"⢂","⡾":"⢁","⡿":"⢀","⢀":"⡿","⢁":"⡾","⢂":"⡽","⢃":"⡼","⢄":"⡻","⢅":"⡺","⢆":"⡹","⢇":"⡸","⢈":"⡷","⢉":"⡶","⢊":"⡵","⢋":"⡴","⢌":"⡳","⢍":"⡲","⢎":"⡱","⢏":"⡰","⢐":"⡯","⢑":"⡮","⢒":"⡭","⢓":"⡬","⢔":"⡫","⢕":"⡪","⢖":"⡩","⢗":"⡨","⢘":"⡧","⢙":"⡦","⢚":"⡥","⢛":"⡤","⢜":"⡣","⢝":"⡢","⢞":"⡡","⢟":"⡠","⢠":"⡟","⢡":"⡞","⢢":"⡝","⢣":"⡜","⢤":"⡛","⢥":"⡚","⢦":"⡙","⢧":"⡘","⢨":"⡗","⢩":"⡖","⢪":"⡕","⢫":"⡔","⢬":"⡓","⢭":"⡒","⢮":"⡑","⢯":"⡐","⢰":"⡏","⢱":"⡎","⢲":"⡍","⢳":"⡌","⢴":"⡋","⢵":"⡊","⢶":"⡉","⢷":"⡈","⢸":"⡇","⢹":"⡆","⢺":"⡅","⢻":"⡄","⢼":"⡃","⢽":"⡂","⢾":"⡁","⢿":"⡀","⣀":"⠿","⣁":"⠾","⣂":"⠽","⣃":"⠼","⣄":"⠻","⣅":"⠺","⣆":"⠹","⣇":"⠸","⣈":"⠷","⣉":"⠶","⣊":"⠵","⣋":"⠴","⣌":"⠳","⣍":"⠲","⣎":"⠱","⣏":"⠰","⣐":"⠯","⣑":"⠮","⣒":"⠭","⣓":"⠬","⣔":"⠫","⣕":"⠪","⣖":"⠩","⣗":"⠨","⣘":"⠧","⣙":"⠦","⣚":"⠥","⣛":"⠤","⣜":"⠣","⣝":"⠢","⣞":"⠡","⣟":"⠠","⣠":"⠟","⣡":"⠞","⣢":"⠝","⣣":"⠜","⣤":"⠛","⣥":"⠚","⣦":"⠙","⣧":"⠘","⣨":"⠗","⣩":"⠖","⣪":"⠕","⣫":"⠔","⣬":"⠓","⣭":"⠒","⣮":"⠑","⣯":"⠐","⣰":"⠏","⣱":"⠎","⣲":"⠍","⣳":"⠌","⣴":"⠋","⣵":"⠊","⣶":"⠉","⣷":"⠈","⣸":"⠇","⣹":"⠆","⣺":"⠅","⣻":"⠄","⣼":"⠃","⣽":"⠂","⣾":"⠁","⣿":"⠀"}
        invertedDotArt: str = ""
        for char in dotArt:
            invertedDotArt += dotArtInversionMap.get(char, char)
        return invertedDotArt

    def get_crop_dimensions(self, frame_files, frames_dir, ascii_width, ditherer_name, threshold, invert, target_width, target_height) -> tuple:
        """
        Extract random frames and get max crop dimensions to avoid per-frame bbox calculation
        """

        # Select 7 random frames
        sample_frames: list = random.sample(frame_files, min(7, len(frame_files)))
        max_width = 0
        max_height = 0

        temp_crop_dir: str = tempfile.mkdtemp(prefix='crop_test_')

        try:
            for frame_file in sample_frames:
                frame_path: str = os.path.join(frames_dir, frame_file)

                # Convert frame to ASCII
                ascii_art: str = self.convert_to_braille(
                    image_path=frame_path,
                    ascii_width=ascii_width,
                    ditherer_name=ditherer_name,
                    threshold=threshold,
                    invert=invert,
                    color=False
                )

                # Render to get bbox
                html_content: str = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        @page {{ size: {target_width}px {target_height}px; margin: 0; }}
                        * {{ margin: 0; padding: 0;}}
                        html, body {{
                            font-family: 'Noto Sans Mono', 'DejaVu Sans Mono', 'Courier New', monospace;
                            font-size: 12pt;
                            line-height: 1.0;
                            background: white;
                            color: black;
                            white-space: pre;
                            margin: 0;
                            padding: 0;
                            width: 100%;
                            height: 100%;
                            overflow: hidden;
                        }}
                        body {{
                            display: block;
                        }}
                    </style>
                </head>
                <body>{ascii_art}</body>
                </html>
                """

                img_bytes = HTML(string=html_content).write_png()
                img: ImageFile = Image.open(io.BytesIO(img_bytes))

                # Get bbox
                bg_color: float | tuple[int, ...] | None = img.getpixel((0, 0))
                bg: Image.Image = Image.new(img.mode, img.size, bg_color)
                diff: Image.Image = ImageChops.difference(img, bg)
                diff = ImageChops.add(diff, diff, 2.0, -100)
                bbox: tuple[int, int, int, int] | None = diff.getbbox()

                if bbox:
                    width: int = bbox[2] - bbox[0]
                    height: int = bbox[3] - bbox[1]
                    max_width: int = max(max_width, width)
                    max_height: int = max(max_height, height)

        finally:
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_crop_dir, ignore_errors=True)

        return max_width, max_height




    def convert_video_to_braille_video(self, video_path, ascii_width=100, ditherer_name='atkinson',
                          threshold=127, invert=False, color=False, frame_skip=1, progress_callback=None,
                          output_path=None, temp_dir=None) -> tuple:
        """
        Convert video to Braille ASCII art video with preserved audio
        
        Parameters:
        - video_path: Path to video file or file-like object
        - ascii_width: Width of the ASCII art
        - ditherer_name: Dithering algorithm to use
        - threshold: Threshold for dithering [0-255]
        - invert: Whether to invert the image
        - color: Color mode (ignored for video output)
        - frame_skip: Process every Nth frame (for performance)
        - progress_callback: Function to call with progress updates
        - output_path: Path to save output video (if None, a temp file is created)
        - temp_dir: Directory for temporary files
        
        Returns:
        - Path to output video file
        - Video metadata
        """
        
        def render_braille_frame(ascii_text, output_path, target_width=800, target_height=600, crop_width=None, crop_height=None) -> None:
            """Render Braille ASCII text to image using WeasyPrint"""
            
            html_content: str = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    @page {{ size: {target_width}px {target_height}px; margin: 0; }}
                    * {{ margin: 0; padding: 0; }}
                    html, body {{
                        font-family: 'Noto Sans Mono', 'DejaVu Sans Mono', 'Courier New', monospace;
                        font-size: 12pt;
                        line-height: 1.0;
                        background: white;
                        color: black;
                        white-space: pre;
                        margin: 0;
                        padding: 0;
                        width: 100%;
                        height: 100%;
                        overflow: hidden;
                    }}
                    body {{
                        display: block;
                    }}
                </style>
            </head>
            <body>{ascii_text}</body>
            </html>
            """
            # Convert HTML to PNG
            try:
                img_bytes = HTML(string=html_content).write_png()
            except Exception as e:
                print(f"Error rendering HTML with WeasyPrint: {e}")
                return
            try:
                img: ImageFile = Image.open(io.BytesIO(img_bytes))
            except Exception as e:
                print(f"Error opening image with Pillow: {e}")
                return
            if crop_width and crop_height:
                crop_box = (0, 0, crop_width, crop_height)
                cropped_img: Image.Image = img.crop(crop_box)
                cropped_img.save(output_path)
            else:
                img.save(output_path)

        # Create temp directory if not provided
        if temp_dir is None:
            temp_dir: str = tempfile.mkdtemp(prefix='braille_video_')
        
        frames_dir: str = os.path.join(temp_dir, "frames")
        ascii_frames_dir: str = os.path.join(temp_dir, "ascii_frames")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(ascii_frames_dir, exist_ok=True)
        
        # Create temp file for output if not specified
        if output_path is None:
            output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            output_path: str = output_file.name
            output_file.close()
        
        # Handle input video (file path vs file-like object)
        if isinstance(video_path, str):
            video_file: str = video_path
        else:
            # Handle file-like object
            temp_input = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_input.write(video_path.read())
            temp_input.close()
            video_file = temp_input.name
        
        try:
            # Get video properties using OpenCV
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            original_fps: float = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration: float | Literal[0] = frame_count / original_fps if original_fps > 0 else 0
            
            # Calculate target FPS after frame skipping
            target_fps: float = original_fps / frame_skip
            
            # Extract frames
            frame_number = 0
            saved_frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames according to frame_skip parameter
                if frame_number % frame_skip == 0:
                    frame_path: str = os.path.join(frames_dir, f"frame_{saved_frame_count:06d}.png")
                    cv2.imwrite(frame_path, frame)
                    saved_frame_count += 1
                
                frame_number += 1
                
                # Update progress
                if progress_callback and frame_number % 30 == 0:
                    progress_callback(0.3 * frame_number / frame_count)
            
            cap.release()
            
            # Get list of extracted frames
            frame_files: List[str] = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
            total_frames: int = len(frame_files)
            
            if total_frames == 0:
                raise ValueError("No frames were extracted from the video")

            
            
            # Calculate target image dimensions
            target_width: int = min(1920, ascii_width * 8)  # Scale up for visibility
            target_height = int(target_width * height / width)  # Maintain aspect ratio

            crop_width, crop_height = self.get_crop_dimensions(
                frame_files, frames_dir, ascii_width, ditherer_name, threshold, invert, target_width, target_height
            )
            # Process each frame to ASCII art
            for i, frame_file in enumerate(frame_files):
                frame_path = os.path.join(frames_dir, frame_file)
                
                # Convert frame to ASCII using existing method
                ascii_art: str = self.convert_to_braille(
                    image_path=frame_path,
                    ascii_width=ascii_width,
                    ditherer_name=ditherer_name,
                    threshold=threshold,
                    invert=invert,
                    color=False  # No color for video output
                )
                
                # Render ASCII art to image
                ascii_frame_path: str = os.path.join(ascii_frames_dir, f"ascii_{frame_file}")
                render_braille_frame(ascii_art, ascii_frame_path, target_width, target_height, crop_width, crop_height)
                
                # Update progress
                if progress_callback:
                    progress_callback(0.3 + 0.6 * i / total_frames)
            
            # Extract audio from original video
            audio_path: str = os.path.join(temp_dir, "audio.aac")
            extract_audio_cmd = [
                "ffmpeg", "-y", "-i", video_file, "-vn", "-acodec", "aac", audio_path
            ]
            
            has_audio = False
            try:
                result = subprocess.run(extract_audio_cmd, capture_output=True, text=True)
                if result.returncode == 0 and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                    has_audio = True
            except subprocess.CalledProcessError:
                pass
            
            # Create video from ASCII frames
            if has_audio:
                create_video_cmd = [
                    "ffmpeg", "-y", "-framerate", str(target_fps),
                    "-i", os.path.join(ascii_frames_dir, "ascii_frame_%06d.png"),
                    "-i", audio_path,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest",
                    "-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
                    output_path
                ]
            else:
                create_video_cmd = [
                    "ffmpeg", "-y", "-framerate", str(target_fps),
                    "-i", os.path.join(ascii_frames_dir, "ascii_frame_%06d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
                    output_path
                ]
            
            # Execute video creation
            result = subprocess.run(create_video_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Error creating video: {result.stderr}")
            
            # Update final progress
            if progress_callback:
                progress_callback(1.0)
            
            # Prepare metadata
            metadata = {
                "original_fps": original_fps,
                "target_fps": target_fps,
                "original_resolution": f"{width}x{height}",
                "output_resolution": f"{target_width}x{target_height}",
                "duration": duration,
                "frame_count": total_frames,
                "has_audio": has_audio,
                "temp_dir": temp_dir
            }
            
            return output_path, metadata
        
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise e

def main():
    converter = BrailleAsciiConverter()
    parser = argparse.ArgumentParser(description='Convert an image to ASCII art.')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--width', type=int, default=100, help='Width of the ASCII art')
    parser.add_argument('--ditherer', default='floyd_steinberg', choices=converter.ditherers.keys(), help='Dithering algorithm')
    parser.add_argument('--threshold', type=int, default=127, help='Threshold for dithering [0-256]')
    parser.add_argument('--invert', action='store_true', help='Invert the image')
    parser.add_argument('--color', choices=['ansi', 'fg', 'bg', 'all'], default=None, help='Color the ASCII art') 
    parser.add_argument('--output', help='Output file path')
    args = parser.parse_args()

    ascii_art = converter.convert_to_braille(
        image_path=args.image_path,
        ascii_width=args.width,
        ditherer_name=args.ditherer,
        threshold=args.threshold,
        invert=args.invert,
        color=args.color
    )

    print(ascii_art)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(ascii_art)
        print(f'ASCII art saved to {args.output}')

if __name__ == '__main__':
    main()
