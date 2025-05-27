import subprocess
import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from img2ascii import BrailleAsciiConverter
import io
import uuid, time
from cachetools import LRUCache
from PIL import Image
from PIL import ImageFilter
import tempfile, os, shutil, time


cache: LRUCache = LRUCache(maxsize=40)

app: FastAPI = FastAPI()
templates: Jinja2Templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
conversion_progress = {}
temporary_directories = {}


def update_progress(task_id, progress, status="processing") -> None:
    conversion_progress[task_id] = {
        "progress": progress,
        "status": status,
        "last_update": time.time()
    }

@app.get("/conversion-progress/{task_id}")
async def check_progress(task_id: str):
    if task_id not in conversion_progress:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return conversion_progress[task_id]


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/convert-to-ascii", response_class=HTMLResponse)
async def convert_to_ascii(
    file: UploadFile, 
    width: int = Form(100),
    threshold: float = Form(128),
    ditherer: str = Form("stucki"),
    invert: bool = Form(False),
    color: str = Form("")
):
    try:

        if file.content_type.split("/")[0] == "text":
            _ = file.file.read().decode()
            if invert:
                converter = BrailleAsciiConverter()
                return converter.invertDotArt(_)
            return _
        
        else:
            image_bytes = await file.read()
            image_bytes_hash = hash(image_bytes)
            width = min(300, width)
            key = (image_bytes_hash, width, threshold, ditherer, invert, color)
            if key in cache:
                return cache[key]

            image_bytes_io = io.BytesIO(image_bytes)

            converter = BrailleAsciiConverter()
            ascii_art = converter.convert_to_braille(
                image_path=image_bytes_io,
                ascii_width=width,
                threshold=threshold,
                ditherer_name=ditherer.lower(),
                invert=invert,
                color=color
            )

            cache[key] = ascii_art
            return ascii_art
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/resize", response_class=HTMLResponse)
def show_resize_page(request: Request):
    return templates.TemplateResponse("resize.html", {"request": request})

@app.post("/resize-image", response_class=HTMLResponse)
async def resize_image(
    request: Request, 
    file: UploadFile, 
    max_size_kb: int = Form(...), 
    format: str = Form("JPEG")
):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Not an image file.")

        format = format.upper()
        if format not in ["JPEG", "PNG", "WEBP"]:
            raise HTTPException(status_code=400, detail="Unsupported format.")

        # Read the image
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        original_width, original_height = image.size
        
        # Apply a light sharpening filter to maintain perceived quality
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=75, threshold=3))
        
        # First attempt: just try the original size with specified quality
        quality = 95 if format in ["JPEG", "WEBP"] else None
        scale_factor = 1.0
        max_attempts = 16
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            output = io.BytesIO()
            
            # Resize the image if scale factor is less than 1
            if scale_factor < 1.0:
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            else:
                resized_image = image
            
            # Save with current quality
            save_kwargs = {"format": format}
            if format in ["JPEG", "WEBP"] and quality is not None:
                save_kwargs["quality"] = quality
            
            resized_image.save(output, **save_kwargs)
            
            # Check if size requirement is met
            size_kb = output.tell() / 1024
            if size_kb <= max_size_kb:
                break
                
            # Try reducing quality first if it's above threshold
            if format in ["JPEG", "WEBP"] and quality is not None and quality > 50:
                quality -= 10
            # Then try reducing dimensions
            else:
                scale_factor *= 0.8  # Reduce by 20% each time
                if quality is not None and quality > 50:
                    quality = max(50, quality - 5)  # Continue reducing quality
        
        # If we couldn't meet the target size, use the smallest version we got
        output.seek(0)
        
        # Return the image as base64 for the template
        import base64
        encoded = base64.b64encode(output.getvalue()).decode('utf-8')
        mime = f"image/{format.lower()}"
        data_url = f"data:{mime};base64,{encoded}"
        
        # Include metadata about the compression in the response
        final_size_kb = output.tell() / 1024
        compression_info = {
            "original_size": len(content) / 1024,
            "final_size": final_size_kb,
            "target_size": max_size_kb,
            "quality": quality,
            "scale_factor": scale_factor,
            "original_dimensions": f"{original_width}x{original_height}",
            "final_dimensions": f"{int(original_width * scale_factor)}x{int(original_height * scale_factor)}" if scale_factor < 1.0 else f"{original_width}x{original_height}"
        }

        return templates.TemplateResponse(
            "download.html", 
            {
                "request": request, 
                "image_data": data_url,
                "compression_info": compression_info
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resize failed: {str(e)}")

    
@app.get("/download-ascii-video/{task_id}")
async def download_ascii_video(task_id: str):
    if task_id not in conversion_progress:
        raise HTTPException(status_code=404, detail="Task not found")
    
    progress_info = conversion_progress[task_id]
    
    if progress_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Video conversion not completed yet")
    
    # The file path should be stored in the progress info
    if "file_path" not in progress_info:
        raise HTTPException(status_code=500, detail="File path not found")
    
    print(f"Returning file: {progress_info['file_path']} for task {task_id}")
    
    return FileResponse(
        path=progress_info["file_path"],
        filename=progress_info["filename"],
        media_type="video/mp4"
    )

@app.post("/convert-video-to-ascii")
async def convert_video_to_ascii(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    width: int = Form(100),
    threshold: float = Form(128),
    ditherer: str = Form("stucki"),
    invert: bool = Form(False),
    frame_skip: int = Form(1)
):
    """
    Convert uploaded video to ASCII art and return as downloadable video file
    """
    try:
        if not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="Uploaded file is not a video")
        
        # Create task ID for progress tracking
        task_id = str(uuid.uuid4())
        update_progress(task_id, 0.0, "initializing")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Save uploaded file
        temp_input_path = os.path.join(temp_dir, "input" + os.path.splitext(file.filename)[1])
        with open(temp_input_path, "wb") as f:
            video_bytes = await file.read()
            f.write(video_bytes)
        
        # Update progress after file received
        update_progress(task_id, 0.1, "file_received")
        
        # Prepare output path
        output_filename = f"ascii_{os.path.splitext(file.filename)[0]}.mp4"
        output_path = os.path.join(temp_dir, output_filename)
        
        # Define progress callback
        def progress_callback(progress_value : float ) -> None:
            # Scale progress from 0.2-0.9 range
            scaled_progress: float = 0.2 + (progress_value * 0.7)
            update_progress(task_id, scaled_progress, "processing_frames")
        
        # Create converter
        converter = BrailleAsciiConverter()
        
        # Start conversion in background task
        def process_video() -> None:
            try:
                update_progress(task_id, 0.2, "extracting_frames")
                
                # Convert video to ASCII
                result_path, metadata = converter.convert_video_to_braille_video(
                    video_path=temp_input_path,
                    ascii_width=min(300, width),
                    ditherer_name=ditherer.lower(),
                    threshold=threshold,
                    invert=invert,
                    frame_skip=frame_skip,
                    progress_callback=progress_callback,
                    temp_dir=temp_dir,
                    output_path=output_path
                )
                
                update_progress(task_id, 1.0, "completed")
                conversion_progress[task_id]["file_path"] = result_path  
                conversion_progress[task_id]["filename"] = output_filename
                temporary_directories[temp_dir] = time.time()
                
                # Keep task info for a while before cleanup
                def cleanup_task() -> None:
                    time.sleep(60)  # Keep progress info for a minute
                    if task_id in conversion_progress:
                        del conversion_progress[task_id]
                
                def remove_temp_dir() -> None:
                    try:
                        current_time = time.time()
                        if temp_dir in temporary_directories:
                            creation_time = temporary_directories[temp_dir]
                            if current_time - creation_time > 300:
                                shutil.rmtree(temp_dir, ignore_errors=True)
                                del temporary_directories[temp_dir]
                    except Exception as e:
                        print(f"Error removing temp dir {temp_dir}: {e}")
                
                background_tasks.add_task(cleanup_task)
                background_tasks.add_task(remove_temp_dir)
                
            except Exception as e:
                update_progress(task_id, 0, f"error: {str(e)}")
                print(e)
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass
        
        # Start processing
        background_tasks.add_task(process_video)
        
        # Return task ID for client to check progress
        return {"task_id": task_id, "status": "processing"}
        
    except Exception as e:
        # Clean up on error
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/video", response_class=HTMLResponse)
def video_upload_form(request: Request):
    return templates.TemplateResponse("video_upload.html", {"request": request})

@app.post("/preview-video-frames")
async def preview_video_frames(
    file: UploadFile = File(...),
    num_frames: int = 20,
    width: int = Form(100),
    threshold: float = Form(128),
    ditherer: str = Form("stucki"),
    invert: bool = Form(False)
    ):
    """
    Extract sample frames from a video and convert them to ASCII frame images
    """
    try:
        if not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="Uploaded file is not a video")

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        preview_frames_dir = os.path.join(temp_dir, "preview_frames")
        os.makedirs(preview_frames_dir, exist_ok=True)

        # Save uploaded file
        temp_input_path = os.path.join(temp_dir, "input" + os.path.splitext(file.filename)[1])
        with open(temp_input_path, "wb") as f:
            video_bytes = await file.read()
            f.write(video_bytes)

        # Open video
        video = cv2.VideoCapture(temp_input_path)
        if not video.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        # Get video metadata
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0

        # Calculate frame positions for preview (evenly distributed)
        if frame_count <= num_frames:
            frame_positions = list(range(frame_count))
        else:
            step = frame_count / num_frames
            frame_positions = [int(i * step) for i in range(num_frames)]

        # Extract frames and save as images
        saved_frames = []
        for i, pos in enumerate(frame_positions):
            # Set position
            video.set(cv2.CAP_PROP_POS_FRAMES, pos)
            
            # Read frame
            ret, frame = video.read()
            if not ret:
                continue
            
            # Save frame as image
            frame_path = os.path.join(preview_frames_dir, f"frame_{i:06d}.png")
            cv2.imwrite(frame_path, frame)
            
            # Store metadata
            timestamp = pos / fps if fps > 0 else 0
            saved_frames.append({
                "frame_path": frame_path,
                "time": timestamp,
                "frame": pos
            })

        video.release()
        
        if not saved_frames:
            raise HTTPException(status_code=400, detail="No frames could be extracted")

        # Create temporary video from extracted frames
        temp_video_path = os.path.join(temp_dir, "preview_video.mp4")
        create_preview_video_cmd = [
            "ffmpeg", "-y", "-framerate", "1",  # 1 FPS for preview
            "-i", os.path.join(preview_frames_dir, "frame_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
            temp_video_path
        ]
        
        result = subprocess.run(create_preview_video_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Error creating preview video: {result.stderr}")

        # Use existing video converter to process the preview video
        converter = BrailleAsciiConverter()
        
        try:
            output_video_path, metadata = converter.convert_video_to_braille_video(
                video_path=temp_video_path,
                ascii_width=min(300, width),
                ditherer_name=ditherer.lower(),
                threshold=threshold,
                invert=invert,
                color=False,
                frame_skip=1,
                progress_callback=None
            )
            
            # Get the ASCII frame images from the temp directory used by converter
            converter_temp_dir = metadata.get('temp_dir')
            ascii_frames_dir = os.path.join(converter_temp_dir, "ascii_frames")
            
            # Collect ASCII frame images and convert to base64 or file paths
            frames_data = []
            if os.path.exists(ascii_frames_dir):
                ascii_frame_files = sorted([f for f in os.listdir(ascii_frames_dir) if f.startswith("ascii_")])
                
                for i, (frame_info, ascii_frame_file) in enumerate(zip(saved_frames, ascii_frame_files)):
                    ascii_frame_path = os.path.join(ascii_frames_dir, ascii_frame_file)
                    
                    if os.path.exists(ascii_frame_path):
                        # Convert to base64 for API response
                        with open(ascii_frame_path, "rb") as img_file:
                            import base64
                            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                        
                        frames_data.append({
                            "image_base64": img_base64,
                            "time": frame_info["time"],
                            "frame": frame_info["frame"]
                        })
            
            # Clean up converter temp directory
            if converter_temp_dir and os.path.exists(converter_temp_dir):
                shutil.rmtree(converter_temp_dir, ignore_errors=True)
            
            # Clean up output video
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error converting frames: {str(e)}")

        # Clean up main temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        return {
            "frames": frames_data,
            "metadata": {
                "frame_count": frame_count,
                "fps": fps,
                "duration": duration,
                "preview_frames_count": len(frames_data)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        try:
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))
    