from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from img2ascii import BrailleAsciiConverter
import io
from cachetools import LRUCache
#Temporary endpoint added
from fastapi.responses import StreamingResponse
from PIL import Image
from PIL import ImageFilter

cache = LRUCache(maxsize=40)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


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
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        
        # First attempt: just try the original size with specified quality
        quality = 95 if format in ["JPEG", "WEBP"] else None
        scale_factor = 1.0
        max_attempts = 10
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
                    quality = max(50, quality - 5)  # Continue reducing quality but more gently
        
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
