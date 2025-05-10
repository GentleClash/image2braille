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

        image = Image.open(io.BytesIO(await file.read()))
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        
        output = io.BytesIO()

        quality = 95 if format in ["JPEG", "WEBP"] else None
        while quality is None or quality > 10:
            output.seek(0)
            save_kwargs = {"format": format}
            if format in ["JPEG", "WEBP"]:
                save_kwargs["quality"] = quality

            image.save(output, **save_kwargs)
            size_kb = output.tell() / 1024
            if size_kb <= max_size_kb or format not in ["JPEG", "WEBP"]:
                break
            quality -= 5

        output.seek(0)
        #return StreamingResponse(output, media_type=f"image/{format.lower()}")
        import base64

        encoded = base64.b64encode(output.getvalue()).decode('utf-8')
        mime = f"image/{format.lower()}"
        data_url = f"data:{mime};base64,{encoded}"

        return templates.TemplateResponse("download.html", {"request": request, "image_data": data_url})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resize failed: {str(e)}")
