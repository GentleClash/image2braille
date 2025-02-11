from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from img2ascii import BrailleAsciiConverter
import io
from cachetools import LRUCache

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
