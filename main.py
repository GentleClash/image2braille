from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from img2ascii import BrailleAsciiConverter
import io

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

        image_bytes = await file.read()
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
        return ascii_art
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
