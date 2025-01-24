# image2ascii
Image to braille/ascii web app

Color Logic by https://github.com/TheFel0x/img2braille?tab=readme-ov-file
Ditherers by https://github.com/hbldh/hitherdither/tree/master


## Usage

1. Clone the repository
2. Install the requirements
3. Run the app

```bash
git clone https://github.com/GentleClash/image2braille
cd image2braille
pip install -r requirements.txt
uvicorn main:app --reload
``` 

Feel free to fork the repository and make your own changes.
Play around with settings, generally threshold (no dithering) with slightly higher threshold values give better results.

### Warning : Do not choose ANSIFG color scheme, it was meant for terminal output, not for webpages.