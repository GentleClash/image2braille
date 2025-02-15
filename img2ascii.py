import argparse, cv2
import numpy as np
from functools import lru_cache
from numba import njit, prange

class KernelDitherer:
    def __init__(self, origin, numerators, denominator=1):
        self.origin = origin
        self.numerators = numerators
        self.denominator = denominator

    def weights(self):
        weights = []
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

    def dither(self, image_array, threshold):
        height, width = image_array.shape
        output = np.zeros((height, width), dtype=np.uint8)
        weights_list = self.weights()
        working_array = image_array.copy().astype(np.float32)
        weights = np.array(weights_list, dtype=np.float32) if weights_list else np.zeros((0, 3), dtype=np.float32)
        output = _apply_dither(working_array, output, weights, threshold, height, width)
        return output
    
@njit(parallel=True)
def _apply_dither(working_array, output, weights, threshold, height, width):        
    for y in prange(height):
        for x in range(width):
            old_pixel = working_array[y, x]
            new_pixel = 255 if old_pixel > threshold else 0
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
    def __init__(self):
        self.ASCII_X_DOTS = 2
        self.ASCII_Y_DOTS = 4
        
        self.ditherers = {
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
                          threshold=127, invert=False, color=False): 
        
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
        width = ascii_width * self.ASCII_X_DOTS
        height = ascii_height * self.ASCII_Y_DOTS
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
        color_image = cv2.resize(color_image, (width, height), interpolation=cv2.INTER_LANCZOS4)
        
        image_array = np.array(image)
    
        # Applying dithering
        ditherer = self.ditherers[ditherer_name]
        dithered = ditherer.dither(image_array, threshold)
        
        ascii_art = []
        target_value = 255 if invert else 0
        
        for y in range(0, height, self.ASCII_Y_DOTS):
            line = []
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
        dotArtInversionMap = {"⠀":"⣿","⠁":"⣾","⠂":"⣽","⠃":"⣼","⠄":"⣻","⠅":"⣺","⠆":"⣹","⠇":"⣸","⠈":"⣷","⠉":"⣶","⠊":"⣵","⠋":"⣴","⠌":"⣳","⠍":"⣲","⠎":"⣱","⠏":"⣰","⠐":"⣯","⠑":"⣮","⠒":"⣭","⠓":"⣬","⠔":"⣫","⠕":"⣪","⠖":"⣩","⠗":"⣨","⠘":"⣧","⠙":"⣦","⠚":"⣥","⠛":"⣤","⠜":"⣣","⠝":"⣢","⠞":"⣡","⠟":"⣠","⠠":"⣟","⠡":"⣞","⠢":"⣝","⠣":"⣜","⠤":"⣛","⠥":"⣚","⠦":"⣙","⠧":"⣘","⠨":"⣗","⠩":"⣖","⠪":"⣕","⠫":"⣔","⠬":"⣓","⠭":"⣒","⠮":"⣑","⠯":"⣐","⠰":"⣏","⠱":"⣎","⠲":"⣍","⠳":"⣌","⠴":"⣋","⠵":"⣊","⠶":"⣉","⠷":"⣈","⠸":"⣇","⠹":"⣆","⠺":"⣅","⠻":"⣄","⠼":"⣃","⠽":"⣂","⠾":"⣁","⠿":"⣀","⡀":"⢿","⡁":"⢾","⡂":"⢽","⡃":"⢼","⡄":"⢻","⡅":"⢺","⡆":"⢹","⡇":"⢸","⡈":"⢷","⡉":"⢶","⡊":"⢵","⡋":"⢴","⡌":"⢳","⡍":"⢲","⡎":"⢱","⡏":"⢰","⡐":"⢯","⡑":"⢮","⡒":"⢭","⡓":"⢬","⡔":"⢫","⡕":"⢪","⡖":"⢩","⡗":"⢨","⡘":"⢧","⡙":"⢦","⡚":"⢥","⡛":"⢤","⡜":"⢣","⡝":"⢢","⡞":"⢡","⡟":"⢠","⡠":"⢟","⡡":"⢞","⡢":"⢝","⡣":"⢜","⡤":"⢛","⡥":"⢚","⡦":"⢙","⡧":"⢘","⡨":"⢗","⡩":"⢖","⡪":"⢕","⡫":"⢔","⡬":"⢓","⡭":"⢒","⡮":"⢑","⡯":"⢐","⡰":"⢏","⡱":"⢎","⡲":"⢍","⡳":"⢌","⡴":"⢋","⡵":"⢊","⡶":"⢉","⡷":"⢈","⡸":"⢇","⡹":"⢆","⡺":"⢅","⡻":"⢄","⡼":"⢃","⡽":"⢂","⡾":"⢁","⡿":"⢀","⢀":"⡿","⢁":"⡾","⢂":"⡽","⢃":"⡼","⢄":"⡻","⢅":"⡺","⢆":"⡹","⢇":"⡸","⢈":"⡷","⢉":"⡶","⢊":"⡵","⢋":"⡴","⢌":"⡳","⢍":"⡲","⢎":"⡱","⢏":"⡰","⢐":"⡯","⢑":"⡮","⢒":"⡭","⢓":"⡬","⢔":"⡫","⢕":"⡪","⢖":"⡩","⢗":"⡨","⢘":"⡧","⢙":"⡦","⢚":"⡥","⢛":"⡤","⢜":"⡣","⢝":"⡢","⢞":"⡡","⢟":"⡠","⢠":"⡟","⢡":"⡞","⢢":"⡝","⢣":"⡜","⢤":"⡛","⢥":"⡚","⢦":"⡙","⢧":"⡘","⢨":"⡗","⢩":"⡖","⢪":"⡕","⢫":"⡔","⢬":"⡓","⢭":"⡒","⢮":"⡑","⢯":"⡐","⢰":"⡏","⢱":"⡎","⢲":"⡍","⢳":"⡌","⢴":"⡋","⢵":"⡊","⢶":"⡉","⢷":"⡈","⢸":"⡇","⢹":"⡆","⢺":"⡅","⢻":"⡄","⢼":"⡃","⢽":"⡂","⢾":"⡁","⢿":"⡀","⣀":"⠿","⣁":"⠾","⣂":"⠽","⣃":"⠼","⣄":"⠻","⣅":"⠺","⣆":"⠹","⣇":"⠸","⣈":"⠷","⣉":"⠶","⣊":"⠵","⣋":"⠴","⣌":"⠳","⣍":"⠲","⣎":"⠱","⣏":"⠰","⣐":"⠯","⣑":"⠮","⣒":"⠭","⣓":"⠬","⣔":"⠫","⣕":"⠪","⣖":"⠩","⣗":"⠨","⣘":"⠧","⣙":"⠦","⣚":"⠥","⣛":"⠤","⣜":"⠣","⣝":"⠢","⣞":"⠡","⣟":"⠠","⣠":"⠟","⣡":"⠞","⣢":"⠝","⣣":"⠜","⣤":"⠛","⣥":"⠚","⣦":"⠙","⣧":"⠘","⣨":"⠗","⣩":"⠖","⣪":"⠕","⣫":"⠔","⣬":"⠓","⣭":"⠒","⣮":"⠑","⣯":"⠐","⣰":"⠏","⣱":"⠎","⣲":"⠍","⣳":"⠌","⣴":"⠋","⣵":"⠊","⣶":"⠉","⣷":"⠈","⣸":"⠇","⣹":"⠆","⣺":"⠅","⣻":"⠄","⣼":"⠃","⣽":"⠂","⣾":"⠁","⣿":"⠀"}
        invertedDotArt = ""
        for char in dotArt:
            invertedDotArt += dotArtInversionMap.get(char, char)
        return invertedDotArt

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
