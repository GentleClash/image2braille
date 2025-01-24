import io
from PIL import Image
import numpy as np

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
                weights.append([
                    x - origin_x,
                    y - origin_y,
                    self.numerators[y][x] / self.denominator
                ])
        return weights

    def dither(self, image_array, threshold):
        height, width = image_array.shape
        output = np.zeros((height, width), dtype=np.uint8)
        
        working_array = image_array.copy().astype(float)
        weights = self.weights()
        
        for y in range(height):
            for x in range(width):
                old_pixel = working_array[y, x]
                new_pixel = 255 if old_pixel > threshold else 0
                output[y, x] = new_pixel
                
                error = old_pixel - new_pixel
                
                for weight_x, weight_y, weight in weights:
                    new_x = x + weight_x
                    new_y = y + weight_y
                    
                    if (0 <= new_x < width and 
                        0 <= new_y < height and 
                        weight != 0):
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

    def convert_to_braille(self, image_path, ascii_width=100, ditherer_name='floyd_steinberg', 
                          threshold=127, invert=False, color=False):
        color_image = Image.open(image_path)
            
        image = color_image.convert('L')
        
        # Calculating height maintaining aspect ratio
        aspect_ratio = image.height / image.width
        ascii_height = int(ascii_width * self.ASCII_X_DOTS * aspect_ratio / self.ASCII_Y_DOTS)
        
        # Resizing both images
        width = ascii_width * self.ASCII_X_DOTS
        height = ascii_height * self.ASCII_Y_DOTS
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        color_image = color_image.resize((width, height), Image.Resampling.LANCZOS)
        
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
                    r, g, b = color_image.getpixel((x, y))[:3]
                    
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

def main():
    converter = BrailleAsciiConverter()
    ascii_art = converter.convert_to_braille(
        image_path='sample.png',
        ascii_width=100,
        ditherer_name='stucki',
        threshold=200,
        invert=True,
        color=True  
    )
    print(ascii_art)
    
    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write(ascii_art)

if __name__ == '__main__':
    main()
