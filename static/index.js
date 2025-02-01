document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const asciiPreview = document.getElementById('asciiPreview');
    const copyBtn = document.getElementById('copyBtn');
    const asciiWidthInput = document.getElementById('asciiWidth');
    const thresholdInput = document.getElementById('threshold');
    const dithererSelect = document.getElementById('dithererName');
    const invertCheckbox = document.getElementById('invert');
    const colorSelect = document.getElementById('color');

    const convertImage = async () => {
        if (!fileInput.files.length) return;

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('width', asciiWidthInput.value);
        formData.append('threshold', thresholdInput.value);
        formData.append('ditherer', dithererSelect.value);
        formData.append('invert', invertCheckbox.checked);
        formData.append('color', colorSelect.value);
        console.group('Form Data');
        for (const [key, value] of formData.entries()) {
            console.log(`${key}: ${value}`);
        }
        console.groupEnd();



        try {
            const response = await fetch('/convert-to-ascii', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Conversion failed');
            }

            const asciiText = await response.text();

            asciiPreview.innerHTML = asciiText;
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to convert image to ASCII');
        }
    };

    fileInput.addEventListener('change', convertImage);
    asciiWidthInput.addEventListener('input', convertImage);
    thresholdInput.addEventListener('input', convertImage);
    dithererSelect.addEventListener('change', convertImage);
    invertCheckbox.addEventListener('change', convertImage);
    colorSelect.addEventListener('change', convertImage);

    function processPixelText(elementId) {
        const element = document.getElementById(elementId);
        const modifiedInnerHTML = element.innerHTML.replace(/<br>/g, '<font>\n</font>');
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = modifiedInnerHTML;
        
        return tempDiv.textContent;
      }
      
      function extractTextWithNewlines(element) {
        const temp = document.createElement('div');
        temp.innerHTML = element.innerHTML;
        
        temp.querySelectorAll('br').forEach(br => {
          br.replaceWith('\n');
        });
        
        return temp.textContent;
      }


    copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(extractTextWithNewlines(asciiPreview))
            .then(() => {
                alert('ASCII text copied to clipboard');
            })
            .catch(err => {
                console.error('Failed to copy:', err);
                alert('Failed to copy ASCII text');
            });
    });
});