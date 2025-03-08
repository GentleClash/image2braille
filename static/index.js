document.addEventListener('DOMContentLoaded', () => {
    let isConverting = false;
    const fileInput = document.getElementById('fileInput');
    const asciiPreview = document.getElementById('asciiPreview');
    const copyBtn = document.getElementById('copyBtn');
    const asciiWidthInput = document.getElementById('asciiWidth');
    const thresholdInput = document.getElementById('threshold');
    const dithererSelect = document.getElementById('dithererName');
    const invertCheckbox = document.getElementById('invert');
    const colorSelect = document.getElementById('color');
    const downloadBtn = document.getElementById('downloadBtn');


    const convertImage = async () => {
        if (!fileInput.files.length) return;
        if (isConverting) return;
        isConverting = true;
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
        } finally {
            isConverting = false;
        }
    };

    fileInput.addEventListener('change', convertImage);
    asciiWidthInput.addEventListener('input', convertImage);
    thresholdInput.addEventListener('input', convertImage);
    dithererSelect.addEventListener('change', convertImage);
    invertCheckbox.addEventListener('change', convertImage);
    colorSelect.addEventListener('change', convertImage);

    copyBtn.addEventListener('click', () => {
        htmlToCopy = asciiPreview.innerHTML;
        navigator.clipboard.writeText(htmlToCopy)
            .then(() => {
                alert('ASCII text copied to clipboard');
            })
            .catch(err => {
                console.error('Failed to copy:', err);
                alert('Failed to copy ASCII text');
            });
    });


    downloadBtn.addEventListener('click', () => {
        htmlToDownload = asciiPreview.innerHTML;
        if (htmlToDownload === '') {
            alert('Nothing to download');
            return;
        }
        const blob = new Blob([asciiPreview.innerHTML], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'ascii.txt';
        a.click();
        URL.revokeObjectURL(url);
    }
    );
    
});
