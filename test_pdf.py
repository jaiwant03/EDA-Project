import os
import subprocess
import pypdf

img_path = os.path.abspath('outputs/plots/01_timeseries_target_pollution.png').replace('\\', '/')
html_content = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{ font-family: sans-serif; }}
img {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
<h1>Test Title Page</h1>
<p>Testing image rendering below:</p>
<img src="file:///{img_path}" />
<p>End of test page.</p>
</body>
</html>'''

with open('test_img.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

test_html_url = 'file:///' + os.path.abspath('test_img.html').replace('\\', '/')
test_pdf = os.path.abspath('test_img.pdf')

cmd = [
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    '--headless',
    '--disable-gpu',
    '--no-sandbox',
    '--allow-file-access-from-files',
    f'--print-to-pdf={test_pdf}',
    '--no-pdf-header-footer',
    test_html_url
]
res = subprocess.run(cmd, capture_output=True, text=True)

print('Return code:', res.returncode)
print('PDF exists:', os.path.exists(test_pdf), 'size:', os.path.getsize(test_pdf) if os.path.exists(test_pdf) else 0)

if os.path.exists(test_pdf):
    reader = pypdf.PdfReader(test_pdf)
    print('Pages:', len(reader.pages))
    for i, p in enumerate(reader.pages):
        print(f'Page {i+1} text:', repr(p.extract_text()))
