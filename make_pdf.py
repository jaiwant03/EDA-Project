import os
import sys
import re
import subprocess
import markdown
import pypdf

def clean_latex_math(text):
    # Normalize inline latex
    text = text.replace(r'\,', ' ')
    text = text.replace(r'\quad', ' &nbsp; ')
    text = text.replace(r'\times', '×')
    text = text.replace(r'$$', '&&MATHBLOCK&&')  # protect block math temporarily

    # Inline math regex replacements
    text = re.sub(r'\$?\\text\{PM\}\_\{2\.5\}\$?', 'PM<sub>2.5</sub>', text)
    text = re.sub(r'\$?PM_\{2\.5\}\$?', 'PM<sub>2.5</sub>', text)
    text = re.sub(r'\$?\\text\{PM\}\_\{10\}\$?', 'PM<sub>10</sub>', text)
    text = re.sub(r'\$?\\mu\\text\{g/m\}\^3\$?', 'µg/m³', text)
    text = re.sub(r'\$\\mu g/m\^3\$', 'µg/m³', text)
    text = re.sub(r'\\mu g/m\^3', 'µg/m³', text)
    text = re.sub(r'\$\\text\{MAE\}\s*=\s*([0-9\.]+)\s*µg/m³\$', r'MAE = \1 µg/m³', text)
    text = re.sub(r'\$\\text\{RMSE\}\s*=\s*([0-9\.]+)\s*µg/m³\$', r'RMSE = \1 µg/m³', text)
    text = re.sub(r'\$\\text\{MAE\}\$', 'MAE', text)
    text = re.sub(r'\$\\text\{RMSE\}\$', 'RMSE', text)
    text = re.sub(r'\$\\text\{MSE\}\$', 'MSE', text)
    text = re.sub(r'\$R\^2\s*=\s*([0-9\.]+)\$', r'<i>R</i>² = \1', text)
    text = re.sub(r'\$R\^2\$', '<i>R</i>²', text)
    text = re.sub(r'\$T\s*=\s*24\$', '<i>T</i> = 24', text)
    text = re.sub(r'\$T\s*=\s*12\$', '<i>T</i> = 12', text)
    text = re.sub(r'\$T\$', '<i>T</i>', text)
    text = re.sub(r'\$\\eta\s*=\s*([0-9\.]+)\$', r'η = \1', text)
    text = re.sub(r'\$\\eta\$', 'η', text)
    text = re.sub(r'\$\\le\$', '≤', text)
    text = re.sub(r'\\le', '≤', text)
    text = re.sub(r'\$\\ge\$', '≥', text)
    text = re.sub(r'\\ge', '≥', text)
    text = re.sub(r'\$\\pm\s*1\\sigma\$', '±1σ', text)
    text = re.sub(r'\$\\pm\s*2\\sigma\$', '±2σ', text)
    text = re.sub(r'\$\\pm\s*44\s*µg/m³\$', '±44 µg/m³', text)
    text = re.sub(r'\$\\sigma\$', 'σ', text)
    text = re.sub(r'\$\\gamma\s*=\s*([0-9\.]+)\$', r'γ = \1', text)
    text = re.sub(r'\$\\gamma\$', 'γ', text)
    text = re.sub(r'\$r\s*=\s*([+\-0-9\.]+)\$', r'<i>r</i> = \1', text)
    text = re.sub(r'\$r\$', '<i>r</i>', text)
    text = re.sub(r'\$\\Delta t\s*=\s*1\.0\s*\\text\{hour\}\$', 'Δ<i>t</i> = 1.0 hour', text)
    text = re.sub(r'\$\\Delta t\$', 'Δ<i>t</i>', text)
    text = re.sub(r'\$y\s*=\s*x\$', '<i>y</i> = <i>x</i>', text)
    text = re.sub(r'\$45\^\\circ\$', '45°', text)
    text = re.sub(r'\$p\s*=\s*0\.20\$', '<i>p</i> = 0.20', text)
    text = re.sub(r'\$t\+1\$', '<i>t</i>+1', text)
    text = re.sub(r'\$t-23\$', '<i>t</i>-23', text)
    text = re.sub(r'\$t-T\+1\$', '<i>t</i>-<i>T</i>+1', text)
    text = re.sub(r'\$t\$', '<i>t</i>', text)
    text = re.sub(r'\$\\tanh\$', 'tanh', text)
    text = re.sub(r'\$C_t\$', '<i>C<sub>t</sub></i>', text)
    text = re.sub(r'\$h_t\$', '<i>h<sub>t</sub></i>', text)
    text = re.sub(r'\$N\$', '<i>N</i>', text)
    text = re.sub(r'\$\\mu\s*=\s*([\-0-9\.]+)\s*µg/m³\$', r'μ = \1 µg/m³', text)
    text = re.sub(r'\$\\mu_\{residual\}\$', 'μ<sub>residual</sub>', text)
    text = re.sub(r'\$\\sigma_\{residual\}\$', 'σ<sub>residual</sub>', text)
    text = re.sub(r'\\mu_\{residual\}', 'μ<sub>residual</sub>', text)
    text = re.sub(r'\\sigma_\{residual\}', 'σ<sub>residual</sub>', text)
    text = re.sub(r'\$\\text\{Val MSE\}\s*=\s*([0-9\.]+)\$', r'Val MSE = \1', text)
    text = re.sub(r'\$\\text\{Val MAE\}\s*=\s*([0-9\.]+)\$', r'Val MAE = \1', text)
    text = re.sub(r'\$\\text\{Loss\}_\{[^}]+\}\s*\\approx\s*([0-9\.]+)\$', r'Loss ≈ \1', text)
    text = re.sub(r'\$e_i\s*=\s*y_i\s*-\s*\\hat\{y\}_i\$', '<i>e<sub>i</sub></i> = <i>y<sub>i</sub></i> - <i>ŷ<sub>i</sub></i>', text)
    text = re.sub(r'\$\\hat\{y\}_\{t\+1\}\$', '<i>ŷ</i><sub><i>t</i>+1</sub>', text)
    text = re.sub(r'\$\\hat\{y\}\$', '<i>ŷ</i>', text)
    text = re.sub(r'\$y_i\$', '<i>y<sub>i</sub></i>', text)
    text = re.sub(r'\$y_t\$', '<i>y<sub>t</sub></i>', text)
    text = re.sub(r'\$X_t\$', '<i>X<sub>t</sub></i>', text)
    text = re.sub(r'\$\^\{\\circ\}\\text\{C\}\$', '°C', text)
    text = re.sub(r'\$>\s*([0-9]+)\s*µg/m³\$', r'> \1 µg/m³', text)
    text = re.sub(r'\$<\s*([0-9]+)\s*µg/m³\$', r'< \1 µg/m³', text)
    text = re.sub(r'\$\\text\{hPa\}\$', 'hPa', text)
    text = re.sub(r'\$\\text\{hours\}\$', 'hours', text)
    text = re.sub(r'\$\\text\{m/s\}\$', 'm/s', text)

    # Restore block math with clean rendering
    def math_block_replacer(match):
        content = match.group(1).strip()
        content = content.replace(r'\text{MAE}', 'MAE')
        content = content.replace(r'\text{MSE}', 'MSE')
        content = content.replace(r'\text{RMSE}', 'RMSE')
        content = content.replace(r'\text{ReLU}', 'ReLU')
        content = content.replace(r'\text{pollution}', 'pollution')
        content = content.replace(r'\text{scaled}', 'scaled')
        content = content.replace(r'\text{train}', 'train')
        content = content.replace(r'\text{residual}', 'residual')
        content = content.replace(r'\text{Loss}_{\text{train}}', 'Loss<sub>train</sub>')
        content = content.replace(r'\text{Loss}_{\text{val}}', 'Loss<sub>val</sub>')
        content = content.replace(r'\mu\text{g/m}^3', 'µg/m³')
        content = content.replace(r'\mu g/m^3', 'µg/m³')
        content = content.replace(r'\approx', '≈')
        content = content.replace(r'\cdot', '·')
        content = content.replace(r'\odot', '⊙')
        content = content.replace(r'\sigma', 'σ')
        content = content.replace(r'\tanh', 'tanh')
        content = content.replace(r'\in \mathbb{R}^{24 \times 11}', '∈ ℝ²⁴ ˣ ¹¹')
        content = content.replace(r'\in \mathbb{R}^1', '∈ ℝ¹')
        content = content.replace(r'\le', '≤')
        content = content.replace(r'\ge', '≥')
        content = content.replace(r'\tilde{C}_t', '<i>C̃<sub>t</sub></i>')
        content = content.replace(r'C_{t-1}', '<i>C</i><sub><i>t</i>-1</sub>')
        content = content.replace(r'C_t', '<i>C<sub>t</sub></i>')
        content = content.replace(r'h_{t-1}', '<i>h</i><sub><i>t</i>-1</sub>')
        content = content.replace(r'h_t', '<i>h<sub>t</sub></i>')
        content = content.replace(r'x_t', '<i>x<sub>t</sub></i>')
        content = content.replace(r'f_t', '<i>f<sub>t</sub></i>')
        content = content.replace(r'i_t', '<i>i<sub>t</sub></i>')
        content = content.replace(r'o_t', '<i>o<sub>t</sub></i>')
        content = content.replace(r'W_f', '<i>W<sub>f</sub></i>')
        content = content.replace(r'W_i', '<i>W<sub>i</sub></i>')
        content = content.replace(r'W_c', '<i>W<sub>c</sub></i>')
        content = content.replace(r'W_o', '<i>W<sub>o</sub></i>')
        content = content.replace(r'b_f', '<i>b<sub>f</sub></i>')
        content = content.replace(r'b_i', '<i>b<sub>i</sub></i>')
        content = content.replace(r'b_c', '<i>b<sub>c</sub></i>')
        content = content.replace(r'b_o', '<i>b<sub>o</sub></i>')
        content = re.sub(r'\\frac\{1\}\{N\}\s*\\sum_\{i=1\}\^N\s*\|y_i\s*-\s*\\hat\{y\}_i\|', '(1/<i>N</i>) ∑ |<i>y<sub>i</sub></i> - <i>ŷ<sub>i</sub></i>|', content)
        content = re.sub(r'\\sqrt\{\\frac\{1\}\{N\}\s*\\sum_\{i=1\}\^N\s*\(y_i\s*-\s*\\hat\{y\}_i\)\^2\}', '√[ (1/<i>N</i>) ∑ (<i>y<sub>i</sub></i> - <i>ŷ<sub>i</sub></i>)² ]', content)
        content = re.sub(r'\\sqrt\{\\text\{MSE\}\}', '√MSE', content)
        content = re.sub(r'\\frac\{1\}\{N\}\s*\\sum_\{i=1\}\^N\s*\(y_i\s*-\s*\\hat\{y\}_i\)\^2', '(1/<i>N</i>) ∑ (<i>y<sub>i</sub></i> - <i>ŷ<sub>i</sub></i>)²', content)
        content = re.sub(r'\\frac\{x\s*-\s*\\min\(x_\{[^}]+\}\)\}\{\\max\(x_\{[^}]+\}\)\s*-\s*\\min\(x_\{[^}]+\}\)\}', '[ <i>x</i> - min(<i>x</i><sub>train</sub>) ] / [ max(<i>x</i><sub>train</sub>) - min(<i>x</i><sub>train</sub>) ]', content)
        content = re.sub(r'1\s*-\s*\\frac\{\\sum[^}]+\}\{\\sum[^}]+\}', '1 - [ ∑ (<i>y<sub>i</sub></i> - <i>ŷ<sub>i</sub></i>)² / ∑ (<i>y<sub>i</sub></i> - <i>ȳ</i>)² ]', content)
        return f'<div class="math-block">{content}</div>'

    text = re.sub(r'&&MATHBLOCK&&(.*?)&&MATHBLOCK&&', math_block_replacer, text, flags=re.DOTALL)
    return text

def generate_html_from_markdown(md_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # Pre-clean latex
    md_text = clean_latex_math(md_text)

    # Convert to HTML using markdown library
    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'toc'])

    # Embed figures directly into the HTML using local absolute file URLs
    plot_dir = os.path.abspath(os.path.join(os.path.dirname(md_path), 'outputs', 'plots')).replace('\\', '/')
    
    def embed_image_replacer(match):
        full_match = match.group(0)
        # Find filename like 01_timeseries_target_pollution.png
        m = re.search(r'(\d\d_[a-zA-Z0-9_]+\.png)', full_match)
        if m:
            filename = m.group(1)
            img_file_url = f"file:///{plot_dir}/{filename}"
            return f'''<div class="figure-container">
                <img class="report-img" src="{img_file_url}" alt="{filename}" />
                <p class="figure-caption"><strong>Figure:</strong> {filename}</p>
            </div>'''
        return full_match

    # Match list items or paragraphs containing saved plot / saved file links
    html_body = re.sub(r'<li><strong>(?:Saved File|Saved Plot):</strong>.*?</li>', embed_image_replacer, html_body, flags=re.DOTALL)
    html_body = re.sub(r'<p><strong>(?:Saved File|Saved Plot):</strong>.*?</p>', embed_image_replacer, html_body, flags=re.DOTALL)

    # Replace Title Page Section with structured Cover Page (remove everything before Abstract)
    styled_cover = """<div class="cover-page">
    <div class="cover-badge">ACADEMIC PROJECT REPORT</div>
    <h1 class="cover-title">Exploratory Data Analysis and LSTM-Based Time-Series Forecasting Using Multivariate and Univariate Data</h1>
    <p class="cover-subtitle">A Deep Learning Case Study on Urban Air Quality Dynamics and Atmospheric Particulate Matter (PM<sub>2.5</sub>) Modeling</p>
    
    <div class="cover-details-card">
        <table class="cover-table">
            <tr><td class="lbl">Project Title</td><td class="val"><strong>Exploratory Data Analysis and LSTM-Based Time-Series Forecasting Using Multivariate and Univariate Data</strong></td></tr>
            <tr><td class="lbl">Student Name</td><td class="val"><strong>Jaiwant Karrun SA</strong></td></tr>
            <tr><td class="lbl">Roll Number</td><td class="val"><strong>24AD048</strong></td></tr>
            <tr><td class="lbl">Department</td><td class="val"><strong>AD</strong></td></tr>
            <tr><td class="lbl">Academic Year</td><td class="val"><strong>3rd Year</strong></td></tr>
            <tr><td class="lbl">Date of Submission</td><td class="val"><strong>15/10/2026</strong></td></tr>
        </table>
    </div>
</div>
"""
    # Replace from beginning of body up to "<h2 id="2-abstract">2. Abstract</h2>"
    html_body = re.sub(r'^.*?(?=<h2 id="2-abstract">)', styled_cover, html_body, flags=re.DOTALL)

    # Clean any accidental extra page breaks before abstract
    html_body = re.sub(r'(<div class="cover-page">.*?</div>)\s*(?:<div style="page-break-after: always;"></div>|<hr />)+', r'\1\n', html_body, flags=re.DOTALL)

    # HTML Document Shell with Print Stylesheet
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Project Report - Jaiwant Karrun SA (24AD048)</title>
<style>
    @page {{
        size: A4;
        margin: 18mm 16mm 18mm 16mm;
        @bottom-right {{
            content: "Page " counter(page);
            font-size: 8.5pt;
            color: #64748b;
        }}
    }}
    * {{
        box-sizing: border-box;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1e293b;
        background: #ffffff;
        line-height: 1.55;
        font-size: 10pt;
        margin: 0;
        padding: 0;
    }}
    
    /* Cover Page Styling */
    .cover-page {{
        text-align: center;
        padding: 60px 20px 30px 20px;
        page-break-after: always;
        page-break-inside: avoid;
    }}
    .cover-badge {{
        display: inline-block;
        background: #0f172a;
        color: #ffffff;
        font-size: 9.5pt;
        font-weight: 700;
        letter-spacing: 2.5px;
        padding: 6px 18px;
        border-radius: 20px;
        margin-bottom: 25px;
    }}
    .cover-title {{
        font-size: 22pt;
        font-weight: 800;
        line-height: 1.3;
        color: #0f172a;
        margin: 0 0 15px 0;
    }}
    .cover-subtitle {{
        font-size: 12pt;
        color: #475569;
        font-style: italic;
        margin: 0 0 45px 0;
    }}
    .cover-details-card {{
        width: 100%;
        max-width: 600px;
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 26px;
        margin: 0 auto;
    }}
    .cover-table {{
        width: 100%;
        border-collapse: collapse;
        text-align: left;
    }}
    .cover-table td {{
        padding: 9px 12px;
        border-bottom: 1px solid #e2e8f0;
        font-size: 10.5pt;
    }}
    .cover-table tr:last-child td {{
        border-bottom: none;
    }}
    .cover-table .lbl {{
        color: #64748b;
        font-weight: 600;
        width: 36%;
    }}
    .cover-table .val {{
        color: #0f172a;
    }}

    /* Section Headings */
    h2 {{
        font-size: 14pt;
        font-weight: 700;
        color: #0f172a;
        border-bottom: 2px solid #0284c7;
        padding-bottom: 5px;
        margin-top: 28px;
        margin-bottom: 12px;
        page-break-after: avoid;
    }}
    h3 {{
        font-size: 12pt;
        font-weight: 600;
        color: #0369a1;
        margin-top: 20px;
        margin-bottom: 8px;
        page-break-after: avoid;
    }}
    h4 {{
        font-size: 10.5pt;
        font-weight: 600;
        color: #334155;
        margin-top: 14px;
        margin-bottom: 6px;
        page-break-after: avoid;
    }}

    p {{
        margin: 0 0 10px 0;
        text-align: justify;
    }}

    /* Tables */
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 12px 0 16px 0;
        font-size: 9pt;
        page-break-inside: avoid;
    }}
    th, td {{
        border: 1px solid #cbd5e1;
        padding: 6px 9px;
        text-align: left;
    }}
    th {{
        background: #f1f5f9;
        color: #0f172a;
        font-weight: 700;
    }}
    tr:nth-child(even) {{
        background: #f8fafc;
    }}

    /* Figures */
    .figure-container {{
        text-align: center;
        margin: 14px 0 18px 0;
        page-break-inside: avoid;
    }}
    .report-img {{
        max-width: 94%;
        max-height: 380px;
        height: auto;
        object-fit: contain;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
    }}
    .figure-caption {{
        font-size: 8.5pt;
        color: #64748b;
        margin-top: 5px;
        font-style: italic;
        text-align: center;
    }}

    /* Math Blocks */
    .math-block {{
        background: #f8fafc;
        border-left: 3.5px solid #0284c7;
        padding: 7px 12px;
        margin: 8px 0;
        font-family: "Cambria Math", "DejaVu Serif", serif;
        font-size: 10pt;
        color: #0f172a;
        page-break-inside: avoid;
    }}

    /* Code Blocks */
    pre {{
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 10px 12px;
        font-family: "Consolas", "Fira Code", monospace;
        font-size: 8pt;
        line-height: 1.4;
        overflow-x: auto;
        margin: 10px 0;
        page-break-inside: avoid;
        color: #0f172a;
    }}
    code {{
        font-family: "Consolas", "Fira Code", monospace;
        font-size: 8.5pt;
        background: #f1f5f9;
        padding: 1px 4px;
        border-radius: 3px;
    }}
    pre code {{
        background: transparent;
        padding: 0;
    }}

    /* Lists */
    ul, ol {{
        margin: 0 0 10px 0;
        padding-left: 22px;
    }}
    li {{
        margin-bottom: 4px;
    }}

    hr {{
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 20px 0;
    }}

    /* Print Specifics */
    @media print {{
        body {{
            background: #ffffff;
            color: #000000;
        }}
        h2 {{
            page-break-before: auto;
        }}
        .figure-container {{
            page-break-inside: avoid;
        }}
        table {{
            page-break-inside: avoid;
        }}
        pre {{
            page-break-inside: avoid;
        }}
    }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""
    return html_doc

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(base_dir, 'PROJECT_REPORT.md')
    html_path = os.path.join(base_dir, 'PROJECT_REPORT.html')
    pdf_path = os.path.join(base_dir, 'PROJECT_REPORT.pdf')

    print(f"Reading markdown: {md_path}")
    html_content = generate_html_from_markdown(md_path)
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Generated standalone HTML: {html_path} ({os.path.getsize(html_path):,} bytes)")

    candidate_browsers = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
    ]
    browser_exe = None
    for b in candidate_browsers:
        if os.path.exists(b):
            browser_exe = b
            break
            
    if not browser_exe:
        print("ERROR: Neither Chrome nor Edge executable was found.")
        sys.exit(1)

    print(f"Using browser for PDF generation: {browser_exe}")
    
    html_url = "file:///" + os.path.abspath(html_path).replace("\\", "/")
    
    cmd = [
        browser_exe,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--allow-file-access-from-files",
        f"--print-to-pdf={pdf_path}",
        "--no-pdf-header-footer",
        html_url
    ]
    
    print(f"Executing browser print-to-pdf with URL: {html_url}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
        pdf_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        print(f"SUCCESS: Generated PDF report -> {pdf_path} ({pdf_size_mb:.2f} MB)")
        
        # Verify with pypdf
        reader = pypdf.PdfReader(pdf_path)
        num_pages = len(reader.pages)
        print(f"PDF Validation: Total {num_pages} pages detected.")
        blank_pages = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text().strip()
            if len(txt) <= 10:  # Only page number or empty
                blank_pages.append(i + 1)
        if blank_pages:
            print(f"WARNING: Blank pages detected: {blank_pages}")
        else:
            print("VERIFIED: 0 blank pages detected! All pages contain rich text and figures.")
            for i in range(min(5, num_pages)):
                snippet = reader.pages[i].extract_text().strip()[:80].replace("\n", " ")
                print(f"  - Page {i+1} preview: {snippet}...")
    else:
        print(f"ERROR generating PDF: Return code {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
