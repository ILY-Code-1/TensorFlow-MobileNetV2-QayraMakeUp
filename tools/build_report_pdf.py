"""
build_report_pdf.py — Konversi docs/LAPORAN_PROYEK.md -> docs/LAPORAN_PROYEK.pdf.

- Markdown -> HTML (ekstensi tables + fenced_code).
- Gambar confusion_matrix.png di-embed sebagai data-URI base64 (tidak bergantung path saat render).
- HTML -> PDF via xhtml2pdf (pure-python, tanpa dependensi native).
"""

import base64
import re
import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "docs" / "LAPORAN_PROYEK.md"
PDF = ROOT / "docs" / "LAPORAN_PROYEK.pdf"
CM_IMG = ROOT / "confusion_matrix.png"

CSS = """
@page { size: A4; margin: 1.8cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; color: #1a1a1a; line-height: 1.45; }
h1 { font-size: 19pt; color: #0b3d91; border-bottom: 2px solid #0b3d91; padding-bottom: 4px; }
h2 { font-size: 14pt; color: #0b3d91; margin-top: 16px; }
h3 { font-size: 11.5pt; color: #333; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; }
th, td { border: 1px solid #999; padding: 4px 6px; font-size: 9.5pt; text-align: left; }
th { background-color: #e8eef9; }
code { background-color: #f2f2f2; font-family: Courier, monospace; font-size: 9pt; }
img { width: 12cm; }
hr { border: none; border-top: 1px solid #ccc; }
"""


def main():
    if not MD.exists():
        sys.exit(f"ERROR: {MD} tidak ada.")

    text = MD.read_text(encoding="utf-8")
    html_body = markdown.markdown(text, extensions=["tables", "fenced_code"])

    # Embed confusion_matrix.png sebagai data-URI base64.
    if CM_IMG.exists():
        b64 = base64.b64encode(CM_IMG.read_bytes()).decode("ascii")
        data_uri = f"data:image/png;base64,{b64}"
        html_body = re.sub(r'src="[^"]*confusion_matrix\.png"', f'src="{data_uri}"', html_body)
    else:
        print(f"[!] {CM_IMG.name} tidak ditemukan — gambar dilewati.")

    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"

    with open(PDF, "wb") as f:
        result = pisa.CreatePDF(html, dest=f, encoding="utf-8")

    if result.err:
        sys.exit(f"ERROR: gagal membuat PDF ({result.err} error).")
    print(f"PDF berhasil dibuat: {PDF.relative_to(ROOT).as_posix()} ({PDF.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
