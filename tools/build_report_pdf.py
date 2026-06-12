"""
build_report_pdf.py — Konversi docs/LAPORAN_PROYEK.md -> docs/LAPORAN_PROYEK.pdf.

- Markdown -> HTML (ekstensi tables + fenced_code).
- Gambar confusion_matrix.png di-embed base64, terpusat, lebar ~87% area teks, + caption.
- Tabel: colWidths eksplisit proporsional, header tebal, border konsisten, tidak tumpuk.
- Heading tidak yatim (-pdf-keep-with-next), nomor halaman di footer.
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

# A4 = 595x842pt. Frame konten + frame footer (nomor halaman).
CSS = """
@page {
    size: a4 portrait;
    @frame content_frame { left: 50pt; width: 495pt; top: 45pt; height: 742pt; }
    @frame footer_frame {
        -pdf-frame-content: footerContent;
        left: 50pt; width: 495pt; top: 795pt; height: 28pt;
    }
}
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; color: #1a1a1a; line-height: 1.45; }
h1 { font-size: 19pt; color: #0b3d91; border-bottom: 2px solid #0b3d91; padding-bottom: 4px;
     -pdf-keep-with-next: true; }
h2 { font-size: 14pt; color: #0b3d91; margin-top: 16px; -pdf-keep-with-next: true; }
h3 { font-size: 11.5pt; color: #333; -pdf-keep-with-next: true; }
p { -pdf-keep-with-next: false; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; table-layout: fixed; }
th, td { border: 1px solid #888; padding: 4px 5px; font-size: 9.5pt; word-wrap: break-word; }
th { background-color: #e8eef9; text-align: center; font-weight: bold; }
td { text-align: center; }
td.kelas, th.kelas { text-align: left; }
code { font-family: Courier, monospace; font-size: 9pt; }
pre { background-color: #f4f4f4; border: 1px solid #ddd; padding: 6px; font-size: 8.5pt;
      font-family: Courier, monospace; }
hr { border: none; border-top: 1px solid #ccc; }
.figure { text-align: center; margin: 12px 0; }
.caption { font-size: 9pt; color: #555; font-style: italic; margin-top: 5px; }
#footerContent { text-align: center; font-size: 8pt; color: #999; }
"""

# Lebar kolom eksplisit per-tabel (dipilih dari header baris pertama).
COL_PROFILES = {
    ("Kelas", "Precision", "Recall", "F1-score", "Support"):
        ["24%", "20%", "18%", "20%", "18%"],
    ("Kelas", "Sumber", "Keterangan"):
        ["20%", "32%", "48%"],
    ("Kelas", "Train", "Val", "Test", "Total"):
        ["28%", "18%", "18%", "18%", "18%"],
    ("Fase", "Train acc", "Val acc"):
        ["48%", "26%", "26%"],
    ("Versi", "Dataset", "Akurasi", "Catatan singkat"):
        ["12%", "34%", "16%", "38%"],
}


def _strip_tags(s):
    return re.sub(r"<[^>]+>", "", s).strip()


def add_colgroups(html):
    """Sisipkan <colgroup> lebar eksplisit ke tiap <table> berdasarkan header-nya;
    fallback lebar sama rata bila header tak dikenal. Kolom pertama -> rata kiri."""
    def repl(m):
        table = m.group(0)
        headers = tuple(_strip_tags(h) for h in re.findall(r"<th[^>]*>(.*?)</th>", table, re.S))
        widths = COL_PROFILES.get(headers)
        n = len(headers) if headers else 0
        if widths is None and n:
            widths = [f"{100.0 / n:.4f}%"] * n
        if widths:
            cols = "".join(f'<col width="{w}" />' for w in widths)
            table = re.sub(r"(<table[^>]*>)", r"\1" + f"<colgroup>{cols}</colgroup>", table, count=1)
            table = re.sub(r"<th>", '<th class="kelas">', table, count=1)
            table = re.sub(r"<tr>\s*<td>", '<tr><td class="kelas">', table)
        return table

    return re.sub(r"<table>.*?</table>", repl, html, flags=re.S)


def embed_figure(html):
    """Ganti <p><img ...confusion_matrix.png...></p> jadi figure terpusat + caption + base64."""
    if not CM_IMG.exists():
        print(f"[!] {CM_IMG.name} tidak ditemukan — gambar dilewati.")
        return html
    b64 = base64.b64encode(CM_IMG.read_bytes()).decode("ascii")
    data_uri = f"data:image/png;base64,{b64}"
    figure = (
        '<div class="figure">'
        f'<img src="{data_uri}" style="width: 430pt;" />'
        '<div class="caption">Gambar 1. Confusion matrix model final (V4) — '
        'akurasi 58,65% (baris = label asli, kolom = prediksi).</div>'
        '</div>'
    )
    # Hapus <p> pembungkus img bila ada, ganti seluruhnya dengan figure.
    html = re.sub(r'<p>\s*<img[^>]*confusion_matrix\.png[^>]*>\s*</p>', figure, html)
    html = re.sub(r'<img[^>]*confusion_matrix\.png[^>]*>', figure, html)  # jaga-jaga
    return html


def main():
    if not MD.exists():
        sys.exit(f"ERROR: {MD} tidak ada.")

    text = MD.read_text(encoding="utf-8")
    html_body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    html_body = add_colgroups(html_body)
    html_body = embed_figure(html_body)

    footer = '<div id="footerContent">Halaman <pdf:pagenumber> &nbsp;|&nbsp; QayraMakeUp — Laporan Proyek</div>'
    html = (f"<html><head><meta charset='utf-8'><style>{CSS}</style></head>"
            f"<body>{footer}{html_body}</body></html>")

    with open(PDF, "wb") as f:
        result = pisa.CreatePDF(html, dest=f, encoding="utf-8")

    if result.err:
        sys.exit(f"ERROR: gagal membuat PDF ({result.err} error).")
    print(f"PDF berhasil dibuat: {PDF.relative_to(ROOT).as_posix()} ({PDF.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
