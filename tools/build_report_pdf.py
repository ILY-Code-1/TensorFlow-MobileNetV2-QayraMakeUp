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
table { border-collapse: collapse; width: 100%; margin: 8px 0; table-layout: fixed; }
th, td { border: 1px solid #999; padding: 4px 5px; font-size: 9.5pt; word-wrap: break-word; }
th { background-color: #e8eef9; text-align: center; }
td { text-align: center; }
td.kelas, th.kelas { text-align: left; }
code { background-color: #f2f2f2; font-family: Courier, monospace; font-size: 9pt; }
img { width: 12cm; }
hr { border: none; border-top: 1px solid #ccc; }
"""

# Lebar kolom eksplisit per-tabel (dipilih dari header baris pertama).
# table-layout:fixed + colgroup -> xhtml2pdf menghormati lebar ini dan membungkus header.
COL_PROFILES = {
    ("Kelas", "Precision", "Recall", "F1-score", "Support"):
        ["24%", "20%", "18%", "20%", "18%"],
    ("Kelas", "Sumber", "Keterangan"):
        ["20%", "32%", "48%"],
    ("Kelas", "Train", "Val", "Test", "Total"):
        ["28%", "18%", "18%", "18%", "18%"],
    ("Fase", "Train acc", "Val acc"):
        ["48%", "26%", "26%"],
}


def _strip_tags(s):
    return re.sub(r"<[^>]+>", "", s).strip()


def add_colgroups(html):
    """Sisipkan <colgroup> lebar eksplisit ke tiap <table> berdasarkan header-nya;
    fallback ke lebar sama rata jika header tak dikenal. Juga tandai kolom pertama
    sebagai .kelas agar rata kiri."""
    def repl(m):
        table = m.group(0)
        headers = tuple(_strip_tags(h) for h in re.findall(r"<th[^>]*>(.*?)</th>", table, re.S))
        widths = COL_PROFILES.get(headers)
        n = len(headers) if headers else 0
        if widths is None and n:
            widths = [f"{100.0 / n:.4f}%"] * n
        if widths:
            cols = "".join(f'<col width="{w}" />' for w in widths)
            colgroup = f"<colgroup>{cols}</colgroup>"
            table = re.sub(r"(<table[^>]*>)", r"\1" + colgroup, table, count=1)
            # kolom pertama (Kelas/Fase) -> rata kiri
            table = re.sub(r"<th>", '<th class="kelas">', table, count=1)
            table = table.replace("<tr>\n<td>", '<tr>\n<td class="kelas">')
            table = re.sub(r"<tr>\s*<td>", '<tr><td class="kelas">', table)
        return table

    return re.sub(r"<table>.*?</table>", repl, html, flags=re.S)


def main():
    if not MD.exists():
        sys.exit(f"ERROR: {MD} tidak ada.")

    text = MD.read_text(encoding="utf-8")
    html_body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    html_body = add_colgroups(html_body)

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
