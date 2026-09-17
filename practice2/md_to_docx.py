"""Chuyển practice2/report.md sang practice2/report.docx (python-docx, không cần pandoc)."""

import re, os, sys
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_PARAGRAPH_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.dml.color import ColorFormat

HERE = Path(__file__).parent
SRC = HERE / "report.md"
DST = HERE / "report.docx"

# ── Style / màu ────────────────────────────────────────────────────────────
BLUE      = RGBColor(0x1F, 0x49, 0x7D)
LIGHT_BLUE= RGBColor(0x2E, 0x75, 0xB6)
GRAY      = RGBColor(0x59, 0x56, 0x59)
TABLE_HDR_BG = "1F497D"
TABLE_ALT_BG = "D9E2F3"
TABLE_BORDER_COLOR = "B4C6E7"

HEADING_MAP = {
    1: {"size": Pt(18), "color": BLUE, "bold": True},
    2: {"size": Pt(13), "color": LIGHT_BLUE, "bold": True},
    3: {"size": Pt(11), "color": BLUE, "bold": True},
}

def set_cell_shading(cell, color_hex):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), color_hex)
    shd.set(qn("w:val"), "clear")
    cell._tc.get_or_add_tcPr().append(shd)

def set_table_borders(table, color_hex="B4C6E7", size="4"):
    def _set_border(edge):
        tbl = table._tbl
        tblPr = tbl.tblPr if tbl.tblPr is not None else OxmlElement("w:tblPr")
        borders = tblPr.find(qn("w:tblBorders"))
        if borders is None:
            borders = OxmlElement("w:tblBorders"); tblPr.append(borders)
        b = OxmlElement(f"w:{edge}")
        b.set(qn("w:val"), "single"); b.set(qn("w:sz"), size)
        b.set(qn("w:color"), color_hex); b.set(qn("w:space"), "0")
        # thay nếu đã có
        old = borders.find(qn(f"w:{edge}"))
        if old is not None: borders.remove(old)
        borders.append(b)
    for e in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        _set_border(e)

# ── Các helper paragraph ──────────────────────────────────────────────────
def add_heading(doc, text, level):
    p = doc.add_heading(level=min(level, 3))
    run = p.add_run(text)
    cfg = HEADING_MAP.get(level, HEADING_MAP[3])
    run.font.size = cfg["size"]; run.font.color.rgb = cfg["color"]
    run.bold = cfg["bold"]
    if level == 1:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        # đường kẻ dưới tiêu đề lớn
        pPr = p._p.get_or_add_pPr()
        pBdr = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom"); bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "6"); bottom.set(qn("w:color"), "1F497D")
        bottom.set(qn("w:space"), "1"); pBdr.append(bottom); pPr.append(pBdr)
    return p

def add_paragraph_text(doc, text, italic=False, bullet=False, bold=False, style=None):
    p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    if bullet:
        p.style = "List Bullet"
    run = p.add_run(text) if text else p.add_run("")
    run.italic = italic; run.bold = bold
    run.font.size = Pt(10)
    # giữ nguyên khoảng cách gọn
    pf = p.paragraph_format
    pf.space_after = Pt(4)
    if bullet:
        pf.left_indent = Inches(0.25)
    return p

def _apply_inline(run, bold, italic, code):
    if bold: run.bold = True
    if italic: run.italic = True
    if code:
        run.font.name = "Consolas"; run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0xC0, 0x39, 0x2B)

def add_rich_paragraph(doc, raw, bullet=False):
    """Raw có thể chứa **bold**, *italic*, `code`, [link](..). Đơn giản: parse bold/italic/code."""
    p = doc.add_paragraph(style="List Bullet") if bullet else doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    if bullet:
        p.paragraph_format.left_indent = Inches(0.25)
    # tách code trước để không lẫn
    # token pattern: `code` | **bold** | *italic* | [text](url)
    pat = re.compile(r"`([^`]+)`|\*\*([^*]+)\*\*|\*([^*]+)\*|\[([^\]]+)\]\([^\)]+\)")
    last = 0
    for m in pat.finditer(raw):
        if m.start() > last:
            r = p.add_run(raw[last:m.start()]); r.font.size = Pt(10)
        if m.group(1) is not None:  # code
            r = p.add_run(m.group(1)); _apply_inline(r, False, False, True)
        elif m.group(2) is not None:  # bold
            r = p.add_run(m.group(2)); _apply_inline(r, True, False, False); r.font.size = Pt(10)
        elif m.group(3) is not None:  # italic
            r = p.add_run(m.group(3)); _apply_inline(r, False, True, False); r.font.size = Pt(10)
        elif m.group(4) is not None:  # link — chỉ giữ text + gạch chân
            r = p.add_run(m.group(4)); r.underline = True; r.font.color.rgb = LIGHT_BLUE; r.font.size = Pt(10)
        last = m.end()
    if last < len(raw):
        r = p.add_run(raw[last:]); r.font.size = Pt(10)
    # nếu không có token nào
    if not pat.search(raw) and not p.runs:
        r = p.add_run(raw); r.font.size = Pt(10)
    return p

def add_code_block(doc, lines, lang=""):
    # mỗi block là một bảng 1 ô, nền xám nhạt, border xám — tái hiện code block
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl, "D9D9D9", "4")
    cell = tbl.cell(0, 0)
    set_cell_shading(cell, "F2F2F2")
    # đặt độ rộng bảng ~100% trang
    tblPr = tbl._tbl.tblPr
    tblW = OxmlElement("w:tblW"); tblW.set(qn("w:w"), "0"); tblW.set(qn("w:type"), "auto")
    if tblPr is None:
        tblPr = OxmlElement("w:tblPr"); tbl._tbl.insert(0, tblPr)
    # padding trong ô
    tcPr = cell._tc.get_or_add_tcPr()
    mar = OxmlElement("w:tcMar")
    for side in ["top", "left", "bottom", "right"]:
        e = OxmlElement(f"w:{side}"); e.set(qn("w:w"), "80"); e.set(qn("w:type"), "dxa"); mar.append(e)
    tcPr.append(mar)
    text = "\n".join(lines).rstrip()
    # khoảng trống nhỏ trên (lang label)
    if lang:
        pp = cell.paragraphs[0]; rr = pp.add_run(lang); rr.font.size = Pt(7)
        rr.font.color.rgb = GRAY; rr.font.name = "Consolas"
        pp2 = cell.add_paragraph()
    else:
        pp2 = cell.paragraphs[0]
    for i, ln in enumerate(text.split("\n")):
        p = pp2 if i == 0 else cell.add_paragraph()
        r = p.add_run(ln if ln else " ")
        r.font.name = "Consolas"; r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)

def add_md_table(doc, header, rows):
    cols = len(header)
    tbl = doc.add_table(rows=1 + len(rows), cols=cols)
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = True
    set_table_borders(tbl)

    def style_row(cells, is_header=False, alt=False):
        bg = TABLE_HDR_BG if is_header else (TABLE_ALT_BG if alt else "FFFFFF")
        for c in cells:
            set_cell_shading(c, bg)
            for par in c.paragraphs:
                par.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER if is_header else WD_PARAGRAPH_ALIGNMENT.LEFT
                for rr in par.runs:
                    rr.font.size = Pt(8) if not is_header else Pt(8.5)
                    rr.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if is_header else RGBColor(0x33, 0x33, 0x33)
                    rr.bold = is_header
                    if not is_header: rr.font.name = "Calibri"
                    else: rr.font.name = "Calibri"

    # header
    hdr_cells = tbl.rows[0].cells
    for j, h in enumerate(header):
        hdr_cells[j].text = ""
        rr = hdr_cells[j].paragraphs[0].add_run(h)
        rr.bold = True; rr.font.size = Pt(8.5)
    style_row(hdr_cells, is_header=True)

    # data
    for i, row in enumerate(rows):
        cells = tbl.rows[i + 1].cells
        for j, val in enumerate(row):
            cells[j].text = ""
            # xử lý <điền> nổi bật
            if "<điền" in val:
                rr = cells[j].paragraphs[0].add_run(val)
                rr.italic = True; rr.font.color.rgb = RGBColor(0xC0, 0x39, 0x2B)
            else:
                rr = cells[j].paragraphs[0].add_run(val)
            rr.font.size = Pt(8)
        style_row(cells, alt=(i % 2 == 1))

    # small gap sau bảng
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

# ── Parse markdown ────────────────────────────────────────────────────────
def parse_and_build(src_path: Path, doc: Document):
    raw = src_path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    i = 0
    in_code = False
    code_lang = ""
    code_buf = []
    pending_list_indent = 0

    def flush_code():
        nonlocal code_buf, code_lang, in_code
        if code_buf:
            add_code_block(doc, code_buf, code_lang)
            code_buf = []; code_lang = ""; in_code = False

    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # code fence
        if stripped.startswith("```"):
            if not in_code:
                in_code = True
                code_lang = stripped[3:].strip()
                code_buf = []
            else:
                flush_code()
            i += 1; continue
        if in_code:
            code_buf.append(line); i += 1; continue

        # horizontal rule
        if re.match(r"^-{3,}$", stripped) or re.match(r"^\*{3,}$", stripped):
            p = doc.add_paragraph(); pPr = p._p.get_or_add_pPr()
            pBdr = OxmlElement("w:pBdr")
            btm = OxmlElement("w:bottom"); btm.set(qn("w:val"), "single")
            btm.set(qn("w:sz"), "4"); btm.set(qn("w:color"), "B4C6E7")
            btm.set(qn("w:space"), "1"); pBdr.append(btm); pPr.append(pBdr)
            i += 1; continue

        # heading
        m = re.match(r"^(#{1,3})\s+(.*)", stripped)
        if m:
            add_heading(doc, m.group(2).strip(), len(m.group(1)))
            i += 1; continue

        # table (header + separator + rows)
        if "|" in line and i + 1 < n and re.match(r"^\s*\|?[\s\-:|]+\|?\s*$", lines[i+1]):
            # thu thập hết các dòng bảng
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2  # qua separator
            rows = []
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            add_md_table(doc, header, rows)
            continue

        # bullet list
        if re.match(r"^\s*[-*]\s+", line):
            content = re.sub(r"^\s*[-*]\s+", "", line).strip()
            add_rich_paragraph(doc, content, bullet=True)
            i += 1; continue
        if re.match(r"^\s*\d+\.\s+", line):
            content = re.sub(r"^\s*\d+\.\s+", "", line).strip()
            # giữ số thứ tự trong text
            num = re.match(r"^\s*(\d+)\.", line).group(1)
            add_rich_paragraph(doc, f"{num}. {content}", bullet=False)
            i += 1; continue

        # trống — bỏ qua (đã có spacing)
        if not stripped:
            i += 1; continue

        # đoạn thường
        add_rich_paragraph(doc, stripped, bullet=False)
        i += 1

    flush_code()

def build_docx():
    doc = Document()

    # — Page setup A4, margin gọn —
    for sec in doc.sections:
        sec.page_width = Inches(8.27); sec.page_height = Inches(11.69)
        sec.top_margin = Inches(0.6); sec.bottom_margin = Inches(0.6)
        sec.left_margin = Inches(0.75); sec.right_margin = Inches(0.75)
        # header/footer font mặc định
        sec.header.is_linked_to_previous = False
        sec.footer.is_linked_to_previous = False

    # — Default font cho style Normal —
    style = doc.styles["Normal"]
    style.font.name = "Calibri"; style.font.size = Pt(10)
    style.paragraph_format.space_after = Pt(4)
    style.paragraph_format.line_spacing = 1.05

    # — Header của trang —
    hdr = doc.sections[0].header
    hp = hdr.paragraphs[0]; hp.alignment = WD_PARAGRAPH_ALIGNMENT.RIGHT
    rr = hp.add_run("Practice 2 — Pre-trained Neural Networks  •  Deep Learning  •  11/09/2026")
    rr.font.size = Pt(7); rr.font.color.rgb = GRAY; rr.italic = True

    # — Footer: số trang —
    ftp = doc.sections[0].footer.paragraphs[0]; ftp.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    rr = ftp.add_run(); rr.font.size = Pt(7); rr.font.color.rgb = GRAY
    # field PAGE đơn giản: dùng text, Word tự đánh nếu thêm field — ở đây để placeholder
    rr2 = ftp.add_run("Trang "); rr2.font.size = Pt(7); rr2.font.color.rgb = GRAY

    # — Cover (tiêu đề lớn) —
    # đọc tiêu đề từ md để làm cover nổi bật
    # parse xong vẫn in tiếp nội dung — cover là extra
    # nhưng để đơn giản: cứ parse md như thường và heading cấp 1 đầu sẽ là cover

    parse_and_build(SRC, doc)

    # — Lưu —
    doc.save(str(DST))
    print(f"Đã tạo: {DST}  ({DST.stat().st_size/1024:.0f} KB)")

if __name__ == "__main__":
    if not SRC.exists():
        print(f"Không tìm thấy {SRC}", file=sys.stderr); sys.exit(1)
    build_docx()
