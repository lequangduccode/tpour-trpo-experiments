"""Export the editable report/deck via installed Microsoft Office, with hidden app windows."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"


def main():
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize()
    word = None
    powerpoint = None
    result = {}
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        for name in ("Bao_cao_chi_tiet_TPOUR_TRPO", "Kich_ban_thuyet_trinh_TPOUR_TRPO"):
            doc = word.Documents.Open(str(OUT / (name + ".docx")), ReadOnly=True, AddToRecentFiles=False)
            try:
                doc.Repaginate()
                result[name + "_pages"] = doc.ComputeStatistics(2)
                doc.ExportAsFixedFormat(str(OUT / (name + ".pdf")), 17)
                if name.startswith("Bao_cao"):
                    doc.SaveAs2(str(OUT / (name + ".doc")), FileFormat=0, AddToRecentFiles=False)
            finally:
                doc.Close(SaveChanges=0)
        word.Quit()
        word = None
        powerpoint = win32com.client.DispatchEx("PowerPoint.Application")
        powerpoint.DisplayAlerts = 1
        presentation = powerpoint.Presentations.Open(str(OUT / "Slide_thuyet_trinh_TPOUR_TRPO.pptx"),
                                                     ReadOnly=True, Untitled=False, WithWindow=False)
        try:
            result["slides"] = presentation.Slides.Count
            overflow = []
            for slide in presentation.Slides:
                for shape in slide.Shapes:
                    if shape.HasTextFrame and shape.TextFrame.HasText:
                        text = shape.TextFrame.TextRange
                        if text.BoundTop + text.BoundHeight > shape.Top + shape.Height + 2:
                            overflow.append({"slide": slide.SlideIndex, "text": text.Text[:100]})
            result["text_overflow"] = overflow
            presentation.SaveAs(str(OUT / "Slide_thuyet_trinh_TPOUR_TRPO.pdf"), 32)
            previews = OUT / "previews"
            previews.mkdir(exist_ok=True)
            presentation.Export(str(previews), "PNG", 1600, 900)
        finally:
            presentation.Close()
        powerpoint.Quit()
        powerpoint = None
        (OUT / "evidence/office_export.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        contact_sheets()
        print(result)
    finally:
        if word is not None:
            word.Quit()
        if powerpoint is not None:
            powerpoint.Quit()
        pythoncom.CoUninitialize()


def contact_sheets():
    import fitz
    from PIL import Image, ImageDraw
    preview = OUT / "previews"
    report = fitz.open(OUT / "Bao_cao_chi_tiet_TPOUR_TRPO.pdf")
    for i, page in enumerate(report, 1):
        page.get_pixmap(matrix=fitz.Matrix(.6, .6)).save(str(preview / f"report_page_{i:02d}.png"))
    for name, files, columns, tile in (
        ("report_contact_sheet.png", [preview / f"report_page_{i:02d}.png" for i in range(1, len(report)+1)], 4, (330, 485)),
        ("slides_contact_sheet.png", sorted([p for p in preview.glob("Slide*.PNG") if p.stem[5:].isdigit()], key=lambda p: int(p.stem[5:])), 3, (480, 295))
    ):
        canvas = Image.new("RGB", (columns*tile[0], ((len(files)+columns-1)//columns)*tile[1]), "#dce4ef")
        draw = ImageDraw.Draw(canvas)
        for i, path in enumerate(files):
            image = Image.open(path).convert("RGB")
            image.thumbnail((tile[0]-20, tile[1]-36))
            x, y = (i % columns)*tile[0], (i // columns)*tile[1]
            canvas.paste(image, (x+10, y+28))
            draw.text((x+12, y+7), f"{i+1:02d}", fill="black")
        canvas.save(preview / name)


if __name__ == "__main__":
    main()
