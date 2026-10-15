"""Render the saved deck with installed PowerPoint and check text bounds."""
import json
from pathlib import Path
import pythoncom
import win32com.client

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/presentation_final'
pythoncom.CoInitialize()
app=None;deck=None
try:
    app=win32com.client.DispatchEx('PowerPoint.Application')
    app.DisplayAlerts=1
    deck=app.Presentations.Open(str(OUT/'Slide_thuyet_trinh_TPOUR_TRPO_cap_nhat.pptx'),ReadOnly=True,Untitled=False,WithWindow=False)
    overflow=[]
    for slide in deck.Slides:
        for shape in slide.Shapes:
            if shape.HasTextFrame and shape.TextFrame.HasText:
                t=shape.TextFrame.TextRange
                if t.BoundTop+t.BoundHeight>shape.Top+shape.Height+3 or t.BoundLeft+t.BoundWidth>shape.Left+shape.Width+3:
                    overflow.append({'slide':slide.SlideIndex,'text':t.Text[:100]})
    deck.SaveAs(str(OUT/'Slide_thuyet_trinh_TPOUR_TRPO_cap_nhat.pdf'),32)
    result={'slides':deck.Slides.Count,'text_overflow':overflow,'renderer':'Microsoft PowerPoint'}
    (OUT/'render_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
finally:
    if deck is not None:deck.Close()
    if app is not None:app.Quit()
    pythoncom.CoUninitialize()
