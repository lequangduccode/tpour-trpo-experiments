"""Update only local TOC fields and render the rebuilt PTIT document in Word."""
import hashlib
import json
from pathlib import Path
import pythoncom
import win32com.client

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/ptit_verified'
NAME='Bao_cao_TPOUR_TRPO_PTIT_cap_nhat_da_ra_soat'
pythoncom.CoInitialize();app=None;doc=None
try:
    app=win32com.client.DispatchEx('Word.Application');app.Visible=False;app.DisplayAlerts=0;app.AutomationSecurity=3
    doc=app.Documents.Open(str(OUT/(NAME+'.docx')),ReadOnly=False,AddToRecentFiles=False)
    doc.Repaginate()
    for toc in doc.TablesOfContents:toc.Update()
    doc.Repaginate()
    for toc in doc.TablesOfContents:toc.UpdatePageNumbers()
    math_count=doc.OMaths.Count
    pages=doc.ComputeStatistics(2)
    # Equations must fit the text column. The PDF is also inspected visually.
    doc.Save();doc.ExportAsFixedFormat(str(OUT/(NAME+'.pdf')),17)
    result={'pages':pages,'word_math_objects':math_count,'toc_count':doc.TablesOfContents.Count,
            'renderer':'Microsoft Word','output_sha256':hashlib.sha256((OUT/(NAME+'.docx')).read_bytes()).hexdigest()}
    (OUT/'word_render_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
finally:
    if doc is not None:doc.Close(SaveChanges=0)
    if app is not None:app.Quit()
    pythoncom.CoUninitialize()
