"""Export a saved DOCX to PDF using the installed Word document converter, hidden and read-only."""
import argparse
import json
from pathlib import Path


def export(source,output):
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize();word=None;doc=None
    try:
        word=win32com.client.DispatchEx('Word.Application');word.Visible=False;word.DisplayAlerts=0
        word.AutomationSecurity=3
        doc=word.Documents.Open(str(Path(source).resolve()),ReadOnly=True,AddToRecentFiles=False)
        doc.Repaginate();pages=doc.ComputeStatistics(2)
        doc.ExportAsFixedFormat(str(Path(output).resolve()),17)
        print(json.dumps({'pages':pages,'pdf':str(Path(output).resolve())}))
    finally:
        if doc is not None:doc.Close(SaveChanges=0)
        if word is not None:word.Quit()
        pythoncom.CoUninitialize()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();export(a.source,a.output)
