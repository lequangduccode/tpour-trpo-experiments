"""Render saved LIVE inference evidence as a plain figure (not a simulated web screenshot)."""
import json
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw(root='results/anchored_kaggle'):
    root=Path(root); result=json.loads((root/'live_demo/anchored_custom_query.json').read_text(encoding='utf-8'))
    fig,ax=plt.subplots(figsize=(11,6)); ax.axis('off')
    ax.set_title('Live inference: trained TPOUR + retention (seed42)',fontsize=15,pad=20)
    lines=['QUERY: '+textwrap.fill(result['query'],90),'']
    for row in result['results']:
        lines.extend([textwrap.fill(f'{row["rank"]}. {row["title"]}',100),
                      f'Document ID: {row["docid"]} | dot-product score: {row["score"]:.5f}',
                      textwrap.fill(textwrap.shorten(row['excerpt'],width=240,placeholder=' …'),110),''])
    ax.text(0,1,'\n'.join(lines),transform=ax.transAxes,va='top',fontsize=10)
    fig.text(.05,.02,'Rendered from live inference JSON. New query has no gold labels; no accuracy claimed.',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]); out=root/'demo/06_live_inference.png'; fig.savefig(out,dpi=180); plt.close(fig);print(out)


if __name__=='__main__':draw()
