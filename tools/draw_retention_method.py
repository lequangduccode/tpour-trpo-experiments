"""Draw the implemented training change with Matplotlib, without generated imagery."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw(path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(11,5)); ax.set_xlim(0,11); ax.set_ylim(0,5); ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',facecolor=color,edgecolor='#555555'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    def arrow(a,b,label=''):
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#444444','lw':1.3})
        if label: ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.1,label,fontsize=8,ha='center')
    box(.3,3.9,10.2,.55,'Same training texts: query / preferred / rejected','#f3f3f3')
    box(.4,2.35,3,1,'Current encoder\ntrainable student','#ebeff5')
    box(4,2.35,3,1,'Momentum reference\nEMA of current encoder','#f1eade')
    box(7.6,2.35,3,1,'Fixed Contriever teacher\nfrozen at initialization','#f4e5e8')
    arrow((1.9,3.9),(1.9,3.35)); arrow((5.5,3.9),(5.5,3.35)); arrow((9.1,3.9),(9.1,3.35))
    arrow((3.4,2.85),(4,2.85),'EMA')
    box(.4,.55,4.2,1.05,'Original objective\n0.925 CE + 0.075 TRPO\n(reference + document queue)','#f1eade')
    box(6,.55,4.6,1.05,'Added retention objective\n10 x (ranking KL + relative embedding MSE)\nfixed teacher targets, detached','#f4e5e8')
    arrow((1.9,2.35),(1.9,1.6)); arrow((5.5,2.35),(3.5,1.6)); arrow((9.1,2.35),(9.1,1.6)); arrow((2.8,2.35),(7,1.6))
    ax.text(5.5,.08,'Both losses update only the student. Inference uses only the trained student.',ha='center',fontsize=10)
    fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)
    print(path)


if __name__=='__main__': draw('reports/anchored_improvement/method.png')
