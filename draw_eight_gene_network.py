"""Draw four strain RNA responses on the same eight-gene MG1655 reference scaffold.
Usage: python draw_eight_gene_network.py --project '/path/to/project'
Requires pandas, numpy, matplotlib. No models are refitted.
"""
from pathlib import Path
import argparse, json, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

GENES=['flhD','flhC','fliA','fliD','cheZ','flgK','flgL','motB']
STRAINS=['MG1655','TW09308','TW11588','IAI1']

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'eight_gene_network')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    def export(name):
        obj=json.loads((a.project/'data/regulondb'/f'{name}.json').read_text())
        df=pd.read_csv(io.StringIO(obj['content']),sep='\t',dtype=str).fillna('')
        df.columns=[c.split(')',1)[-1].strip() for c in df.columns]
        return df.apply(lambda col:col.str.strip())
    tf=export('NetworkRegulatorGene'); sigma=export('NetworkSigmaGene')
    rows=[]
    for r in tf.loc[tf.regulatorName.eq('FlhDC') & tf.regulatedGeneName.isin(GENES)].itertuples():
        rows.append(dict(source='FlhDC',target=r.regulatedGeneName,function=r.function,confidence=r.confidenceLevel,shown=True))
    for r in sigma.loc[sigma.sigmaName.eq('sigma28') & sigma.regulatedGeneName.isin(GENES)].itertuples():
        rows.append(dict(source='FliA',target=r.regulatedGeneName,function=r.function,confidence=r.confidenceLevel,shown=r.regulatedGeneName!='fliA'))
    edges=pd.DataFrame(rows).drop_duplicates();edges.to_csv(a.out/'reference_edges.csv',index=False)
    assert len(edges.loc[edges.shown])==7
    assert edges.function.eq('+').all()
    raw=pd.read_csv(a.project/'results/displayed_effects.csv')
    assert not raw.duplicated(['strain','gene']).any()
    audit=pd.read_csv(a.project/'results/selected_sequence_audits.csv')
    data=[]
    for strain in STRAINS:
        for gene in GENES:
            hit=raw.loc[raw.strain.eq(strain)&raw.gene.eq(gene)]
            if len(hit):
                r=hit.iloc[0]
                data.append(dict(strain=strain,gene=gene,log2FC=r.log2FC,fold_change=2**r.log2FC,padj=r.padj,
                    ci_low=r.log2FC-1.96*r.lfcSE,ci_high=r.log2FC+1.96*r.lfcSE,
                    significant=bool(r.padj<.05),status='displayed'))
            else:
                sq=audit.loc[audit.strain.eq(strain)&audit.gene_name.eq(gene)]
                reason='not_in_displayed_results'
                if len(sq)==1 and not bool(sq.iloc[0].sequence_supported_pair):reason='sequence_audit_failed; length_filter_failed'
                data.append(dict(strain=strain,gene=gene,log2FC=np.nan,fold_change=np.nan,padj=np.nan,
                    ci_low=np.nan,ci_high=np.nan,significant=False,status=reason))
    data=pd.DataFrame(data);data.to_csv(a.out/'node_responses.csv',index=False)
    norm=TwoSlopeNorm(vmin=-4,vcenter=0,vmax=4);cmap=plt.get_cmap('RdBu_r')
    pos={'flhD':(.31,.87),'flhC':(.69,.87),'fliA':(.5,.59),
         'fliD':(0,.23),'cheZ':(.25,.23),'flgK':(.5,.23),'flgL':(.75,.23),'motB':(1,.23)}
    def panel(ax,strain):
        ax.set(xlim=(-.13,1.13),ylim=(-.02,1.07));ax.axis('off')
        ax.set_title(strain,fontsize=16,fontweight='bold',pad=12)
        group=FancyBboxPatch((.145,.785),.71,.205,boxstyle='round,pad=.015',facecolor='none',edgecolor='#888888',lw=1)
        ax.add_patch(group);ax.text(.5,.965,'FlhDC components (RNA)',ha='center',va='center',fontsize=9,color='#555555')
        # Scaffold is constant across strains; arrows are reference knowledge, not inferred edges.
        for edge in edges.loc[edges.shown].itertuples():
            if edge.source=='FlhDC':
                if edge.target=='fliA':start=(.5,.77);end=(.5,.66);rad=0
                else:start=(.145,.83);end=(-.04,.30);rad=.38
            else:
                start=(.5,.525);x,y=pos[edge.target];end=(x,y+.065)
                rad=0
            patch=FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,lw=1.25,
                linestyle='--' if edge.confidence=='W' else '-',color='#66717d',connectionstyle=f'arc3,rad={rad}',zorder=1)
            ax.add_patch(patch)
        subset=data.loc[data.strain.eq(strain)].set_index('gene')
        for gene,(x,y) in pos.items():
            r=subset.loc[gene];ok=np.isfinite(r.log2FC)
            fc=cmap(norm(r.log2FC)) if ok else '#dddddd'
            node=FancyBboxPatch((x-.09,y-.06),.18,.12,boxstyle='round,pad=.009',facecolor=fc,edgecolor='#69717b',lw=.8,zorder=3)
            ax.add_patch(node)
            color='white' if ok and abs(r.log2FC)>2.5 else '#18212b'
            text=f'{gene}{" *" if r.significant else ""}\n{r.log2FC:+.2f}' if ok else gene+'\nExcluded'
            ax.text(x,y,text,ha='center',va='center',fontsize=9,color=color,linespacing=1.45,zorder=4)
    for layout,filename,size in [((1,4),'four_strains_network',(21,7)),((2,2),'four_strains_network_2x2',(12,12))]:
        fig,axs=plt.subplots(*layout,figsize=size)
        for ax,strain in zip(np.ravel(axs),STRAINS):panel(ax,strain)
        fig.suptitle('Eight selected flagellar-system genes: RNA responses on the MG1655 reference scaffold',fontsize=16,y=.97)
        if layout==(1,4):fig.subplots_adjust(top=.83,bottom=.22,left=.02,right=.98,wspace=.14);cax=fig.add_axes([.36,.19,.28,.022])
        else:fig.subplots_adjust(top=.90,bottom=.18,left=.05,right=.95,hspace=.22,wspace=.15);cax=fig.add_axes([.30,.14,.40,.016])
        cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',ticks=[-4,-2,0,2,4])
        cb.set_label('log2(starvation / chemostat): blue = down; red = up',fontsize=10)
        footer=('Node values: log2FC; * within-strain FDR < 0.05. Unstarred estimates remain uncertain; gray = excluded, not zero expression.\n'
                'Arrows: positive MG1655 reference regulation. Solid = confirmed/strong; dashed = weak. FliA autoregulation and other inputs omitted.\n'
                'Identical arrows are a shared reference, not validated strain-specific networks. RNA changes do not measure protein activity.\n'
                'MG1655 values: MG1655/TW09308 fit; other strains: respective pairwise fits. TW11588 motB: unsupported mapping and length exclusion.')
        fig.text(.5,.025,footer,ha='center',va='bottom',fontsize=9,linespacing=1.5,color='#444444')
        for ext in ['png','svg']:fig.savefig(a.out/f'{filename}.{ext}',dpi=200,facecolor='white')
        plt.close(fig)
    print('Saved network figures, node values and verified reference edges to',a.out)

if __name__=='__main__':main()
