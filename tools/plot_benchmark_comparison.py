# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ruoxi Qiu
"""Plot recorded benchmark aggregates; no model load or simulated observations.

Install matplotlib==3.10.3 and numpy==2.2.6 in a separate plotting environment.
The optional --qa-tools directory supplies the external panel-alignment auditor.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'2b':'#999933','9b':'#44AA99','new27b':'#4477AA',
          'pilot_2b':'#999933','pilot_9b':'#44AA99','pilot_27b':'#4477AA',
          'canopy':'#882255','neohorse':'#B0B0B0','jev':'#B0B0B0',
          'luna':'#B0B0B0','astra':'#B0B0B0'}
WHITE_TEXT = {'new27b','pilot_27b','canopy'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':10,'axes.titlesize':14,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':11,
    'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
    'axes.linewidth':0.7,'text.color':'#18232E','axes.labelcolor':'#18232E',
    'xtick.color':'#596674','ytick.color':'#18232E','pdf.fonttype':42,'svg.fonttype':'none',
    'legend.frameon':False,'savefig.facecolor':'white'})


def rows_for(data, panel, benchmark, order):
    selected = {r['model_id']:r for r in data['rows'] if r['panel']==panel and r['benchmark']==benchmark}
    assert len(selected)==sum(r['panel']==panel and r['benchmark']==benchmark for r in data['rows'])
    return [selected[k] for k in order]


def label_panel(ax, label, title):
    ax.set_label(label)
    ax.set_title(title, loc='left', pad=13, fontweight='bold')


def percent_axes(ax, horizontal=False):
    if horizontal:
        ax.set_xlim(0,105); ax.set_xticks([0,25,50,75,100]); ax.set_xlabel('Accuracy (%)',labelpad=8)
    else:
        ax.set_ylim(0,112);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Accuracy (%)',labelpad=8)
    ax.spines['bottom'].set_color('#A0A8AF')
    ax.tick_params(axis='y',length=0)


def bar_labels(ax, records, show_model_names=True, pilot_hatch=False, fontsize=10.5):
    y=np.arange(len(records));ax.set_ylim(len(records)-.5,-.5)
    ax.set_yticks(y,[r['display_name'] for r in records] if show_model_names else ['']*len(records))
    ax.tick_params(axis='y',pad=11)
    for pos,row in zip(y,records):
        key=row['model_id'];value=row['value']
        if value is None:
            ax.text(2,pos,'Not published',va='center',ha='left',fontsize=fontsize,color='#747E87')
            continue
        ax.barh(pos,value,height=.65,color=COLORS[key],linewidth=0,
            hatch='///' if pilot_hatch and key.startswith('pilot_') else None)
        text=f"{value:.2f}%"
        ax.text(2.5,pos,text,va='center',ha='left',fontsize=fontsize,
            color='white' if key in WHITE_TEXT else '#18232E',
            fontweight='bold' if key=='canopy' else 'normal')
    for tick,row in zip(ax.get_yticklabels(),records):
        if row['model_id']=='canopy':tick.set_color(COLORS['canopy']);tick.set_fontweight('bold')
    percent_axes(ax,horizontal=True)


def export(fig, stem, args, data):
    fig.canvas.draw()
    measured=[{'panel':ax.get_label(),'bbox_pt':[v*scale for v,scale in zip(ax.get_position().bounds,
        [fig.get_figwidth()*72,fig.get_figheight()*72,fig.get_figwidth()*72,fig.get_figheight()*72])]} for ax in fig.axes]
    (args.qa_out/(stem+'.plot_rectangles.json')).write_text(json.dumps(measured,indent=2)+'\n')
    if args.qa_tools:
        sys.path.insert(0,str(args.qa_tools))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig,json_out=args.qa_out/(stem+'.alignment.json'),
            overlay_svg=args.qa_out/(stem+'.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,
            require_panel_labels=False,strict=True)
    fig.savefig(args.out/(stem+'.png'),dpi=300)
    fig.savefig(args.out/(stem+'.svg'))
    fig.savefig(args.out/(stem+'.pdf'))
    plt.close(fig)
    return {'figure':stem,'panels':len(measured),'source_data_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'editable_vector_text':True,'PNG_dpi':300,'source_checkpoint_sha256':data['checkpoint_sha256'],
        'statistics':'Recorded descriptive percentages; counts remain in source data. No artificial error bars or model-averaged composite.',
        'files':{suffix:hashlib.sha256((args.out/(stem+'.'+suffix)).read_bytes()).hexdigest() for suffix in ['png','svg','pdf']}}


def main_figure(data,args):
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    fig.subplots_adjust(left=.21,right=.975,top=.865,bottom=.18,wspace=.18)
    order=['2b','9b','new27b','canopy']
    for ax,benchmark,label,title in zip(axes,['Overall','Hard'],['a','b'],['JevBench','Hard']):
        label_panel(ax,label,title)
        bar_labels(ax,rows_for(data,'jevbench',benchmark,order),show_model_names=ax is axes[0],fontsize=11.5)
    # The shared row-label column applies to both aligned panels.
    # User-requested concise titles replace visible panel letters.
    fig.canvas.draw()
    assert abs(axes[0].get_position().width-axes[1].get_position().width)<1e-10
    return export(fig,'benchmark_comparison',args,data)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=ROOT/'results/benchmark_comparison.json')
    parser.add_argument('--out',type=Path,default=ROOT/'docs/assets');parser.add_argument('--qa-out',type=Path)
    parser.add_argument('--qa-tools',type=Path);args=parser.parse_args()
    if args.qa_out is None:args.qa_out=args.out/'qa'
    args.out.mkdir(parents=True,exist_ok=True);args.qa_out.mkdir(parents=True,exist_ok=True)
    data=json.loads(args.data.read_text())
    assert data['checkpoint_sha256']=='506410609a6ba6eb422182fddda095b81fa4923011ceaa3717e85a74817237f4'
    for row in data['rows']:
        assert row['available']==(row['value'] is not None)
        if row['correct'] is not None:assert abs(row['value']-row['correct']/row['n']*100)<1e-7
    receipts=[main_figure(data,args)]
    (args.qa_out/'export_receipt.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print(json.dumps({'figures':[r['figure'] for r in receipts],'records':len(data['rows']),'model_inference':False}))


if __name__=='__main__':
    main()
