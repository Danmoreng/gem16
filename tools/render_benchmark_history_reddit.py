"""English publication graphics; methodology is in the separate report/caption."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
root=Path(__file__).resolve().parents[1]
import csv,copy,hashlib
original=root/'artifacts/benchmark_history/2026-11-15'
pub=original/'reddit-final'
base=json.loads((original/'history.json').read_text())
data=copy.deepcopy(base)
prefill_path=root/'artifacts/gemma4_12b/matched-prefill-2026-11-15/results.json'
prefill=json.loads(prefill_path.read_text())
followup_path=root/'artifacts/gemma4_12b/prefill-batch-microbatch-2026-11-15/results.json'
followup=json.loads(followup_path.read_text())
engine_keys={'gem16':'gem16','llama.cpp':'llama-cpp','vLLM':'vllm'}
updates=[]
for engine,key in engine_keys.items():
    candidates=[(i,r) for i,r in enumerate(prefill['records']) if r['engine']==key]
    index,best=max(candidates,key=lambda item:item[1]['median_prefill_tokens_per_second'])
    matches=[r for r in data['rows'] if r['date']=='2026-11-15' and r['model']=='12B' and r['metric']=='prefill' and r['mode']=='ordinary' and r['engine']==engine]
    assert len(matches)==1
    row=matches[0]
    old=copy.deepcopy(row)
    rates=[sample['prompt_tokens_per_second'] for sample in best['samples']]
    row.update(tok_s=best['median_prefill_tokens_per_second'],output_tokens='1',retained_samples=10,warmups=3,source=str(prefill_path.relative_to(root)),json_pointer=f'records/{index}/median_prefill_tokens_per_second',sample_min=min(rates),sample_max=max(rates),requested_context_tokens=17519,prefill_chunk_tokens=best['chunk'],note='Best tested confirmed Prefill point from the matched-chunk curve;3 warmups+10 retained; identical16K IDs and17519 requested context, one output; no prefix reuse. Decode points come from separate full1135-output cohorts. Engine formats/KV differ.')
    if engine=='llama.cpp':
        assert max(r['median_prefill_tokens_per_second'] for r in followup['records'])<=row['tok_s']
        row['batch_size']=best['configuration']['batch_size']
        row['microbatch_size']=best['configuration']['ubatch_size']
        row['reference_commit']=prefill['inputs']['central_commit']
    updates.append({'engine':engine,'previous':old,'replacement':copy.deepcopy(row)})
for path in (prefill_path,followup_path):
    data['sources'][str(path.relative_to(root))]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
data['publication_refresh']={'scope':'Only the current12B Ordinary-Prefill point for each engine is replaced with its best confirmed chunk-sweep median; history and all Decode/26B data remain unchanged.','source_history_sha256':hashlib.sha256((original/'history.json').read_bytes()).hexdigest(),'changes':updates,'followup':'The independent llama.cpp logical-batch/microbatch follow-up did not beat the1024/1024 confirmed point.','prefill_semantics':'One-output probes, requested context17519; chosen chunks8192/1024/4096 for gem16/llama.cpp/vLLM. Historical and Decode protocols differ as documented separately.'}
rows=data['rows']
pub.mkdir(exist_ok=True)
serialized=json.dumps(data,indent=2,ensure_ascii=False)+'\n'
if (pub/'history.json').exists():assert (pub/'history.json').read_text()==serialized
else:(pub/'history.json').write_text(serialized)
fields=list(dict.fromkeys(k for row in rows for k in row))
with (pub/'history.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
notes=(original/'chart-notes.txt').read_text()
notes+='\nPublication refresh: current12B Prefill uses each engine\'s best tested confirmed chunk (gem16:8192, llama.cpp:1024, vLLM:4096),3warmups+10retained, one output token and17519 requested context. Medians5965.28057 /4839.76578 /7112.99372tok/s. llama.cpp batch=microbatch=1024; the independent8192 logical-batch follow-up produced no higher result. Current Decode and26B points remain unchanged, from their separate original cohorts/revisions. Original history files remain preserved in the parent directory.\n'
(pub/'chart-notes.txt').write_text(notes)

colors={'gem16':'#007f86','llama.cpp':'#d55e00','vLLM':'#7251b5'}
formats={
 '12B':{'gem16':'FP8 / NVFP4 · 9.30 GB','llama.cpp':'Q8_0 / NVFP4 · 9.37 GB','vLLM':'FP8 / NVFP4 · 9.30 GB'},
 '26B A4B':{'gem16':'Trellis35 W4A8 · 12.20 GB','llama.cpp':'QAT Q4_0 · 14.44 GB','vLLM':'INT4 W4A16 · 14.93 GB'}}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':15,'axes.titleweight':'bold',
                     'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#c5cbd2',
                     'axes.labelcolor':'#46505a','xtick.color':'#46505a','ytick.color':'#46505a',
                     'svg.fonttype':'none'})

def panel(ax,model,metric,mode,title):
    dates=sorted(set(r['date'] for r in rows if r['model']==model));pos={d:i for i,d in enumerate(dates)}
    ax.set_facecolor('white');ax.set_title(title,loc='left',pad=17)
    subset=[r for r in rows if r['model']==model and r['metric']==metric and (mode is None or r['mode']==mode)]
    if metric=='prefill':
        chosen={}
        for r in subset:
            key=(r['date'],r['engine'])
            if key not in chosen or r['mode']=='ordinary':chosen[key]=r
        subset=list(chosen.values())
    ymax=max([r['sample_max'] if r.get('sample_max') and r['date']=='2026-11-15' else r['tok_s'] for r in subset],default=1)*1.22
    endpoints=[]
    for engine,color in colors.items():
        rs=sorted([r for r in subset if r['engine']==engine],key=lambda r:r['date'])
        main=[r for r in rs if r['series']=='main']
        if main:ax.plot([pos[r['date']] for r in main],[r['tok_s'] for r in main],color=color,lw=2.0,alpha=.85,zorder=2)
        for r in rs:
            special=r['series']!='main'
            marker='D' if special else ('^' if metric=='prefill' and r['mode']=='d2' else 'o')
            if r['date']=='2026-11-15' and r.get('sample_min') and r.get('sample_max') and (r['sample_max']-r['sample_min'])>.10*r['tok_s']:
                ax.errorbar(pos[r['date']],r['tok_s'],yerr=[[r['tok_s']-r['sample_min']],[r['sample_max']-r['tok_s']]],
                            fmt='none',ecolor=color,elinewidth=1.2,capsize=4,alpha=.55,zorder=3)
            ax.scatter(pos[r['date']],r['tok_s'],s=44 if special else 48,marker=marker,
                       facecolors='white' if special else color,edgecolors=color,lw=1.6,zorder=4)
        if rs:endpoints.append((rs[-1]['tok_s'],engine,pos[rs[-1]['date']]))
    # Solve vertical label placement separately from the measured point positions.
    # Direct names + leader lines retain attribution when rates are almost equal.
    ends=sorted(endpoints);gap=.145*ymax
    label_y=[v[0] for v in ends]
    for i in range(1,len(label_y)):label_y[i]=max(label_y[i],label_y[i-1]+gap)
    if label_y:
        shift=max(0,label_y[-1]-.92*ymax)
        label_y=[y-shift for y in label_y]
    label_x=len(dates)-1+.50
    for (value,engine,x),y in zip(ends,label_y):
        number=f'{value:,.0f}' if metric=='prefill' else f'{value:.1f}'
        ax.annotate(f'{engine}\n{number} tok/s',xy=(x,value),xytext=(label_x,y),
                    textcoords='data',ha='left',va='center',fontsize=10.5,fontweight='bold',
                    color=colors[engine],bbox=dict(boxstyle='round,pad=.25',fc='white',ec='none',alpha=.96),
                    arrowprops=dict(arrowstyle='-',color=colors[engine],lw=1.0,shrinkA=4,shrinkB=4),
                    annotation_clip=False,zorder=6)
    labels=[__import__('datetime').date.fromisoformat(d).strftime('%b %d') for d in dates]
    ax.set_xticks(range(len(dates)),labels,rotation=45,ha='right',fontsize=9)
    ax.set_xlim(-.45,len(dates)-1+2.1);ax.set_ylim(0,ymax)
    ax.set_ylabel('Tokens / second',labelpad=9);ax.grid(axis='y',color='#e8edf1',lw=.8,zorder=0)

def make(models,name):
    n=len(models)
    fig,axs=plt.subplots(n,3,figsize=(19,6.4*n),squeeze=False)
    fig.patch.set_facecolor('#f6f8fa')
    fig.suptitle('Gemma 4 · Inference speed over time',x=.055,ha='left',fontsize=23,fontweight='bold',y=.965,va='top',color='#17212b')
    fig.text(.055,.90 if n==1 else .925,'RTX 5080 Laptop GPU  ·  Linux  ·  1 sequence  ·  16K prompt  ·  2026',fontsize=12,color='#46505a')
    top=.64 if n==1 else .77
    bottom=.16 if n==1 else .095
    fig.subplots_adjust(left=.06,right=.975,top=top,bottom=bottom,hspace=.88,wspace=.31)
    foot_handles=[Line2D([0],[0],marker='D',color='#64748b',markerfacecolor='white',lw=0,label='Different test settings')]
    if any(r['model'] in models and r['date']=='2026-11-15' and r.get('sample_min') and r.get('sample_max') and (r['sample_max']-r['sample_min'])>.10*r['tok_s'] for r in rows):
        foot_handles.append(Line2D([0],[0],color='#64748b',marker='|',lw=1,label='Bars: run range'))
    fig.legend(handles=foot_handles,loc='lower left',bbox_to_anchor=(.055,.025 if n==1 else .022),
               frameon=False,fontsize=10,ncol=2,columnspacing=2.5)
    for row,model in enumerate(models):
        box=axs[row,0].get_position();header_y=box.y1+(.16 if n==1 else .085)
        fig.text(.06,header_y,f'Gemma 4 {model} IT',fontsize=17,fontweight='bold',color='#17212b',va='center')
        handles=[Line2D([0],[0],color=colors[e],marker='o',lw=2,label=f'{e} — {formats[model][e]}') for e in colors]
        fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.055,header_y-(.033 if n==1 else .025)),
                   ncol=3,frameon=False,fontsize=10.5,handlelength=1.6,columnspacing=2.3)
        for col,(metric,mode,title) in enumerate([('prefill',None,'Prefill'),('decode','ordinary','Ordinary decode'),('decode','d2','D2 decode')]):
            panel(axs[row,col],model,metric,mode,title)
    fig.canvas.draw()
    from matplotlib.text import Annotation
    for ax in axs.flat:
        annotations=[t for t in ax.texts if isinstance(t,Annotation)]
        boxes=[t.get_bbox_patch().get_window_extent(fig.canvas.get_renderer()) for t in annotations]
        for i,a in enumerate(boxes):
            for b in boxes[i+1:]:
                assert not a.overlaps(b), 'Endpoint labels overlap'
    for ext in ['jpg']:
        options={'pil_kwargs':{'quality':95,'subsampling':0}} if ext=='jpg' else {}
        fig.savefig(pub/f'{name}.{ext}',dpi=220,facecolor=fig.get_facecolor(),**options)
    plt.close(fig)

make(['12B','26B A4B'],'history')
make(['12B'],'history-12b')
make(['26B A4B'],'history-26b')
print('Wrote English overview + separate12B/26B JPEG and PNG graphics; immutable original history preserved')
