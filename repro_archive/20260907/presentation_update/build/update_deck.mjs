import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';

const dir=path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/,'$1'));
const workspace=path.dirname(dir);
const source=process.argv.slice(2).find(a=>!a.startsWith('--')) ?? 'D:/Sync/Study/USYD/CapStone/presentation/BetterLViTv1.pptx';
const skill=process.env.PRESENTATION_SKILL_DIR ?? 'C:/Users/dtftn/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const runtimePython=process.env.RUNTIME_PYTHON ?? 'C:/Users/dtftn/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const data=JSON.parse(await fs.readFile(path.join(dir,'evidence.json'),'utf8'));
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snapshot=(await p.inspect({kind:'slide,textbox,shape,table,chart',maxChars:250000})).ndjson.split('\n').filter(Boolean).map(x=>JSON.parse(x));
const F=Object.fromEntries(data.formal.map(x=>[x.id,x]));
const R=Object.fromEntries(data.recent.map(x=>[x.paper_id,x]));
const pct=v=>(100*v).toFixed(3);
const dec=v=>v.toFixed(6);
const pp=v=>(v>=0?'+':'')+(100*v).toFixed(3);
const black='#111111',gray='#606060',light='#F3F3F3',rule='#D6D6D6';
function text(s,str,x,y,w,h,size=24,opt={}){
  const t=s.shapes.add({name:opt.name??str.slice(0,50),geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  t.text=str;t.text.style={typeface:opt.serif?'Cambria':'Calibri',fontSize:size,color:opt.color??black,bold:opt.bold??false,alignment:opt.align??'left',verticalAlignment:opt.valign??'top',wrap:'square',autoFit:'none',insets:{left:0,right:0,top:0,bottom:0},...opt.style};return t;
}
function box(s,str,x,y,w,h,opt={}){
 const b=s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:opt.dark?black:light,line:{fill:rule,width:1}});
 text(s,str,x+14,y+12,w-28,h-24,opt.size??23,{bold:opt.bold??true,color:opt.dark?'#FFFFFF':black,align:opt.align??'center',valign:'middle'});return b;
}
function connect(s,a,b,from='right',to='left') {s.shapes.connect(a,b,{kind:'elbow',fromSide:from,toSide:to,line:{fill:'#555555',width:2},tail:{type:'arrow',width:'med',length:'med'}});}
function note(s,str,y=610){text(s,str,67.2,y,1145.6,40,18,{color:gray});}
function band(s,str,y=552,h=80){box(s,str,67.2,y,1145.6,h,{dark:true,size:27,align:'left'});}
function paras(s,arr,x=67.2,y=184,w=547.2,gap=118,size=25){arr.forEach((a,i)=>text(s,a,x,y+i*gap,w,gap-16,size));}
function table(s,values,x=67.2,y=195,w=1145.6,h=350,widths=null,size=22){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,...(widths?{columnWidths:widths}:{})});
 t.borders.assign({fill:rule,width:1,style:'solid'});
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({margins:{left:12,right:10,top:6,bottom:5},anchor:'center'});
 for(let r=0;r<values.length;r++){t.rows[r].height=h/values.length;for(let c=0;c<values[0].length;c++){let cell=t.getCell(r,c);cell.fill=r===0?black:(r%2===0?'#F4F4F4':'#FFFFFF');cell.text.style={typeface:'Calibri',fontSize:size,bold:r===0,color:r===0?'#FFFFFF':black,verticalAlignment:'middle',alignment:'left',autoFit:'none'};}}
 return t;
}
function chart(s,categories,values,x,y,w,h,{min=0,max=2,format='0.00',title='IoU gain (percentage points)'}={}){
 const c=s.charts.add('bar',{position:{left:x,top:y,width:w,height:h},title,titleTextStyle:{fontSize:21,typeface:'Calibri',fill:black},categories,series:[{name:title,values:values.map(v=>Number(v.toFixed(8))),fill:'#444444',valuesFormatCode:format}],hasLegend:false,barOptions:{direction:'column',grouping:'clustered',gapWidth:140},xAxis:{textStyle:{fontSize:18,typeface:'Calibri',fill:gray}},yAxis:{min,max,numberFormatCode:format,textStyle:{fontSize:15,typeface:'Calibri',fill:gray}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:format,textStyle:{fontSize:19,typeface:'Calibri',fill:black}}});
 applyPresentationChartFont(c,{fontFamily:'Calibri'});return c;
}
function setup(n,title,footer,badge=''){
 const recs=snapshot.filter(r=>r.slide===n),s=p.resolve(recs.find(r=>r.kind==='slide').id);
 for(const r of recs.filter(r=>['textbox','shape'].includes(r.kind))){
  const a=p.resolve(r.id),b=r.bbox??[];
  if(r.name==='Text 0'){a.text=title;a.text.style={typeface:'Cambria',fontSize:40,bold:true,color:black,autoFit:'none',verticalAlignment:'middle',insets:{left:0,right:0,top:0,bottom:0}};a.position={left:67.2,top:38,width:badge?1046.4:1145.6,height:100};}
  else if(b[1]>=660){if(b[0]<1000){a.text=footer;a.text.style={typeface:'Calibri',fontSize:13,color:'#999999',autoFit:'none'};}}
  else if(b[0]>1120 && b[1]<90){if(!badge)s.shapes.deleteById(a.id);else if(r.kind==='textbox'){a.text=badge;a.text.style={typeface:'Calibri',fontSize:13,bold:true,color:'#FFFFFF',alignment:'center',verticalAlignment:'middle',autoFit:'none'};}}
  else s.shapes.deleteById(a.id);
 }
 for(const c of [...s.charts.items])s.charts.deleteById(c.id);
 for(const t of [...s.tables.items])s.tables.deleteById(t.id);
 return s;
}
const github='https://github.com/razaxq/BetterLViT';
const tracker=`${github}/blob/docs/experiment-tracker/docs/EXPERIMENT_TRACKER.md`;
function notes(s,body,refs=[]){s.speakerNotes.textFrame.setText(body+'\n\nEvidence (snapshot 7 September 2026):\n'+refs.join('\n'));}

// Cover: preserve the author's supplied identity and affiliation.
{
 const s=p.slides.items[0],title=p.resolve(snapshot.find(r=>r.slide===1&&r.name==='Text 0').id);
 title.text='BetterLViT: Text-Guided Segmentation\nof COVID-19 Infections in Chest X-rays';
 title.text.style={typeface:'Cambria',fontSize:46,bold:true,color:'#FFFFFF',verticalAlignment:'middle',autoFit:'none'};
 text(s,'Project update: 7 September 2026',67.2,589,1100,32,22,{color:'#BEBEBE'});
 notes(s,'Current presentation replaces the earlier LoRA / modality-dropout narrative. EPPA has positive paired Test results. P11 is a running exploratory architecture candidate and has no verified result in this snapshot.',[tracker]);
}
{
 const s=setup(2,'Better infection masks require control of false positives','The problem');
 paras(s,['The task: predict an infection mask from a chest X-ray and its associated text prompt.','Diffuse lesions and small foreground areas make overlap sensitive to both missed pixels and extra foreground.','The primary objective is higher Test macro IoU. Dice provides a complementary overlap measure.'],67.2,191,540,125,25);
 const a=box(s,'Chest X-ray',690,204,210,82),b=box(s,'Text prompt',975,204,210,82),c=box(s,'BetterLViT',780,356,330,84,{dark:true}),d=box(s,'Infection mask',780,509,330,74);
 connect(s,a,c,'bottom','top');connect(s,b,c,'bottom','top');connect(s,c,d,'bottom','top');
 notes(s,'Task schematic, not a clinical image or a prediction example. IoU = TP/(TP+FP+FN). No claim that every CXR has a report or that the model is clinically validated.',[tracker]);
}
{
 const s=setup(3,'One supported contribution, one candidate under test','Roadmap');
 const rows=[['C1','FAM-EPPA V4-B: frequency-aware decoder fusion','Positive Test IoU gains in paired internal ablations support EPPA as the first architecture contribution.'],['C2','P11: dual-grain visual guides','A 28 × 28 shallow visual branch complements the existing 14 × 14 semantic tokens. Performance and novelty remain to be established.'],['E','Evidence from controlled experiments','Completed Test results, negative mechanisms and reproducible checkpoints guide the next decision. This is the evaluation framework.']];
 rows.forEach((r,i)=>{let y=184+i*147;box(s,r[0],67.2,y,79,79,{dark:true,size:30});text(s,r[1],178,y,1000,45,28,{bold:true});text(s,r[2],178,y+51,1000,77,23);});
 notes(s,'The Capstone requires two innovations. This deck does not claim that the second has been achieved. LoRA and auxiliary supervision are not counted as independent innovations.',[tracker,`${github}/blob/2fc6ab5c8e4662d741fd8b994e55b780391948ac/docs/P11_DUAL_GRAIN.md`]);
}
{
 const s=setup(4,'The model combines a CNN path with text-aware tokens','Architecture','BASE');
 paras(s,['CNN features retain spatial detail at four resolutions.','The visual Transformer streams use text features and return semantic guides for the decoder.','Each stream uses 14 × 14 tokens. The shallow guides expand those tokens back to the CNN resolution.'],67.2,195,530,125,25);
 const a=box(s,'CXR',680,187,190,64),b=box(s,'CNN hierarchy',680,323,215,83),c=box(s,'CXR-BERT\nFrozen',978,187,210,83),d=box(s,'Visual + text\nTransformer',978,323,210,83),e=box(s,'EPPA decoder and mask head',727,518,460,83,{dark:true});
 connect(s,a,b,'bottom','top');connect(s,b,d);connect(s,c,d,'bottom','top');connect(s,b,e,'bottom','top');connect(s,d,e,'bottom','top');
 notes(s,'Schematic of the current implementation. It corrects the old deck claim that PLAM directly performs text-to-pixel attention. Current reconstructed visual tokens guide the decoder. The original LViT paper and the internal B0 baseline are distinct comparisons.',[`${github}/blob/2fc6ab5c8e4662d741fd8b994e55b780391948ac/nets/LViT.py`,`${github}/blob/2fc6ab5c8e4662d741fd8b994e55b780391948ac/nets/Vit.py`]);
}
{
 const s=setup(5,'LoRA gives mixed results across the paired settings','Language branch','BASE');
 paras(s,['CXR-BERT supplies domain-specific text features. Historical runs compared frozen and LoRA-adapted text encoders.','LoRA reduces IoU with original PLAM + Dice/BCE, but improves it with EPPA + Dice/Focal.','The current architecture experiments keep CXR-BERT frozen and LoRA disabled to hold the language branch fixed.'],67.2,185,530,126,25);
 table(s,[['Matched setting','Frozen IoU','LoRA IoU'],['PLAM + Dice/BCE','74.808% (B0)','74.328% (A0)'],['EPPA + Dice/Focal','75.541% (A9)','75.991% (A4)']],656,211,556,244,[246,155,155],21);
 text(s,'150 epochs, seed 1219\nTest macro IoU, threshold 0.5',656,482,550,80,21,{color:gray});
 band(s,'LoRA remains a configuration choice. It is not the second contribution.',577,67);
 notes(s,'Do not infer LoRA is uniformly harmful: A4 exceeds A9 by 0.4497 percentage points of Test IoU. A0 is 0.4795 points below B0. These are single-seed results, not stability estimates.',[tracker]);
}
{
 const s=setup(6,'EPPA separates semantic fusion from detail refinement','Contribution 1: FAM-EPPA V4-B','EPPA');
 const a=box(s,'CNN skip',67,184,275,68),b=box(s,'PLAM semantic guide',483,184,309,68),c=box(s,'Upsampled decoder',918,184,295,68);
 const d=box(s,'Haar low / high\nreconstructions',67,328,275,96),e=box(s,'Low-frequency context\nand adaptive filtering',483,328,309,96),f=box(s,'Semantic fusion',918,328,295,96);
 const g=box(s,'Fused skip with bounded detail residuals',290,506,700,79,{dark:true});
 connect(s,a,d,'bottom','top');connect(s,d,e);connect(s,b,e,'bottom','top');connect(s,c,f,'bottom','top');connect(s,e,f);connect(s,d,g,'bottom','top');connect(s,f,g,'bottom','top');
 note(s,'Adaptive ALPF / AHPF operates at up4 and up3. EPPA guides the decoder at all four stages.');
 notes(s,'Conceptual module diagram of implemented FAM-EPPA V4-B. ALPF smooths aligned decoder features. AHPF adds a complementary high-frequency skip residual. Skip, PLAM and decoder features remain separate until fusion. The figure abstracts internal channel projections, gates and convolutions.',[`${github}/blob/2fc6ab5c8e4662d741fd8b994e55b780391948ac/nets/eppa.py`,`${github}/blob/2fc6ab5c8e4662d741fd8b994e55b780391948ac/Config.py`]);
}
{
 const s=setup(7,'Adaptive filters vary by location and channel group','Contribution 1: filtering mechanism','EPPA');
 paras(s,['Haar analysis supplies complementary low- and high-frequency reconstructions.','V4-B predicts mixtures over identity, 3 × 3 blur and 5 × 5 blur filters using feature context.','Bounded ALPF and AHPF strengths control decoder smoothing and skip-detail residuals.'],67.2,190,530,126,25);
 box(s,'Filter bank',672,190,519,58,{dark:true});
 ['Identity','3 × 3 binomial','5 × 5 binomial'].forEach((x,i)=>box(s,x,672+i*177,298,165,77,{size:21}));
 box(s,'Spatial + group-specific mixture weights',672,428,519,79);
 text(s,'High-frequency responses also contain ribs and noise. They are not lesion boundaries by definition.',672,541,519,85,23,{color:gray});
 notes(s,'V4-B implements efficient mixtures over a fixed normalized filter bank rather than a full CARAFE/unfold implementation. Mixture weights vary over space and groups. The ALPF strength is in (0,0.50), and AHPF uses a floor 0.02 with maximum 0.30 under the current configuration. Frequency response is not proof of pathology.',[`${github}/blob/2fc6ab5c8e4662d741fd8b994e55b780391948ac/nets/eppa.py`]);
}
{
 const s=setup(8,'Earlier adaptive designs narrow the current search','Research history','EPPA');
 const cols=[['V4-B: retained','Adaptive low- and high-pass filter mixtures. This mechanism already belongs to EPPA.'],['V4-C / D / F','Semantic resampling, pixel-to-text routing and soft prototypes have already been explored.'],['V4-E / G / H','Reliability scaling and text-conditioned frequency routing did not improve the historical V4-B result.']];
 cols.forEach((r,i)=>{let x=67+i*397;box(s,'',x,188,358,260);text(s,r[0],x+24,215,310,57,27,{bold:true});text(s,r[1],x+24,286,310,143,23);});
 band(s,'The current candidate changes the spatial detail retained by visual guides.',497,87);
 note(s,'Historical V4 runs use a different protocol and calibrated thresholds. Their scores are kept separate.');
 notes(s,'Historical Test IoU: V4-B .760273, C .759352, D .754129, E .759103, F .756235, G .757653, H .754253. These cannot be mixed with formal 150-epoch or current 80-epoch results. No claim that the full fusion design space is exhausted.',[`${github}/blob/docs/experiment-tracker/repro_archive/20260907/research_20260907/研究建议_历史复核版.md`]);
}
{
 const s=setup(9,'EPPA improves IoU in two formal paired ablations','Completed formal Test results','EPPA');
 table(s,[['ID / fusion','Test Dice','Test IoU'],...['A0','A2','A1','A4'].map(id=>[id+' / '+(['A2','A4'].includes(id)?'EPPA':'PLAM'),pct(F[id].dice)+'%',pct(F[id].iou)+'%'])],67,190,639,300,[280,180,179],23);
 chart(s,['Dice/BCE','Dice/Focal'],[(F.A2.iou-F.A0.iou)*100,(F.A4.iou-F.A1.iou)*100],738,184,474,318,{min:0,max:1.8,format:'0.00',title:'EPPA gain in Test IoU (pp)'});
 text(s,'A0 / A2 use Dice/BCE. A1 / A4 use Dice/Focal.\nAll four use LoRA, 150 epochs and seed 1219.',67,507,1145,61,22,{color:gray});
 band(s,'Historical best at threshold 0.5: A4, Test IoU 75.991% and Dice 84.493%.',585,62);
 notes(s,'Test macro metrics, n=2113, fixed threshold .5. A2-A0 IoU +.014620 (=1.4620 pp), A4-A1 +.012783 (=1.2783 pp). Dice gains .012529 and .009699. These are observed paired single-seed ablations. They do not establish external superiority or multi-seed stability.',[tracker]);
}
{
 const s=setup(10,'Recent experiments isolate why a candidate helps','Current research questions','EVID');
 const cols=[['01','Does routing add value?','Compare C8 auxiliary supervision alone against P10 with the same supervision plus routing.'],['02','Does a small gain persist?','Separate a case-bootstrap interval from variability across independently trained seeds.'],['03','Can finer visual guides help?','Test P11 against C4 while keeping EPPA, text encoding and the segmentation loss fixed.']];
 cols.forEach((r,i)=>{const x=67+i*397;box(s,'',x,184,358,351);text(s,r[0],x+25,209,305,56,44,{serif:true,color:'#A5A5A5',bold:true});text(s,r[1],x+25,280,305,82,28,{bold:true});text(s,r[2],x+25,379,305,136,23);});
 note(s,'A working branch and a positive validation score are insufficient evidence of a successful second contribution.');
 notes(s,'Questions summarize the current evidence requirements. C8/P10 and C4/P11 isolate different mechanisms. Test inspection has occurred repeatedly, so recent comparisons are exploratory.',[tracker]);
}
{
 const s=setup(11,'RACE-PE routing has not shown an additional Test gain','Completed 80-epoch exploratory Test results','EVID');
 const labels={'C4':'EPPA control','P9':'RACE-PE V1','C8':'Auxiliary supervision','P10':'Auxiliary + V2 routing'};
 table(s,[['Experiment','Dice','IoU','Precision','Recall'],...['C4','P9','C8','P10'].map(id=>[`${id}: ${labels[id]}`,pct(R[id].macro_dice)+'%',pct(R[id].macro_iou)+'%',pct(R[id].macro_precision)+'%',pct(R[id].macro_recall)+'%'])],67,184,1146,305,[426,180,180,180,180],23);
 paras(s,['P9 vs C4: IoU −0.105 pp, with lower precision and higher recall.','P10 vs C8: IoU −0.052 pp. Routing lowers precision by 1.092 pp.'],67,523,1146,53,24);
 notes(s,'n=2113, threshold=.5, seed1219, 80 epochs, checkpoints selected by validation IoU. P9 and P10 IoU difference intervals include zero; do not claim statistically significant IoU degradation. Precision reductions have negative paired case-bootstrap intervals. Routing does not have evidence of additional gain.',[`${github}/blob/docs/experiment-tracker/repro_archive/20260907/race_pe_test_20260907/c4_vs_p9_test.json`,`${github}/blob/docs/experiment-tracker/repro_archive/20260907/race_pe_v2_test_20260907/c8_vs_p10_test.json`]);
}
{
 const s=setup(12,'The auxiliary-supervision gain remains uncertain','Test uncertainty and attribution','EVID');
 chart(s,['C4 control','C8 aux only','P10 aux + route'],['C4','C8','P10'].map(id=>R[id].macro_iou),67,179,634,353,{min:0.75,max:0.76,format:'0.00%',title:'Test macro IoU'});
 text(s,'C8 vs C4',761,189,432,41,28,{bold:true});
 text(s,'+0.222 pp IoU',761,250,440,64,41,{serif:true,bold:true});
 text(s,'95% paired case-bootstrap interval:\n−0.000447 to +0.446301 pp\nThe interval includes zero.',761,333,440,122,24);
 text(s,'A single training seed cannot establish a stable gain across runs.',761,477,440,77,24);
 band(s,'Auxiliary supervision is not claimed as the second architecture contribution.',577,67);
 notes(s,'C8-C4 exact Test IoU difference .002215500198590665. Paired image-bootstrap CI [-.000004467094779456424,.0044630123051547]. Multiplication by100 gives [-.0004467094779456424,.44630123051547] percentage points. Case bootstrap measures sampling uncertainty for this fitted pair, not training-seed variability. Chart uses a truncated axis to show the small differences.',[`${github}/blob/docs/experiment-tracker/repro_archive/20260907/race_pe_v2_test_20260907/c4_vs_c8_test.json`]);
}
{
 const s=setup(13,'Small-area cases expose substantial false positives','Validation-only diagnosis','EVID');
 table(s,[['GT area quartile','Images','IoU','Precision','Recall'],...data.val_groups.map((g,i)=>[i===0?'Smallest 25%':i===3?'Largest 25%':`Quartile ${i+1}`,String(g.n),pct(g.iou)+'%',pct(g.precision)+'%',pct(g.recall)+'%'])],67,191,1146,290,[386,160,200,200,200],23);
 text(s,'In the smallest group, mean FP ≈ 672 pixels and mean FN ≈ 244 pixels.',67,515,1146,43,28,{bold:true});
 text(s,'Finer guides should improve foreground precision as well as preserve missed detail.',67,572,1146,53,25);
 notes(s,'C4 validation n=1429, threshold .5. Groups sort by total GT lesion pixels per image, not connected-lesion diameter. Smallest group range202–2075 pixels. Recomputed quartiles directly from archived per-image JSON. The approximate FP/FN counts derive from per-image precision/recall and GT area. GT is used only for diagnosis. These data do not establish that token granularity causes error.',[`${github}/blob/docs/experiment-tracker/repro_archive/20260907/race_pe_v2_results_20260907/c4_validation.json`]);
}
{
 const s=setup(14,'P11 adds finer visual queries to shallow semantic guides','Candidate 2: dual-grain visual guides','P11');
 const a=box(s,'Shallow CNN\nfeatures x1 / x2',67,185,250,91),b=box(s,'Average pooling\n28 × 28 fine queries',419,185,325,91),c=box(s,'Existing 14 × 14\nsemantic keys / values',888,185,325,91);
 text(s,'+30,016\nparameters',67,362,250,108,31,{bold:true,serif:true});
 const e=box(s,'Fine queries +\nattention context',419,365,325,91),f=box(s,'Local residual +\noutput projection',888,365,325,91),g=box(s,'Resize and add to PLAM1 / PLAM2, then use the unchanged EPPA decoder',194,552,891,71,{dark:true,size:24});
 connect(s,a,b);connect(s,b,e,'bottom','top');connect(s,c,e,'bottom','top');connect(s,e,f);connect(s,f,g,'bottom','right');
 notes(s,'Implemented candidate, not a verified gain or an established novel dual-scale-attention method. The CNN already has detail; the candidate adds spatial distinctions to semantic guides. Stage1 pool stride8, stage2 stride4. Hidden width32,4 attention heads. Only the output projection starts at zero, and a CPU RNG fork preserves baseline common initialization. Added parameters30016. Existing EPPA, text encoder and loss are unchanged.',[`${github}/blob/2fc6ab5c8e4662d741fd8b994e55b780391948ac/nets/dual_grain.py`,`${github}/blob/2fc6ab5c8e4662d741fd8b994e55b780391948ac/docs/P11_DUAL_GRAIN.md`]);
}
{
 const s=setup(15,'The P11 comparison preserves the baseline controls','Reproducibility','P11');
 const panels=[['Identical starting prediction','All shared initial weights and the RNG state match C4. The added output projection starts at zero.'],['A branch that can learn','Preflight verifies nonzero gradients, within-patch variation and use of coarse context after updates.'],['Deterministic real-batch checks','Two independent CUDA checks on a Train batch of 16 reproduce the same output hash.'],['Traceable runs and artifacts','Source commits, experiment tags, runtime metadata and result JSONs identify each experiment.']];
 panels.forEach((r,i)=>{let x=67+(i%2)*591,y=184+Math.floor(i/2)*192;box(s,'',x,y,554,166);text(s,r[0],x+25,y+20,504,44,27,{bold:true});text(s,r[1],x+25,y+72,504,83,23);});
 band(s,'Preflight validates implementation behavior. Test evaluation decides performance.',587,61);
 notes(s,'P11 source2fc6ab5c8e4662d741fd8b994e55b780391948ac. Both preflight JSONs statusok. Peak allocated15.7254GiB, reserved16.5566GiB. Technical checks and short preflight losses are not experiment performance results.',[`${github}/blob/docs/experiment-tracker/repro_archive/20260907/dual_grain_preflight/first.json`,`${github}/blob/docs/experiment-tracker/repro_archive/20260907/dual_grain_preflight/repeat.json`]);
}
{
 const s=setup(16,'P11 has started, with its Test result still pending','Status as of 7 September 2026','P11');
 text(s,'Latest verified snapshot: epoch 40 of 80, 19:27 Australia/Sydney.',67,176,1146,52,27,{bold:true});
 box(s,'',67,254,554,258);box(s,'',658,254,554,258);
 text(s,'Locked experiment',92,276,502,48,29,{bold:true});
 text(s,'Train from scratch, seed 1219, batch 16.\nFrozen CXR-BERT, EPPA and Dice/Focal.\nNo added auxiliary or boundary loss.\nBest checkpoint by Val macro IoU.',92,343,502,144,24);
 text(s,'After training',683,276,502,48,29,{bold:true});
 text(s,'Export best-checkpoint Val and Test results.\nCompare Test macro IoU / Dice with C4.\nInspect precision and small-area behavior.\nA positive result requires paired-seed follow-up.',683,343,502,144,24);
 band(s,'No verified P11 IoU gain is available in this presentation snapshot.',557,79);
 notes(s,'Status is taken from the existing midpoint record at2026-09-07T09:27:04Z. This deck update does not create a third monitoring inspection or connect to a live training session. P11 has80epochs, seed1219, automatic Test evaluation only after successful training. The user requested prediction-based monitoring, with a scheduled final check.',[`${github}/blob/docs/experiment-tracker/repro_archive/20260907/dual_grain_preflight/monitor_schedule.json`,`${github}/blob/2fc6ab5c8e4662d741fd8b994e55b780391948ac/experiment_manifests/p11_dual_grain.json`]);
}
{
 const s=setup(17,'EPPA is supported, with the second contribution still open','Conclusions');
 const rows=[['C1','EPPA improves Test IoU in two matched formal ablations. The best historical A4 run reaches 75.991% IoU at threshold 0.5.'],['C2','P11 tests a complementary visual-guide mechanism. The implementation passes preflight, but performance evidence is pending.'],['E','RACE-PE results separate supervision effects from routing. Current evidence does not support routing as the second contribution.']];
 rows.forEach((r,i)=>{let y=176+i*132;box(s,r[0],67,y,79,79,{dark:true,size:30});text(s,r[1],178,y+5,1005,99,27);});
 text(s,'Next evidence',67,583,230,38,27,{bold:true});
 text(s,'P11 Test comparison, paired seeds, and a baseline / EPPA / candidate / combined ablation.',325,582,887,61,24);
 notes(s,'Observed single-seed gains support continued investigation of EPPA. Do not claim stable multi-seed improvement, SOTA or a completed second innovation. Broader patient-independent and external-cohort validation remains future work.',[tracker]);
}
{
 const s=p.slides.items[17];notes(s,'Main presentation ends here. Backup material contains complete formal results, recent mechanism comparisons, formulas and provenance.',[tracker]);
 const d=p.slides.items[18];const sub=p.resolve(snapshot.find(r=>r.slide===19&&r.name==='Text 1').id);sub.text='B1–B9: results, mechanisms, protocol and evidence';sub.text.style={typeface:'Calibri',fontSize:24,color:'#BEBEBE'};
}
{
 const s=setup(20,'All completed formal Test ablations','Backup: formal 150-epoch experiments','B1');
 const labels={'B0':'Frozen / PLAM / BCE','A0':'LoRA / PLAM / BCE','A1':'LoRA / PLAM / Focal','A2':'LoRA / EPPA / BCE','A3':'LoRA / FMISeg-adapted / BCE','A4':'LoRA / EPPA / Focal','A5':'LoRA / FMISeg-adapted / Focal','A6':'Frozen / TCSR V1 / PLAM / BCE','A7':'Frozen / TCSR V1 / EPPA / BCE','A8':'Frozen / TCSR V2 / EPPA / Focal','A9':'Frozen / EPPA / Focal'};
 table(s,[['ID','Configuration (all include Dice)','Best epoch','Dice','IoU'],...data.formal.map(r=>[r.id,labels[r.id],String(r.best),pct(r.dice)+'%',pct(r.iou)+'%'])],67,165,1146,438,[75,531,160,190,190],20).cells.block({row:0,column:0,rowCount:12,columnCount:5}).assign({margins:{left:10,right:8,top:3,bottom:3}});
 note(s,'Test macro, n = 2,113, threshold 0.5, seed 1219. FMISeg-adapted is an internal module adaptation.',619);
 notes(s,'Complete formal table from the project tracker, threshold.5 only. Alternate validation-calibrated thresholds remain in the tracker. These are internal ablations, not full external-method reproductions. All11runs150epochs.',[tracker]);
}
{
 const s=setup(21,'The routing comparison separates precision from recall','Backup: recent Test contrasts','B2');
 const contrasts=[['P9 − C4',R.P9.macro_iou-R.C4.macro_iou,R.P9.macro_precision-R.C4.macro_precision,R.P9.macro_recall-R.C4.macro_recall],['C8 − C4',R.C8.macro_iou-R.C4.macro_iou,R.C8.macro_precision-R.C4.macro_precision,R.C8.macro_recall-R.C4.macro_recall],['P10 − C8',R.P10.macro_iou-R.C8.macro_iou,R.P10.macro_precision-R.C8.macro_precision,R.P10.macro_recall-R.C8.macro_recall]];
 table(s,[['Contrast','Δ IoU (pp)','Δ Precision (pp)','Δ Recall (pp)'],...contrasts.map(r=>[r[0],...r.slice(1).map(pp)])],67,185,1146,280,[390,252,252,252],24);
 paras(s,['The routing arms increase recall while reducing precision.','C8 isolates auxiliary supervision. P10 must exceed C8 to attribute a gain to routing.','All three IoU contrast intervals include zero. Single-seed evidence cannot establish stability.'],67,500,1146,51,24);
 notes(s,'All differences computed from the archived full-precision macro metrics, multiplied by100 to express percentage points. Test2113,80epochs,seed1219,threshold.5.',[tracker]);
}
{
 const s=setup(22,'ALPF / AHPF use normalized filter-bank mixtures','Backup: implemented V4-B formulation','B3');
 text(s,'K = {identity, binomial 3 × 3, binomial 5 × 5}',67,181,1146,55,31,{serif:true,bold:true});
 const lines=[['Mixture','w(g, x, k) = softmax over k of predicted filter logits'],['Adaptive filter','F(X) = Σₖ w(g, x, k) · (Kₖ * X)'],['Decoder ALPF','D′ = D + α · [F_low(D) − D]'],['Skip AHPF','ΔS = β · [S − F_high(S)]']];
 table(s,[['Operation','Expression'],...lines],67,253,1146,288,[280,866],26);
 text(s,'Weights depend on location x and channel group g. Low and high paths predict separate mixtures.',67,570,1146,62,24);
 notes(s,'Expressions summarize SpatialAdaptiveFrequencyRefiner, not the complete EPPA block. The mixture is normalized over the three candidate filters for every group and location. Context combines projected low-frequency skip, PLAM and decoder features with GroupNorm and SiLU. Bounded strengths are scalar learned values in the current implementation.',[`${github}/blob/2fc6ab5c8e4662d741fd8b994e55b780391948ac/nets/eppa.py`]);
}
{
 const s=setup(23,'Earlier refiners did not pass their validation gates','Backup: negative mechanism evidence','B4');
 table(s,[['Pilot / control','Mechanism','Val ΔIoU','Decision'],['P5 / C0','TCSR V2.5 + local supervision','−0.002072','Stop'],['P6 / C1','BCDH-R dual-head correction','+0.000152','Stop'],['P7 / C2','Sparse, balanced CDRR correction','−0.004911','Stop']],67,188,1146,273,[235,491,210,210],22);
 paras(s,['P6 did not improve its full acceptance criteria. Its residual largely became global negative confidence adjustment.','P7 constrained support and residual balance, but lost IoU and small-area recall.','These pilots did not access Test. P5 used boundary supervision and is excluded from the final method.'],67,491,1146,52,23);
 notes(s,'40-epoch validation-only pairs, n1429, threshold.5. P5 uses Dice/Tversky and must only be compared with C0. P6/P7 use separate matched Dice/Focal controls. The table reports within-pair changes, not a cross-protocol ranking. P6 exact IoU delta rounds .000152.',[tracker]);
}
{
 const s=setup(24,'Threshold and checkpoint selection precede Test evaluation','Backup: metric and selection protocol','B5');
 paras(s,['Current 80-epoch runs select the best checkpoint by Val macro IoU and keep threshold 0.5.','Formal historical results also retain optional thresholds selected on validation. Both operating points must be labeled.','Test metrics report the selected checkpoint. Test data does not enter gradients or threshold selection.'],67,190,532,128,24);
 table(s,[['A4 operating point','Dice','IoU'],['Fixed 0.5','84.493%','75.991%'],['Val-selected 0.516','84.519%','76.045%']],658,199,554,258,[254,150,150],23);
 text(s,'macro IoU = mean of per-image IoU\nmacro Dice = mean of per-image Dice',658,506,554,91,26,{serif:true});
 notes(s,'Do not compare macro scores against micro scores. Do not compute macro IoU by transforming macro Dice: nonlinear per-image aggregation prevents this shortcut. Historical run selection criteria are preserved and may differ from the current80-epoch rule. Repeated exploratory Test access is disclosed separately.',[tracker]);
}
{
 const s=setup(25,'The current scores use the project’s fixed image splits','Backup: dataset and interpretation','B6');
 table(s,[['QaTa-COV19-v2 split','Images','Role'],['Train','5,716','Parameter updates'],['Validation','1,429','Checkpoint selection and diagnosis'],['Test','2,113','Reported final-checkpoint metrics']],67,189,1146,269,[386,220,540],24);
 paras(s,['Inputs are resized to 224 × 224. Metrics use per-image macro aggregation.','Current 80-epoch comparisons remain exploratory after repeated Test inspections.','These results alone do not establish generalization to independent patients, cohorts or hospitals.'],67,491,1146,51,24);
 notes(s,'This replaces the earlier claim that a grouped modality-dropout experiment is the current main protocol. Historical patient-identification and grouping work is distinct from these fixed image-split results. A complete patient-independent benchmark and external validation remain limitations and future work.',[tracker,`${github}/blob/2fc6ab5c8e4662d741fd8b994e55b780391948ac/experiment_manifests/p11_dual_grain.json`]);
}
{
 const s=setup(26,'Every result maps to a frozen source revision','Backup: experiment provenance','B7');
 const ids=['A4','C4','P9','C8','P10','P11'];
 const sha=id=>id==='P11'?data.runtime.source_git_commit:(F[id]?.sha??R[id].checkpoint_git_commit);
 table(s,[['Experiment','Full source Git commit'],...ids.map(id=>[id,sha(id)])],67,178,1146,336,[220,926],23);
 text(s,'Code and experiment records',67,554,425,32,24,{bold:true});
 text(s,'github.com/razaxq/BetterLViT',521,554,690,32,24);
 text(s,'Verified model artifact storage',67,602,425,32,24,{bold:true});
 text(s,'huggingface.co/buckets/razaxq/BetterLViT',521,602,690,32,24);
 notes(s,'Full source hashes come from the formal tracker and archived Test/runtime JSONs. Each run has an experiment-specific tag. C4/P9/C8/P10 artifacts uploaded and independently verified by file size and Xet hash. Do not claim P11 artifacts have completed upload or final evaluation.',[github,'https://huggingface.co/buckets/razaxq/BetterLViT',tracker]);
}
{
 const s=setup(27,'P11 keeps the training budget and loss fixed','Backup: P11 runtime configuration','B8');
 table(s,[['Item','Value'],['Training','80 epochs from scratch, batch 16, seed 1219'],['Optimizer / schedule','Adam, LR 3e−4, weight decay 1e−4, inherited cosine'],['Loss','0.5 Dice + 0.5 Focal; auxiliary / boundary terms = 0'],['Text encoder','Frozen CXR-BERT, 32 text tokens, LoRA disabled'],['Visual branch','14 × 14 coarse and 28 × 28 fine grids at stages 1 / 2'],['Added parameters','30,016'],['Hardware / software','RTX 4090 D, PyTorch 2.9.1+cu128, CUDA 12.8'],['Preflight peak memory','15.725 GiB allocated; 16.557 GiB reserved']],67,170,1146,423,[326,820],23);
 note(s,'Preflight memory is measured on a real Train batch of 16. It is not a full-run runtime benchmark.',619);
 notes(s,'P11 manifest and runtime metadata. Python3.12.3. The inherited cosine schedule matches C4. No completed wall-clock duration is available in this snapshot. Do not present ETA or two-step preflight timings as completed training duration.',[`${github}/blob/2fc6ab5c8e4662d741fd8b994e55b780391948ac/experiment_manifests/p11_dual_grain.json`,`${github}/blob/docs/experiment-tracker/repro_archive/20260907/dual_grain_preflight/runtime.json`]);
}
{
 const s=setup(28,'The next decision needs both performance and attribution','Backup: planned evidence','B9');
 const rows=[['Performance','P11 vs C4 Test macro IoU / Dice, precision / recall and matched case-bootstrap intervals.'],['Stability','At least three paired training seeds, reported individually and as mean with variability.'],['Attribution','Baseline, EPPA, candidate, and EPPA + candidate. Add a comparable-cost convolution or interpolation control.'],['Failure analysis','Use fixed validation examples and area groups to test whether finer guides reduce false positives.']];
 table(s,[['Question','Evidence still needed'],...rows],67,184,1146,361,[245,901],24);
 text(s,'These are planned comparisons. None is presented as a completed P11 result.',67,587,1146,53,27,{bold:true});
 notes(s,'Prior research review considers dual-scale and token-granularity methods established precedents. Merely naming a dual-grain attention branch is insufficient for originality. Advance the mechanism only with actual benefit over appropriate controls. No fabricated qualitative cases are used in place of missing matched prediction exports.',[`${github}/blob/docs/experiment-tracker/repro_archive/20260907/research_20260907/研究建议_历史复核版.md`,`${github}/blob/2fc6ab5c8e4662d741fd8b994e55b780391948ac/docs/P11_DUAL_GRAIN.md`]);
}

await fs.writeFile(path.join(dir,'updated-inspect.ndjson'),(await p.inspect({kind:'slide,textbox,table,chart,notes',maxChars:400000})).ndjson);
const candidate=path.join(dir,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<p.slides.items.length;i++){
 await fs.writeFile(path.join(dir,`updated-${i+1}.png`),new Uint8Array(await(await p.slides.items[i].export({format:'png',scale:1})).arrayBuffer()));
 await fs.writeFile(path.join(dir,`updated-${i+1}.layout.json`),await(await p.slides.items[i].export({format:'layout'})).text());
}
if(!process.argv.includes('--draft-only')){
 const finalPath=path.join(workspace,'output','BetterLViT_20260907_updated_v2.pptx');
 const res=await finalizePresentation({workspaceDir:workspace,candidatePath:candidate,finalPath,pythonExecutable:runtimePython,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[5,9,11,13,20,21,22,23,24,25,26,27,28].flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:28,requiredNativeTableOwnerSlides:[5,9,11,13,20,21,22,23,24,25,26,27,28],requiredNativeChartOwnerSlides:[9,12],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'reference',families:['Cambria','Calibri'],referencePath:source,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(source)).digest('hex')},verifyArtifactToolImport:true,receiptPath:path.join(dir,'validation_v2.json')});
 console.log(JSON.stringify(res,null,2));
}
console.log('Rendered 28 updated slides.');
