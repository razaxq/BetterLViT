import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL,fileURLToPath} from 'node:url';
const HERE=path.dirname(fileURLToPath(import.meta.url));
import {FileBlob,PresentationFile} from '@oai/artifact-tool';

const ROOT=process.env.EVIDENCE_ROOT||path.resolve(HERE,'../../..');
const SOURCE=process.env.SOURCE_PPTX||path.join(HERE,'source_20260915.pptx');
const WORK=process.env.OUTPUT_ROOT||path.join(HERE,'.build-output');
const BUILD=path.join(WORK,'build'), OUT=path.join(WORK,'output');
const SKILL=process.env.PRESENTATIONS_SKILL||'C:/Users/dtftn/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const RUNTIME=process.env.CODEX_DEPENDENCIES||'C:/Users/dtftn/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=path.join(RUNTIME,'node/node_modules');
await fs.mkdir(BUILD,{recursive:true});await fs.mkdir(OUT,{recursive:true});
const read=async f=>JSON.parse(await fs.readFile(path.join(ROOT,'docs/results',f),'utf8'));
const primary=await read('stage1_overall_20260915/FINAL_RESULTS.json');
const external=await read('text_guided_baselines_20260920/FINAL_RESULTS.json');
const seeds=[1219,2027,3407];
const names=['LViT-PLAM','FSDR only','PLAM + complete RACE','FSDR + complete RACE'];
const rows=g=>primary.rows.filter(r=>r.group===`J${g}`).sort((a,b)=>a.seed-b.seed);
const avg=a=>a.reduce((a,b)=>a+b,0)/a.length;
const stats=a=>({mean:avg(a),sd:Math.sqrt(a.reduce((s,v)=>s+(v-avg(a))**2,0)/(a.length-1))});
const metric=(g,k)=>stats(rows(g).map(r=>r[k]*100));
const fmt=v=>v.toFixed(4),signed=v=>(v>=0?'+':'')+fmt(v);
const ms=(g,k)=>{const s=metric(g,k);return `${fmt(s.mean)} ± ${fmt(s.sd)}`};
const delta=(a,b)=>rows(a).map((r,i)=>(r.test_iou-rows(b)[i].test_iou)*100);
if(primary.rows.length!==12||primary.status!=='complete')throw Error('Primary evidence incomplete');
if(Math.abs(metric(3,'test_iou').mean-76.2262)>0.00005)throw Error('Primary result changed: review narrative');
const p=await PresentationFile.importPptx(await FileBlob.load(SOURCE));
if(p.slides.items.length!==21)throw Error('Source slide count changed');
const snap=(await p.inspect({kind:'slide,textbox,shape,image,table,chart,notes',maxChars:400000})).ndjson;
const records=snap.trim().split('\n').map(s=>JSON.parse(s));
const byslide=(n,kind)=>records.filter(r=>r.slide===n&&r.kind===kind);
const slide=n=>p.slides.items[n-1];
const text=(s,v,x,y,w,h,size=24,bold=false,color='#111111',family='Calibri',align='left',fill='none')=>{
 const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0}});
 a.text=v;a.text.style={typeface:family,fontSize:size,bold,color,alignment:align,verticalAlignment:'middle',wrap:'square',autoFit:'none',insets:{left:0,right:0,top:0,bottom:0}};return a;
};
const edit=(id,v,size)=>{const a=p.resolve(id);const rec=records.find(r=>r.id===id);const title=rec?.bbox?.[1]===38,foot=rec?.bbox?.[1]>=670;const blackCallout=id==='sh/ove9o7yd';a.text=v;a.text.style={typeface:title?'Cambria':'Calibri',fontSize:size||(title?40:foot?12:24),bold:title||blackCallout,color:blackCallout?'#FFFFFF':id==='sh/65g3298r'?'#BBBBBB':foot?'#999999':'#111111',autoFit:'none',verticalAlignment:'middle'};return a;};
const clear=n=>{const s=slide(n);for(const c of [s.shapes,s.tables,s.images,s.charts])for(const a of [...c.items])c.deleteById(a.id);return s;};
const base=(n,title,footer)=>{const s=clear(n);s.background.fill='#FFFFFF';text(s,title,67.2,38,1146,100,40,true,'#111111','Cambria');text(s,footer,67,679,1040,20,12,false,'#999999');text(s,String(n),1170,679,43,20,12,false,'#999999','Calibri','right');return s;};
const table=(s,values,widths,y=180,h=350,size=26,boldRows=[])=>{
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:67,top:y,width:1146,height:h,columnWidths:widths,values});
 t.borders.assign({fill:'#111111',width:1,style:'solid'});
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({margins:{left:12,right:10,top:values.length>10?1:5,bottom:values.length>10?1:5},anchor:'center'});
 for(let r=0;r<values.length;r++){t.rows[r].height=h/values.length;for(let c=0;c<values[r].length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?'#111111':r%2===0?'#F3F3F3':'#FFFFFF';cell.text.style={typeface:'Calibri',fontSize:size,bold:r===0||boldRows.includes(r),color:r===0?'#FFFFFF':'#111111',autoFit:'none',verticalAlignment:'middle'};
 }}return t;
};
const mainSource='docs/results/stage1_overall_20260915/FINAL_REPORT.md\ndocs/results/stage1_overall_20260915/FINAL_RESULTS.json';
const notes={};
const note=(n,en,cn,sources=mainSource)=>{const v=`English speaker script\n${en}\n\n中文讲稿\n${cn}\n\nSources (snapshot: 22 September 2026)\n${sources}`;slide(n).speakerNotes.textFrame.setText(v);notes[n]=v;};

edit('sh/65g3298r','Final defence · Updated 22 September 2026');
edit('sh/x4r21kru','Separates semantic guidance and spatial detail in LViT skip fusion.\nMean Test IoU gain over matched PLAM: +0.5058 pp (three seeds).',25);
edit('sh/q5wjelsz','Combines learned report cues, anatomical regions and visual evidence.\nComplete RACE adds +0.2428 pp Test IoU to FSDR (three seeds).',25);
edit('sh/ove9o7yd','FSDR + complete RACE: 76.2262% Test IoU, +0.7487 pp over matched PLAM',25);
note(1,'Today I present BetterLViT for text-guided infection segmentation in chest X-rays. The two contributions are FSDR and complete RACE. The main evidence now includes four configurations and three matched training seeds, with all headline results measured on the Test split.','本次答辩介绍 BetterLViT 胸片感染分割。两项贡献是 FSDR 和完整 RACE。主证据已更新为四组配置、三个匹配训练种子，主要成绩均来自测试集。','docs/PAPER_RESULTS.md');
note(2,'FSDR separates semantic refinement from spatial detail. Complete RACE combines report-based routing with its auxiliary training objectives. The three-seed mean IoU improvements are 0.5058 percentage points for FSDR over PLAM and 0.2428 points for adding complete RACE to FSDR. The combined improvement is 0.7487 points over the matched PLAM control. These are module comparisons under one shared recipe.','FSDR 将语义和细节分开精炼。完整 RACE 包括报告引导路由及配套辅助训练目标。三种子平均 IoU 增益分别为：FSDR 相对 PLAM +0.5058 个百分点，完整 RACE 加入 FSDR +0.2428 个百分点，组合相对匹配 PLAM +0.7487 个百分点。监督是完整 RACE 的组成部分，不作为独立创新点。');

// Preserve the source architecture diagrams and comparison, updating their notes.
note(3,'The main image and text branches follow LViT. RACE refines the CNN skips at four scales using report, anatomy and visual evidence. FSDR replaces the PLAM skip-fusion module and receives separate CNN, ViT, decoder and text inputs. The text encoder is frozen and the primary experiments use no LoRA.','主图像和文本分支沿用 LViT。RACE 在四个尺度上依据报告、解剖位置和视觉证据精炼 CNN 跳跃特征。FSDR 替换 PLAM 融合模块，分别接收 CNN、ViT、解码器及文本输入。主实验冻结文本编码器，不使用 LoRA。','nets/LViT.py\nnets/race_fuse.py\nnets/eppa.py\ndocs/FSDR.md');
note(4,'This comparison concerns the PLAM decoder fusion mechanism and the FSDR extension. The matched PLAM baseline uses our common training recipe and frozen CXR-BERT. It is not an unchanged reproduction of the entire official LViT training pipeline.','这里比较的是 PLAM 解码融合机制与 FSDR 扩展。匹配 PLAM 基线使用本项目统一训练配方和冻结 CXR-BERT，不等同于官方 LViT 整体流程原样复现。','nets/LViT.py\ndocs/FSDR.md\ndocs/results/plam_baseline_audit_20260916/AUDIT.md');

// Rebuild the user's old-name screenshot as an editable comparison diagram.
{
 const s=base(5,'PLAM and FSDR fusion paths','Fusion architecture');
 const box=(v,x,y,w,h=56,black=false)=>{const b=s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:black?'#111111':'#F3F3F3',line:{fill:'#BBBBBB',width:1}});b.text=v;b.text.style={typeface:'Calibri',fontSize:22,bold:true,color:black?'#FFFFFF':'#111111',alignment:'center',verticalAlignment:'middle',autoFit:'none',insets:{left:8,right:8,top:4,bottom:4}};return b;};
 const conn=(a,b,from='right',to='left')=>s.shapes.connect(a,b,{kind:'straight',fromSide:from,toSide:to,line:{fill:'#666666',width:1.7},tail:{type:'arrow',width:'med',length:'med'}});
 text(s,'PLAM: add CNN and ViT features before pixel weighting',67,145,1146,34,25,true);
 const c=box('CNN skip C',67,204,166),v=box('ViT feature V',67,276,166),add=box('C + V',288,238,130),plam=box('PLAM',474,238,150),d=box('Decoder D',474,312,150),cat=box('Concatenate',740,265,185),conv=box('Two 3 × 3\nconvolutions',1027,253,185,80);
 conn(c,add);conn(v,add);conn(add,plam);conn(plam,cat);conn(d,cat);conn(cat,conv);
 text(s,'FSDR: refine distinct sources before decoder fusion',67,403,1146,34,25,true);
 const c2=box('CNN skip C',67,452,166,45),v2=box('ViT feature V',67,507,166,45),d2=box('Decoder D',67,562,166,45),t2=box('Text CLS T',67,617,166,45);
 const f=box('FSDR\nSemantic + detail\nrefinement',341,499,264,111,true),y=box('Refined skip Y',685,480,199),dp=box('Refined decoder D′',685,583,199),cat2=box('Concatenate',947,530,144),cv2=box('3 × 3\nconv × 2',1120,519,94,80);
 for(const z of[c2,v2,d2,t2])conn(z,f);conn(f,y);conn(f,dp);conn(y,cat2);conn(dp,cat2);conn(cat2,cv2);
 note(5,'The original PLAM path first adds CNN and reconstructed ViT features. FSDR keeps the sources distinct while calculating semantic and detail corrections, then concatenates the refined skip and decoder features. Adaptive decoder refinement operates at up4 and up3. At up2 and up1, the decoder branch remains unchanged. In the combined model, RACE first refines C.','原 PLAM 先将 CNN 和 ViT 重建特征相加。FSDR 保留输入来源差异，计算语义与细节修正后，再拼接精炼跳跃特征和解码器特征。自适应解码器精炼作用于 up4、up3；up2、up1 的解码器分支保持不变。组合模型会先由 RACE 精炼 C。','nets/LViT.py\nnets/eppa.py\ndocs/FSDR.md');
}
note(6,'FSDR uses fixed Haar decomposition to separate low-frequency and high-frequency features. Low-frequency CNN, ViT and decoder features provide semantic context, conditioned by text. Semantic region support modulates the high-frequency detail branch. Bounded residual updates retain the original CNN feature. Additional adaptive filtering operates at up4 and up3.','FSDR 先用固定 Haar 分解获得低频与高频。CNN、ViT 和解码器低频提供语义上下文，并结合文本条件生成语义区域支持，用来引导高频细节分支。受限残差保留原 CNN 特征。up4、up3 额外使用自适应滤波。','docs/FSDR.md\nnets/eppa.py');
{
 const s=base(7,'Matched experimental design','Primary study: four configurations, three seeds');
 table(s,[['Setting','Shared protocol'],['Configurations','PLAM, FSDR, PLAM + RACE, FSDR + RACE'],['Training','80 epochs, batch 16, 224 × 224 images'],['Text encoder','Frozen CXR-BERT, 32 tokens, no LoRA'],['Optimisation','Adam, Dice/Focal, single-cycle cosine'],['Checkpoint and Test','Best Val macro IoU, fixed probability > 0.5']],[315,831],174,362,25);
 text(s,'Seeds 1219, 2027 and 3407 for every configuration. All 12 models are complete.',67,563,1146,36,25,true);
 text(s,'Train 5,716 / Val 1,429 / Test 2,113. Complete RACE includes routing and auxiliary training.',67,614,1146,44,23,false,'#606060');
 note(7,'All four configurations use the same data split, input size, training budget and optimisation recipe. The three seeds are paired. Eight new runs and four eligible previous runs form the 12-model comparison. The learning rate decays once from 3e-4 to 1e-6. Adam weight decay is 1e-4. Dice and Focal each have weight 0.5, with Focal gamma 2. Complete RACE has auxiliary weight 0.05. Validation macro IoU selects the checkpoint, and Test uses strict probability greater than 0.5.','四组模型采用同一数据划分、输入尺寸、训练预算与优化配方，种子一一匹配。八次新增训练加四个匹配既有模型组成十二模型整体消融。学习率单周期从 3e-4 降至 1e-6，Adam 权重衰减 1e-4，Dice/Focal 各 0.5，gamma=2。完整 RACE 辅助权重 0.05。验证集 macro IoU 选 Best，测试严格使用概率 >0.5。');
}
note(8,'RACE learns report location and count slots from frozen text features. Six anatomical zones map these cues to a spatial prior P. The CNN skip supplies visual evidence E, and slot-visual agreement A gates a bounded residual correction. The signed strength is limited to 0.15 and begins at zero. Auxiliary objectives teach the report slots and visual branches during training. The parser does not supply predictions at inference. The whole-module ablation evaluates routing and the repaired auxiliary supervision together.','RACE 从冻结文本特征学习报告位置和计数槽位，用六个解剖区域生成空间先验 P，再与视觉证据 E 和槽位视觉一致性 A 共同控制受限残差修正。强度限制为 0.15 且零初始化。辅助目标只在训练时教会相应分支，推理时解析器不提供预测。整体消融将路由和绑定修复辅助监督作为一个完整模块评价。','nets/race_fuse.py\nrace_semantics.py\n'+mainSource);
{
 const s=base(9,'Test results: complete three-seed ablation','Primary results: per-image macro metrics');
 text(s,'Mean ± sample SD across seeds 1219, 2027 and 3407',67,146,1146,34,25,false,'#606060');
 table(s,[['Model configuration','Test IoU (%)','Test Dice (%)'],...names.map((name,g)=>[name+(g===3?' (ours)':''),ms(g,'test_iou'),ms(g,'test_dice')])],[532,307,307],198,327,26,[4]);
 text(s,'Combined gain over matched PLAM: +0.7487 pp IoU and +0.6861 pp Dice.',67,557,1146,43,26,true);
 text(s,'Both modules improve the mean. These three seeds support repeatability under this protocol.',67,617,1146,40,24,false,'#606060');
 note(9,'The four configurations isolate FSDR and complete RACE at the whole-module level. The combined model averages 76.2262 percent IoU and 84.7112 percent Dice. Its gains over the matched PLAM baseline are 0.7487 and 0.6861 percentage points respectively. The displayed uncertainty is sample standard deviation across three training seeds, not a confidence interval. The historical 74.8076 percent PLAM result used a different recipe and is not the baseline for these increments.','四组配置在完整模块层面分离 FSDR 与 RACE 的效果。组合模型平均 IoU 76.2262%、Dice 84.7112%，相对匹配 PLAM 分别提高 0.7487、0.6861 个百分点。± 表示三个训练种子的样本标准差，不是置信区间。历史 PLAM 74.8076% 使用不同配方，不作为本页增量基线。');
}
{
 const s=base(10,'Paired-seed Test IoU improvements','Primary results: matched seed differences');
 const comparisons=[['FSDR replacing PLAM',1,0],['Complete RACE added to PLAM',2,0],['Complete RACE added to FSDR',3,1],['FSDR with complete RACE',3,2],['Combined model vs PLAM',3,0]];
 table(s,[['Change','1219','2027','3407','Mean (pp)'],...comparisons.map(([label,a,b])=>[label,...delta(a,b).map(signed),signed(avg(delta(a,b)))])],[506,160,160,160,160],177,362,24,[5]);
 text(s,'All five paired contrasts are positive for each of the three seeds.',67,570,1146,38,26,true);
 text(s,'This supports the two complete modules under the tested recipe. Three seeds do not establish significance.',67,616,1146,45,23,false,'#606060');
 note(10,'Each row compares models trained with the same seed. FSDR improves PLAM both with and without complete RACE. Complete RACE improves both PLAM and FSDR. Every listed paired contrast is positive on all three seeds. This does not establish a universal effect or statistical significance. The three-seed study does not split RACE routing from auxiliary supervision, and the combined result does not prove super-additive synergy.','每行都使用同种子配对。FSDR 在有无 RACE 时均改善 PLAM；完整 RACE 在 PLAM 与 FSDR 上均带来增益。列出的五种比较在三个种子上全部为正，但不据此声称普适性或统计显著。三种子研究没有拆分 RACE 路由与辅助监督，也不声称组合具有超加性协同。');
}
{
 const s=base(11,'Text-guided external comparisons','Full Test set: 2,113 images, per-image macro metrics');
 const e=external.full_test;
 table(s,[['Method','Test IoU (%)','Test Dice (%)','Evaluation source'],['MMI-UNet','75.4783','83.9707','Author checkpoint'],['GuideDecoder',fmt(e.guide.test.macro_iou*100),fmt(e.guide.test.macro_dice*100),'Project-split retraining'],['DD-CMD',fmt(e.ddcmd.test.macro_iou*100),fmt(e.ddcmd.test.macro_dice*100),'Project-split retraining'],['Ours',fmt(metric(3,'test_iou').mean),fmt(metric(3,'test_dice').mean),'Three-seed mean']],[302,230,230,384],181,335,25,[4]);
 text(s,'DD-CMD is 0.0732 pp above our mean IoU in these evaluations.',67,550,1146,40,26,true);
 text(s,'Different training recipes. External rows represent one run or checkpoint.\nMMI-UNet’s training filename list has not been independently verified.',67,603,1146,62,23,false,'#606060');
 note(11,'All rows report the same 2,113 Test images with per-image macro metrics and a strict 0.5 threshold. GuideDecoder is the LanGuideMedSeg implementation, retrained for 100 epochs with best epoch 26. DD-CMD was retrained for 160 epochs with best epoch 115. Each uses seed 1219 and its own principal training recipe. MMI-UNet uses the author checkpoint, whose training filename list was not independently verified, so overlap cannot be excluded. Ours is a three-seed mean. DD-CMD is slightly higher, by 0.0732 IoU percentage points. These are contextual external comparisons, not matched module ablations or proof of state-of-the-art performance.','各行在同一 2113 张测试图上使用逐图 macro 指标及严格 >0.5 阈值。GuideDecoder 即 LanGuideMedSeg 实现，训练 100 轮、Best 为 26；DD-CMD 训练 160 轮、Best 为 115。两者使用 seed1219 及各自主要配方。MMI 使用作者权重，其训练文件清单未独立核查，因此不能排除重叠。我们展示三种子均值。DD-CMD 的 IoU 略高 0.0732 个百分点。这是外部参照，不是模块因果消融，也不据此声称 SOTA。','docs/results/text_guided_baselines_20260920/FINAL_REPORT.md\ndocs/results/text_guided_baselines_20260920/FINAL_RESULTS.json\ndocs/results/external_methods_20260920/full_test/TEST_RESULTS.json');
}
edit('sh/ra943il8','Semantic and detail refinement improves matched PLAM by\n+0.5058 pp mean Test IoU across three seeds.',26);
edit('sh/dcbm583y','Complete RACE adds +0.2428 pp mean Test IoU to FSDR.\nRouting and auxiliary training form one contribution.',26);
edit('sh/b6d4f2lc','Ours: 76.2262% Test IoU, 84.7112% Dice (three-seed mean).\nGain over matched PLAM: +0.7487 pp IoU.',26);
edit('sh/a543mx4r','Remaining work: independent data and repeated routing-versus-auxiliary controls.',24);
note(12,'The completed primary study supports FSDR and complete RACE as two contributions under the shared experimental protocol. The combined model gains 0.7487 IoU percentage points over matched PLAM and all paired contrasts are positive across the three seeds. External DD-CMD remains slightly higher. Independent-cohort evaluation and repeated controls separating the RACE route from its auxiliary training remain future work.','已完成的主实验支持 FSDR 和完整 RACE 在统一协议下作为两项贡献。组合相对匹配 PLAM 提高 0.7487 个 IoU 百分点，三种子所有配对比较均为正。外部 DD-CMD 仍略高。后续需要独立队列验证，以及重复的路由/辅助监督机制对照。');
{
 const s=base(15,'All primary runs: Test results','Backup: all 12 primary models');
 table(s,[['Configuration','Seed','Best epoch','Test IoU (%)','Test Dice (%)'],...names.flatMap((name,g)=>rows(g).map(r=>[name,String(r.seed),String(r.best_epoch),fmt(r.test_iou*100),fmt(r.test_dice*100)]))],[435,120,171,210,210],154,470,21,[10,11,12]);
 text(s,'Each run selects Best by Val macro IoU. Every configuration uses all three seeds.',67,638,1146,26,22,false,'#606060');
 note(15,'This table retains all 12 primary results, including the weaker runs. The combined model achieves IoU values of 76.2485, 76.2562 and 76.1738 percent. Each Best epoch comes from validation selection. We report the mean over every seed rather than choosing a winning seed for the quantitative headline.','本页保留全部十二个主实验模型，包括较弱种子。组合模型三个 IoU 为 76.2485%、76.2562%、76.1738%。Best 轮次均由验证集选择，量化主结论使用全部种子均值，不挑选最优种子作为成绩。');
}
note(16,'The formula keeps the original CNN feature and adds semantic, region and detail residuals. The adaptive filter mixes identity, 3 by 3 blur and 5 by 5 blur through spatial weights. Adaptive low-pass decoder correction and high-pass skip correction operate only at up4 and up3. The implementation retains legacy eppa state-dict keys for checkpoint compatibility. Renaming the module to FSDR did not change its computation.','公式以原 CNN 特征为基础，叠加语义、区域与细节残差。自适应滤波通过空间权重混合恒等、3×3 与 5×5 模糊。解码器低通和跳跃高通修正仅在 up4、up3 启用。实现保留历史 eppa 权重键以兼容检查点，更名为 FSDR 不改变计算。','nets/eppa.py\ndocs/FSDR.md');
{
 const s=base(17,'RACE: routing and auxiliary training','Backup: matched single-cycle comparison, seed 1219');
 table(s,[['Change from FSDR','Δ Test IoU (pp)','Image-paired 95% CI'],['Auxiliary training only','+0.1915','[−0.0498, +0.4340]'],['Routing with auxiliary fixed','+0.2472','[−0.0348, +0.5211]'],['Complete RACE','+0.4387','[+0.1405, +0.7276]']],[505,286,355],181,332,25);
 text(s,'Routing has a positive point estimate, but its interval crosses zero.',67,551,1146,42,26,true);
 text(s,'Single seed. These intervals describe image-level uncertainty.\nThe three-seed main study evaluates complete RACE as one module.',67,605,1146,60,24,false,'#606060');
 const original=byslide(10,'notes')[0]?.text||'';
 note(17,'This earlier single-cycle study uses seed 1219 and the matched auxiliary-only control. It separates the routing increment from auxiliary training. The route alone has a positive point estimate but an image-paired confidence interval that crosses zero. The repeated three-seed study evaluates the complete module. We do not assign its entire gain to routing.','该单周期实验使用 seed1219 及匹配的仅辅助监督对照，用来分离路由增量。路由点估计为正，但逐图配对置信区间跨零。三种子主实验评价完整模块，不能把全部增益归因于路由。','Source: previous defence slide 10, original evidence notes follow.\n'+original);
}
edit('sh/jmd83atk','224 × 224 inputs. Per-image macro IoU and Dice. Fixed probability > 0.5.\nVal macro IoU selects Best. Test has prior research access.',24);
edit('sh/il47u5sz','Train/Val share 434 recoverable patient IDs. No recoverable Test overlap was found.\nAnonymous cases prevent a claim of fully patient-independent splits.',24);
note(19,'The dataset has 5,716 training, 1,429 validation and 2,113 Test images. The primary study selects Best by validation macro IoU and evaluates that fixed checkpoint using probability strictly greater than 0.5. The patient audit finds 434 recoverable IDs shared by Train and Val. No recoverable Test ID overlaps Train or Val, but anonymous samples prevent proving full patient independence. The Test set has prior research access, so it is not an untouched prospective evaluation.','数据划分为训练 5716、验证 1429、测试 2113。主实验由验证 macro IoU 选择 Best，再以严格 >0.5 的阈值评估该固定检查点。审计发现 Train/Val 共享 434 个可恢复患者 ID；可恢复 Test ID 未发现交叉，但匿名病例使完整患者独立性无法证明。Test 曾用于研究，不能宣称完全未接触的前瞻评估。','docs/results/plam_baseline_audit_20260916/AUDIT.md\n'+mainSource);
{
 const s=base(20,'Reproducibility of completed comparisons','Backup: implementation and result provenance');
 table(s,[['Evidence','Completed record'],['Primary module ablation','4 configurations × 3 seeds, all Val and Test results'],['Matched training protocol','Single-cycle cosine, frozen text, no LoRA, 80 epochs'],['External methods','MMI-UNet, GuideDecoder and DD-CMD full Test outputs'],['Runtime provenance','Per-run source SHA, experiment tag and checkpoint hash'],['Integrated paper snapshot','FSDR + complete RACE source and archived result JSON']],[380,766],177,417,24);
 text(s,'Archived runtime commits identify the experiments. Integration preserves their original provenance.',67,621,1146,43,23,false,'#606060');
 note(20,'The repository integrates FSDR and complete RACE source with the result snapshots. Each experiment retains its original full source commit, tag and checkpoint hash. Integration did not retrain or re-evaluate models. The primary study contains 12 single-cycle models. External comparisons have their own source and recipe metadata. Reproduction follows those per-run records.','仓库整合 FSDR 和完整 RACE 源码及结果快照。每项实验保留原始完整 SHA、标签和权重哈希，整合没有重新训练或推理。主实验为十二个单周期模型，外部对照有各自来源和配方。复现遵循每项运行记录。','docs/PAPER_RESULTS.md\ndocs/results/IMPORT_PROVENANCE.json\n'+mainSource+'\ndocs/results/text_guided_baselines_20260920/FINAL_RESULTS.json');
}
// Preserve historical diagnostics, but make their scope unmistakable.
for(const n of [21]){
 const title=byslide(n,'textbox').find(r=>r.bbox?.[1]===38);
 if(n===18)edit(title.id,'Earlier RACE: fixed-model dependence diagnostics',38);
 if(n===21)edit(title.id,'Earlier RACE: report-target binding comparison',39);
 const prior=byslide(n,'notes')[0]?.text||'';
 const prefix=n===18?'Historical diagnostic: earlier checkpoint, validation split, no retraining. These values do not describe the final three-seed model.':'Historical comparison: seed 1219 only. This binding-repair diagnostic does not replace the complete-module three-seed ablation.';
 const cn=n===18?'历史诊断：早期检查点、验证集、未重新训练；这些数值不代表最终三种子模型。':'历史比较：仅 seed1219。绑定修复诊断不替代完整模块三种子消融。';
 const v=`Scope update (22 September 2026)\n${prefix}\n${cn}\n\n${prior}`;
 slide(n).speakerNotes.textFrame.setText(v);notes[n]=v;
 if(n===21){const footer=byslide(n,'textbox').find(r=>r.text==='Backup: binding comparison');if(footer)edit(footer.id,'Backup: binding comparison, seed 1219');}
}
{
 const s=base(18,'Validation and Test under the single-cycle recipe','Backup: three-seed means');
 table(s,[['Configuration','Val IoU (%)','Test IoU (%)'],...names.map((name,g)=>[name,ms(g,'val_iou'),ms(g,'test_iou')])],[532,307,307],181,330,25,[4]);
 text(s,'Validation chooses the checkpoint. Test supplies the final reported result.',67,552,1146,44,26,true);
 text(s,'Mean ± sample SD across the same three seeds.\nThe result applies to the current split and shared training recipe.',67,608,1146,60,24,false,'#606060');
 note(18,'The combined model has the highest validation and Test means among the four matched configurations. Validation IoU selects checkpoints. The headline result is Test IoU. The standard deviations describe variation across three training seeds, and the data-split limitations still apply.','组合在四组匹配配置中具有最高验证和测试均值。验证 IoU 负责选权重，最终成绩以测试 IoU 为准。标准差表示三个训练种子之间的变化，数据划分限制仍然适用。');
}

// Relationship diagrams use rounded nodes and curved data-flow links, as requested.
const colors={bg:'#191919',node:'#333333',line:'#B8B8B8',race:'#276B80',fsdr:'#76554A',text:'#F4F4F4'};
const graph=(n,title,footer)=>{const s=clear(n);s.background.fill=colors.bg;text(s,title,67,40,1146,85,39,true,colors.text,'Cambria');text(s,footer,67,679,1090,20,12,false,'#999999');text(s,String(n),1170,679,43,20,12,false,'#999999','Calibri','right');return s;};
const node=(s,label,x,y,w,h=64,tone='node',size=23)=>{const a=s.shapes.add({geometry:'roundRect',borderRadius:13,position:{left:x,top:y,width:w,height:h},fill:colors[tone]||tone,line:{fill:'#575757',width:1}});a.text=label;a.text.style={typeface:'Calibri',fontSize:size,bold:false,color:colors.text,alignment:'center',verticalAlignment:'middle',autoFit:'none',insets:{left:10,right:10,top:5,bottom:5}};return a;};
const link=(s,a,b,from='right',to='left',kind='curved',color=colors.line)=>s.shapes.connect(a,b,{kind,fromSide:from,toSide:to,line:{fill:color,width:1.8},tail:{type:'triangle',width:'sm',length:'sm'}});
const label=(s,t,x,y,w,h=34,size=22,color='#BBBBBB')=>text(s,t,x,y,w,h,size,false,color);
{
 const s=graph(3,'FSDR and RACE within LViT','Placement: RACE on CNN skips, FSDR inside all four decoder stages');
 const report=node(s,'Report',67,164,154),bert=node(s,'Frozen CXR-BERT',275,164,237),vit=node(s,'LViT transformer\nencoder / decoder',600,150,245,90),v=node(s,'ViT reconstructions\nV₁, V₂, V₃, V₄',944,155,268,80);
 const image=node(s,'Chest X-ray',67,355,154),cnn=node(s,'CNN encoder\nC₁, C₂, C₃, C₄',275,337,237,100),race=node(s,'RACE\nRefine each CNN skip',600,340,245,90,'race'),f=node(s,'FSDR\nReplace PLAM fusion',944,340,268,90,'fsdr');
 const bottle=node(s,'CNN bottleneck C₅',275,550,237),up=node(s,'Upsample decoder D',600,550,245),fusion=node(s,'Concatenate Y and D′\nTwo 3 × 3 convolutions',944,535,268,86);
 link(s,report,bert);link(s,bert,vit);link(s,vit,v);link(s,image,cnn);link(s,cnn,vit,'top','bottom');link(s,cnn,race);link(s,race,f);link(s,v,f,'bottom','top');link(s,bert,race,'bottom','top');link(s,cnn,bottle,'bottom','top');link(s,bottle,up);link(s,up,f,'right','left');link(s,f,fusion,'bottom','top');
 label(s,'Text T also conditions FSDR',715,287,305,32,21);
 label(s,'RACE leaves the encoder and transformer paths intact.',67,472,670,35,22);
 label(s,'Decoder order: up4 → up3 → up2 → up1 → segmentation',525,631,690,30,22);
 note(3,'The original CNN and transformer paths run first. RACE then replaces each decoder-facing CNN skip C1 to C4 with its routed version. It does not feed routed features back into the encoder or transformer. FSDR replaces PLAM inside each of up4, up3, up2 and up1. It receives the routed skip, reconstructed ViT feature, upsampled decoder feature and text. It returns Y and D-prime for concatenation and two convolutions. The first decoder stage starts from bottleneck C5, and subsequent stages use the previous decoder output.','CNN 与 Transformer 主路径先完成计算，RACE 随后精炼送往解码器的 C1–C4 跳跃特征，不将路由结果反馈进编码器或 Transformer。FSDR 位于 up4、up3、up2、up1 内部，替换 PLAM，接收路由跳跃、ViT 重建、上采样解码器与文本输入，输出 Y 与 D′ 后拼接卷积。首级由瓶颈 C5 开始，之后各级使用上一级解码器输出。','nets/LViT.py:471-525\nnets/race_fuse.py\ndocs/FSDR.md');
}
{
 const s=graph(5,'Decoder fusion before and after FSDR','One decoder stage shown. The same interface repeats at up4, up3, up2 and up1.');
 label(s,'Original PLAM fusion',67,145,1000,38,27,'#FFFFFF');
 const c=node(s,'CNN skip C',67,212,174,57),v=node(s,'ViT feature V',67,300,174,57),add=node(s,'C + V',326,250,133),pl=node(s,'PLAM\nPixel weighting',525,239,204,84),d=node(s,'Decoder D',525,345,204,57),cat=node(s,'Concatenate',821,289,177),conv=node(s,'Conv 3 × 3\n× 2',1068,278,144,84);
 link(s,c,add);link(s,v,add);link(s,add,pl);link(s,pl,cat);link(s,d,cat);link(s,cat,conv);
 label(s,'FSDR preserves separate sources',67,429,900,36,27,'#FFFFFF');
 const cr=node(s,'RACE-refined C′',67,490,198,62,'race'),vf=node(s,'ViT feature V',67,591,198,57),t=node(s,'Text CLS T',332,477,194,57),de=node(s,'Upsampled D',332,588,194,57),f=node(s,'FSDR',596,537,178,75,'fsdr',27),y=node(s,'Skip Y',851,478,141,57),dp=node(s,'Decoder D′',851,591,141,57),out=node(s,'Concatenate\n+ 2 × 3 × 3 conv',1068,531,144,95,'node',22);
 link(s,cr,f);link(s,vf,f);link(s,t,f);link(s,de,f);link(s,f,y);link(s,f,dp);link(s,y,out);link(s,dp,out);
 // Source labels make the optional upstream module explicit without merging its role with FSDR.
 note(5,'The upper path is the PLAM decoder operation. The lower path shows the combined model: RACE first produces the CNN skip C-prime, and FSDR receives C-prime, V, D and text separately. FSDR outputs Y and D-prime, which feed concatenation and two 3 by 3 convolutions. FSDR-only experiments use the original C instead of C-prime. Only the deeper up4 and up3 stages adapt D; at up2 and up1, D-prime equals D.','上方为 PLAM 解码操作，下方为组合模型：RACE 先得到 C′，FSDR 分别接收 C′、V、D 和文本，输出 Y 与 D′ 后拼接并进行两层 3×3 卷积。仅 FSDR 对照直接使用原 C。只有 up4、up3 自适应精炼 D，up2、up1 的 D′=D。','nets/LViT.py:156-166\nnets/LViT.py:503-525\nnets/eppa.py');
}
{
 const s=graph(6,'FSDR: semantic guidance and detail refinement','Inside the FSDR node of each LViT decoder stage');
 const c=node(s,'CNN skip C*',67,215,154),haar=node(s,'Haar split',273,215,165),low=node(s,'C* low',486,205,164),high=node(s,'C* high',486,453,164),vd=node(s,'V, decoder D′',67,348,178),h2=node(s,'Haar low',273,348,165),sem=node(s,'Semantic refinement\nChannel + region',731,269,244,89,'fsdr'),t=node(s,'Text CLS T',486,145,164,48),detail=node(s,'Detail refinement\nLocal + context',731,449,244,84,'fsdr'),sum=node(s,'Residual sum\nRefined skip Y',1061,356,153,96,'fsdr',22),identity=node(s,'C* identity',1047,175,166,57);
 link(s,c,haar);link(s,haar,low);link(s,haar,high,'bottom','left');link(s,vd,h2);link(s,low,sem);link(s,h2,sem);link(s,t,sem,'right','top');link(s,high,detail);link(s,sem,detail,'bottom','top');link(s,sem,sum);link(s,detail,sum);link(s,identity,sum,'bottom','top');
 label(s,'Support S',874,386,143,33,21);label(s,'C* = C′ with RACE; C otherwise',67,570,660,34,23);
 label(s,'Y = C* + αp Vlow + Rchannel + αr Rregion + αd Rdetail + Radaptive',67,617,1145,39,25,'#FFFFFF');
 note(6,'C-star denotes the input CNN skip, routed when RACE is enabled. Haar decomposition separates low and high frequencies. Low-frequency skip, ViT and decoder signals plus text generate channel and region corrections and semantic support S. S modulates the high-frequency detail branch, which combines local and contextual convolutions. The final output adds bounded residuals to the original skip. Adaptive filtering, detailed in the backup slide, produces D-prime and R-adaptive at up4 and up3.','C* 表示输入跳跃特征，启用 RACE 时为 C′。Haar 分解低高频，跳跃、ViT、解码器低频与文本形成通道/区域修正和语义支持 S，S 引导由局部与上下文卷积组成的高频细节分支。输出以原跳跃为基础叠加受限残差。备份页展示的自适应滤波在 up4、up3 生成 D′ 和 R_adaptive。','nets/eppa.py:679-777\ndocs/FSDR.md');
}
{
 const s=graph(8,'RACE: report and visual evidence control the skip','One RACE route shown. Four routes act on C₁, C₂, C₃ and C₄ before decoder fusion.');
 const t=node(s,'Frozen text T',67,168,180),pool=node(s,'Masked mean',297,168,180),slots=node(s,'Learned report slots\n6 locations + count',527,153,242,95,'race'),prior=node(s,'Spatial report prior P',951,168,262,64,'race'),basis=node(s,'Six-zone basis B',602,307,229,60),c=node(s,'CNN skip C',67,422,180),e=node(s,'Visual evidence E\nConv + sigmoid',328,408,242,88,'race'),agree=node(s,'Zone agreement A',642,422,238,64,'race'),gate=node(s,'Gate G = P × E × A',951,347,262,64,'race'),r=node(s,'Residual R(C)\nDepthwise + pointwise',328,559,242,88),out=node(s,'C′ = C + s G ⊙ R(C)\nBounded skip correction',951,559,262,88,'race',22);
 link(s,t,pool);link(s,pool,slots);link(s,slots,prior);link(s,basis,prior,'right','left');link(s,slots,agree,'bottom','top');link(s,basis,agree,'bottom','top');link(s,c,e);link(s,e,agree);link(s,e,gate,'top','left','straight');link(s,agree,gate);link(s,prior,gate,'bottom','top');link(s,c,r,'bottom','left');link(s,r,out);link(s,gate,out,'bottom','top');
 label(s,'|s| ≤ 0.15; zero initial correction',620,516,590,32,22);
 label(s,'Auxiliary training teaches the slots and evidence branches. Inference uses learned predictions.',67,649,1146,28,20);
 note(8,'Masked pooling of frozen text feeds the slot head. Six location probabilities combine with the aligned anatomical basis to form P. Each CNN skip generates visual evidence E and a local residual R(C). Zone pooling compares visual evidence with report slots to produce agreement A. P, E and A jointly gate a signed correction, with strength bounded by 0.15 and initial strength zero. Complete RACE includes its auxiliary training objectives, while inference uses predicted slots rather than parser targets.','冻结文本经掩码均值池化后进入槽位头，六个位置概率与对齐解剖基底生成 P。CNN 跳跃生成视觉证据 E 和残差 R(C)，区域池化比较视觉与报告槽位得到 A。P、E、A 共同控制有符号修正，强度上限 0.15、初始为零。完整 RACE 包含辅助训练目标，但推理使用预测槽位，不使用解析器目标。','nets/race_fuse.py\nrace_semantics.py\n'+mainSource);
}
{
 const s=graph(16,'FSDR: spatially adaptive frequency refinement','Adaptive branch at up4 and up3. At up2 and up1, D′ = D and R_adaptive = 0.');
 const inputs=node(s,'C* low, V low, D low',67,158,325,70),context=node(s,'Shared spatial context',480,158,325,70,'fsdr'),wl=node(s,'Low-pass weights\nSpatial softmax',250,290,235,84,'fsdr'),wh=node(s,'High-pass weights\nSpatial softmax',848,290,235,84,'fsdr'),d=node(s,'Decoder D',67,428,145),c=node(s,'Skip C*',658,428,145),lpD=node(s,'LP_low(D)\nWeighted filter bank',250,420,235,82),lpC=node(s,'LP_high(C*)\nWeighted filter bank',848,420,235,82),outD=node(s,'D′ = D + α [LP_low(D) − D]',105,562,477,72,'fsdr',24),outC=node(s,'R_adaptive = β [C* − LP_high(C*)]',690,562,522,72,'fsdr',24);
 link(s,inputs,context);link(s,context,wl,'bottom','top');link(s,context,wh,'right','top');link(s,wl,lpD,'bottom','top');link(s,wh,lpC,'bottom','top');link(s,d,lpD);link(s,c,lpC);link(s,lpD,outD,'bottom','top');link(s,lpC,outC,'bottom','top');
 label(s,'Each filter bank mixes identity, 3 × 3 blur and 5 × 5 blur.',67,641,1146,32,23);
 note(16,'This branch operates at the deeper decoder stages up4 and up3. Low-frequency CNN, reconstructed ViT and decoder features predict separate spatial mixture weights for decoder smoothing and skip high-pass correction. Each mixture selects among identity, 3 by 3 blur and 5 by 5 blur. The decoder uses a bounded residual low-pass update. The skip receives a bounded high-pass residual, added to Y. Shallow stages retain the base Haar semantic-detail branches but bypass these adaptive corrections.','此分支位于较深的 up4、up3。CNN、ViT 重建和解码器低频共同预测两组空间混合权重，分别控制解码器平滑和跳跃高通修正。滤波组由恒等、3×3 模糊、5×5 模糊组成。解码器使用受限低通残差更新，跳跃高通残差加入 Y。浅层仍保留基本 Haar 语义/细节分支，只跳过这项自适应修正。','nets/eppa.py:107-314\nnets/eppa.py:679-777\ndocs/FSDR.md');
}
// The requested presentation name is descriptive and does not introduce a project brand.
slide(1).shapes.deleteById(p.resolve('sh/547294r6').id);
text(slide(1),'FSDR and RACE',67,182,1100,75,56,true,'#FF0000','Cambria');
text(slide(1),'Text-Guided Segmentation of COVID-19 Infections\nin Chest X-rays',67,270,1100,106,40,true,'#FFFFFF','Cambria');
for(let n=1;n<=21;n++){
 for(const a of slide(n).shapes.items){const v=a.text?.toString();if(v?.includes('BetterLViT'))a.text.replace('BetterLViT','FSDR and RACE');}
 if(notes[n]){notes[n]=notes[n].replaceAll('D:/BetterLViT/','').replaceAll('BetterLViT','FSDR and RACE');slide(n).speakerNotes.textFrame.setText(notes[n]);}
}
for(let n=1;n<=21;n++){
 for(const a of slide(n).shapes.items)if(a.position?.top>=670&&/^\d+$/.test(a.text?.toString()||''))a.text=String(n);
 if(!notes[n])notes[n]=byslide(n,'notes')[0]?.text||'';
}
await fs.writeFile(path.join(OUT,'FSDR_RACE_Defense_Speaker_Notes_20260922.md'),'# FSDR and RACE defence speaker notes\n\nUpdated 22 September 2026.\n\n'+Object.entries(notes).map(([n,t])=>`## Slide ${n}\n\n${t}`).join('\n\n'));
await fs.writeFile(path.join(BUILD,'evidence-derived.json'),JSON.stringify({source:SOURCE,primaryStats:names.map((name,g)=>({name,iou:metric(g,'test_iou'),dice:metric(g,'test_dice')})),combinedDelta:avg(delta(3,0)),slides:21},null,2));
const version=process.env.BUILD_VERSION||'v1';
const candidate=path.join(BUILD,`candidate-${version}.pptx`);
await(await PresentationFile.exportPptx(p)).save(candidate);
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const tableOwners=[4,7,9,10,11,15,17,18,19,20,21];
const finalPath=path.join(OUT,`FSDR_RACE_20260922_${version}.pptx`);
await finalizePresentation({workspaceDir:WORK,candidatePath:candidate,finalPath,pythonExecutable:path.join(RUNTIME,'python/python.exe'),integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:21,requiredNativeTableOwnerSlides:tableOwners,requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'reference',families:['Cambria','Calibri'],referencePath:SOURCE,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(SOURCE)).digest('hex')},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,`validation-${version}.json`)});
const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));
await fs.writeFile(path.join(BUILD,`final-inspect-${version}.ndjson`),(await final.inspect({kind:'slide,textbox,shape,table,notes',maxChars:400000})).ndjson);
for(let i=0;i<final.slides.items.length;i++){
 await fs.writeFile(path.join(BUILD,`final-${version}-${i+1}.png`),new Uint8Array(await(await final.slides.items[i].export({format:'png',scale:1})).arrayBuffer()));
 await fs.writeFile(path.join(BUILD,`final-${version}-${i+1}.layout.json`),await(await final.slides.items[i].export({format:'layout'})).text());
}
console.log(JSON.stringify({finalPath,slides:21,rendered:21}));

// All awaited export, validation and render operations completed successfully.
process.exit(0);
