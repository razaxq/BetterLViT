"""Build three native, editable draw.io pages using only Python's standard library.

Run: python build_diagrams.py
No raster images, external fonts, dependencies or remote assets are required.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import json

OUT = Path(__file__).resolve().parent
BLUE = ('#C3D8EF', '#4C78A8')
PURPLE = ('#CECFE8', '#797DB0')
GREEN = ('#D7E8C5', '#76964D')
ORANGE = ('#F7D7B2', '#C5833B')
GOLD = ('#F7E8AD', '#AC8C33')
GRAY = ('#F1F3F5', '#85919D')
INK = '#263747'

class Page:
    def __init__(self, name, w=1880, h=1420):
        self.name, self.w, self.h = name, w, h
        self.model = ET.Element('mxGraphModel', dict(dx='1880',dy='1420',grid='0',gridSize='10',guides='1',tooltips='1',connect='1',arrows='1',fold='1',page='1',pageScale='1',pageWidth=str(w),pageHeight=str(h),math='0',shadow='0',background='#ffffff'))
        self.root = ET.SubElement(self.model, 'root')
        ET.SubElement(self.root,'mxCell',id='0')
        ET.SubElement(self.root,'mxCell',id='1',parent='0')
        self.ids = set(['0','1']); self.edges = []; self.boxes = {}

    def node(self, id, label, x,y,w,h, palette=GRAY, kind='rect',size=22, extra=''):
        assert id not in self.ids, id
        self.ids.add(id)
        styles = {'rect':'rounded=1;arcSize=12;', 'cube':'shape=cube;size=16;boundedLbl=1;spacingTop=12;', 'port':'rounded=1;arcSize=50;', 'circle':'ellipse;aspect=fixed;', 'text':'text;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;', 'group':'rounded=1;arcSize=3;dashed=1;dashPattern=6 4;fillOpacity=35;'}
        style = f'whiteSpace=wrap;html=0;fontFamily=Arial;fontSize={size};fontColor={INK};strokeWidth=1.6;fillColor={palette[0]};strokeColor={palette[1]};spacing=8;' + styles[kind] + extra
        cell = ET.SubElement(self.root,'mxCell',id=id,value=label,style=style,vertex='1',parent='1')
        ET.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
        self.boxes[id] = (x,y,w,h)
        return id

    def text(self,id,label,x,y,w,h,size=22,extra=''):
        return self.node(id,label,x,y,w,h,kind='text',size=size,extra=extra)

    def edge(self,a,b,label='',color=INK,points=None,ex=(1,.5),en=(0,.5),dash=False,extra=''):
        id=f'e{len(self.edges)+1}'
        assert id not in self.ids
        self.ids.add(id); self.edges.append((a,b))
        style=f'edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=0;endArrow=block;endFill=1;strokeWidth=2;strokeColor={color};fontFamily=Arial;fontSize=18;fontColor={INK};labelBackgroundColor=#FFFFFF;exitX={ex[0]};exitY={ex[1]};exitDx=0;exitDy=0;entryX={en[0]};entryY={en[1]};entryDx=0;entryDy=0;'
        if dash: style+='dashed=1;dashPattern=4 4;'
        cell=ET.SubElement(self.root,'mxCell',id=id,value=label,style=style+extra,edge='1',parent='1',source=a,target=b)
        geom=ET.SubElement(cell,'mxGeometry',relative='1',attrib={'as':'geometry'})
        if points:
            arr=ET.SubElement(geom,'Array',attrib={'as':'points'})
            for x,y in points: ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))

    def title(self,title,subtitle):
        self.text('title',title,50,28,1780,58,36,'fontStyle=1;')
        self.text('subtitle',subtitle,50,90,1770,45,21,'fontColor=#5B6877;')

    def validate(self):
        for a,b in self.edges: assert a in self.ids and b in self.ids, (a,b)
        for id,(x,y,w,h) in self.boxes.items(): assert x>=0 and y>=0 and x+w<=self.w and y+h<=self.h, id
        return {'page':self.name,'vertices':len(self.boxes),'edges':len(self.edges)}


def overview():
    p=Page('01 · LViT integration',1880,1500)
    p.title('FSDR and RACE in LViT','Double-U backbone · four routed CNN skips · four frequency-aware decoder stages')
    p.node('vit_group','',50,290,590,850,BLUE,'group')
    p.node('cnn_group','',720,290,1110,850,PURPLE,'group')
    p.text('vit_title','Language-conditioned ViT branch',70,305,550,42,25,'fontStyle=1;')
    p.text('cnn_title','CNN encoder–decoder with RACE and FSDR',750,305,1050,42,25,'fontStyle=1;')
    p.node('report','Radiology report',70,170,220,70,GOLD,'rect',23)
    p.node('bert','Frozen CXR-BERT',340,160,240,90,GOLD,'cube',23)
    p.node('textT','T',620,185,55,40,GOLD,'port',23)
    p.edge('report','bert',color=GOLD[1]); p.edge('bert','textT',color=GOLD[1])
    p.text('text_note','Shared text features T condition\nViT, RACE and FSDR.',710,165,530,90,22)
    p.node('image','Chest X-ray',815,175,185,70,PURPLE,'rect',23)
    # place shared-text note without overlap with image
    p.boxes['text_note']=(1220,165,550,90)
    p.root.find("mxCell[@id='text_note']/mxGeometry").set('x','1220')
    for i,y in enumerate([395,575,755,935],1):
        p.node(f'dv{i}',f'Down ViT {i}',160,y,160,95,BLUE,'cube',22)
        p.node(f'uv{i}',f'Up ViT {i}',410,y,160,95,BLUE,'cube',22)
        p.node(f'ciport{i}',f'C{i}',75,y+27,60,40,PURPLE,'port',20)
        p.edge(f'ciport{i}',f'dv{i}',color=PURPLE[1],dash=True)
        p.edge(f'dv{i}',f'uv{i}',color=BLUE[1])
        p.node(f'tv{i}','T',210,y-48,50,30,GOLD,'port',18)
        p.edge(f'tv{i}',f'dv{i}',color=GOLD[1],ex=(.5,1),en=(.5,0),dash=True)
        p.node(f'tu{i}','T',460,y-48,50,30,GOLD,'port',18)
        p.edge(f'tu{i}',f'uv{i}',color=GOLD[1],ex=(.5,1),en=(.5,0),dash=True)
        p.node(f'c{i}',f'C{i}\nCNN encoder',820,y,180,95,PURPLE,'cube',23)
        p.node(f'race{i}',f'RACE {i}',1090,y+10,150,75,ORANGE,'cube',23)
        p.node(f'up{i}',f'up{i}\nFSDR + concat + conv',1510,y-5,260,110,GREEN,'cube',22)
        p.edge(f'c{i}',f'race{i}',color=ORANGE[1])
        p.edge(f'race{i}',f'up{i}',f'C′{i}',ORANGE[1])
        p.node(f'cond{i}','T, B',1120,y-45,90,33,GOLD,'port',18)
        p.edge(f'cond{i}',f'race{i}',color=GOLD[1],ex=(.5,1),en=(.5,0),dash=True)
        p.node(f'v{i}',f'V{i}, T',1400,y-54,115,37,BLUE,'port',19)
        p.edge(f'v{i}',f'up{i}',color=BLUE[1],ex=(1,.5),en=(.25,0),points=[(1575,y-35)],dash=True)
        p.text(f'rec{i}',f'Reconstruct → V{i}',355,y+100,250,40,19,'fontColor=#4C78A8;align=center;')
        # Reconstructed maps are named output ports to keep the two U paths legible.
        p.edge(f'uv{i}',f'rec{i}',color=BLUE[1],ex=(.5,1),en=(.5,0),dash=True)
        if i>1:
            p.edge(f'dv{i-1}',f'dv{i}',color=BLUE[1],ex=(.18,1),en=(.18,0))
            p.edge(f'uv{i}',f'uv{i-1}',color=BLUE[1],ex=(1,.3),en=(1,.7),points=[(610,y+28),(610,y-180+66)])
            p.edge(f'c{i-1}',f'c{i}',color=PURPLE[1],ex=(.5,1),en=(.5,0))
            p.edge(f'up{i}',f'up{i-1}',color=GREEN[1],ex=(.8,0),en=(.8,1))
    p.node('c5','C5 · Bottleneck',820,1060,440,65,PURPLE,'cube',23)
    p.edge('image','c1',color=PURPLE[1],ex=(.5,1),en=(.5,0))
    p.edge('c4','c5',color=PURPLE[1],ex=(.5,1),en=(.2,0))
    p.edge('c5','up4',color=GREEN[1],ex=(1,.5),en=(.8,1),points=[(1718,1092)])
    p.node('head','1 × 1 conv + sigmoid',1510,235,260,60,GRAY,'rect',21)
    p.node('mask','Segmentation mask',1510,150,260,55,GREEN,'rect',22)
    p.edge('up1','head',color=GREEN[1],ex=(.8,0),en=(.8,1))
    p.edge('head','mask',color=GREEN[1],ex=(.5,0),en=(.5,1))
    # Relocate text note so it does not collide with output.
    p.root.find("mxCell[@id='text_note']/mxGeometry").set('x','1040')
    p.root.find("mxCell[@id='text_note']/mxGeometry").set('width','420')
    p.boxes['text_note']=(1040,165,420,90)
    p.text('vit_note','Same-scale horizontal skips preserve encoder tokens.\nC1–C4 ports refer to the original CNN features.',75,1070,535,62,19)
    p.node('detail_group','',50,1170,1780,260,GREEN,'group')
    p.text('detail_title','Inside each green decoder stage',75,1184,650,40,25,'fontStyle=1;')
    p.node('prev','Previous decoder\n(or C5 at up4)',80,1265,235,80,PURPLE,'cube',21)
    p.node('upsample','Upsample',370,1272,170,65,GRAY,'rect',22)
    p.node('fsdr','FSDR',670,1260,220,90,GREEN,'cube',24)
    p.node('skipin','C′i, Vi, T',680,1215,200,32,BLUE,'port',19)
    p.node('concat','Concat [Y, D′]',1100,1272,230,65,GRAY,'rect',22)
    p.node('conv','Two 3 × 3 convs',1410,1263,260,85,PURPLE,'cube',23)
    p.edge('prev','upsample',color=GREEN[1]);p.edge('upsample','fsdr','D',GREEN[1])
    p.edge('skipin','fsdr',color=BLUE[1],ex=(.5,1),en=(.5,0),dash=True)
    p.edge('fsdr','concat','Y, D′',GREEN[1]);p.edge('concat','conv',color=GREEN[1])
    p.text('legend','Solid arrows: feature flow     Dashed arrows / matching names: conditioning or same-scale ports     B: six-zone anatomical basis',75,1370,1725,42,20)
    p.text('footer','RACE changes decoder-bound CNN skips only. FSDR replaces PLAM. Adaptive filtering is enabled in up4 / up3.',50,1440,1780,38,21,'fontColor=#5B6877;')
    return p


def fsdr():
    p=Page('02 · FSDR',1880,1590)
    p.title('FSDR · Frequency-aware Semantic–Detail Refinement','Separate CNN detail, ViT semantics and decoder context before fusion · one instance per decoder stage')
    p.node('main_group','',50,155,1780,735,GREEN,'group')
    p.text('main_title','A. Semantic and detail paths',70,165,800,45,26,'fontStyle=1;')
    p.node('c','C*\nCNN skip',90,280,145,90,PURPLE,'cube',23)
    p.node('haar','Fixed Haar\nsplit',290,285,150,80,GRAY,'rect',22)
    p.node('cl','C_low',500,280,125,65,PURPLE,'cube',21)
    p.node('ch','C_high',500,530,125,70,PURPLE,'cube',21)
    p.edge('c','haar');p.edge('haar','cl',color=PURPLE[1])
    p.edge('haar','ch',color=PURPLE[1],ex=(.5,1),points=[(365,565)])
    p.node('lowports','V_low, D′_low, T',690,225,550,38,BLUE,'port',21)
    p.node('channel','Channel modulation\nGAP / GMP → MLP + text\ng = 1 + 0.5 tanh(·)',710,300,235,130,GREEN,'rect',21)
    p.node('region','Semantic region\nProject + norm → text FiLM\nSimilarity maps → Support S',1000,300,270,130,GREEN,'rect',21)
    p.node('rsem','Semantic residual\nR_sem = αp V_low + R_channel\n+ αr R_region',1370,315,300,125,GREEN,'rect',21)
    p.node('vlsem','V_low',1450,250,140,35,BLUE,'port',20)
    p.edge('vlsem','rsem',color=BLUE[1],ex=(.5,1),en=(.5,0),dash=True)
    p.edge('cl','channel',color=GREEN[1])
    p.edge('cl','region',color=GREEN[1],ex=(.5,0),en=(.5,0),points=[(560,275),(1135,275)])
    p.edge('lowports','channel',color=BLUE[1],ex=(.2,1),en=(.5,0),dash=True)
    p.edge('lowports','region',color=BLUE[1],ex=(.8,1),en=(.75,0),dash=True)
    p.edge('channel','rsem','',GREEN[1],ex=(.5,1),en=(.2,1),points=[(827,475),(1430,475)])
    p.edge('region','rsem','R_region',GREEN[1])
    p.text('channel_formula','R_channel = C_low ⊙ (g − 1)',685,437,340,32,18)
    p.node('refine','Detail refinement\nLocal + dilated conv\nGN → SiLU → conv',720,535,250,115,ORANGE,'rect',22)
    p.node('detail','R_detail\nC_high ⊙ (0.5 + S)\n+ refine(C_high)',1100,530,260,125,ORANGE,'rect',22)
    p.edge('ch','refine',color=ORANGE[1]);p.edge('refine','detail','refine(C_high)',ORANGE[1])
    p.edge('ch','detail','C_high',ORANGE[1],ex=(.5,1),en=(.5,1),points=[(562,710),(1230,710)])
    p.edge('region','detail','',GREEN[1],ex=(.7,1),en=(.9,0),points=[(1189,450),(1334,450)],extra='jumpStyle=arc;jumpSize=8;')
    p.text('support_label','Support S',1100,487,215,32,19,'align=right;')
    p.node('sum','+',1690,565,65,65,GREEN,'circle',36)
    p.node('y','Y',1690,750,70,65,GREEN,'cube',25)
    p.edge('rsem','sum','R_sem',GREEN[1],ex=(1,.5),en=(.5,0),points=[(1722,377)])
    p.edge('detail','sum','αd R_detail',ORANGE[1])
    p.edge('c','sum','Original C* identity bypass',PURPLE[1],ex=(.5,0),en=(.8,0),points=[(162,215),(1780,215),(1780,540),(1742,540)])
    p.node('radport','R_adaptive',1400,725,200,45,ORANGE,'port',21)
    p.edge('radport','sum',color=ORANGE[1],ex=(1,.5),en=(.2,1),points=[(1640,747),(1640,665),(1703,665)],dash=True)
    p.edge('sum','y',color=GREEN[1],ex=(.5,1),en=(.5,0))
    p.text('main_note','C* = C′ with RACE; C* = C in the FSDR-only control.\nHaar low/high components are reconstructed at the input spatial resolution.',80,770,1210,85,21)
    p.text('yformula','Y = C* + R_sem + αd R_detail + R_adaptive',780,820,990,40,24,'fontStyle=1;align=right;')

    p.node('adapt_group','',50,925,1780,480,BLUE,'group')
    p.text('adapt_title','B. Spatial adaptive filtering · up4 and up3 only',70,937,1500,42,26,'fontStyle=1;')
    p.node('context','C_low, V_low, D_low\nProject → sum → GN → SiLU',90,1050,330,100,BLUE,'rect',22)
    p.node('wl','Low-filter weights\nConv → spatial softmax',510,1020,280,85,BLUE,'rect',21)
    p.node('wh','High-filter weights\nConv → spatial softmax',510,1190,280,85,ORANGE,'rect',21)
    p.edge('context','wl',color=BLUE[1],ex=(1,.3),en=(0,.5))
    p.edge('context','wh',color=ORANGE[1],ex=(1,.8),en=(0,.5),points=[(460,1130),(460,1232)])
    p.node('dfilter','LP_low(D)\nIdentity / Blur3 / Blur5',870,1020,280,85,BLUE,'rect',21)
    p.node('cfilter','LP_high(C*)\nIdentity / Blur3 / Blur5',870,1190,280,85,ORANGE,'rect',21)
    p.edge('wl','dfilter','weights',BLUE[1]);p.edge('wh','cfilter','weights',ORANGE[1])
    p.node('dport','D',950,975,70,30,PURPLE,'port',19)
    p.node('cport','C*',950,1138,70,32,PURPLE,'port',19)
    p.edge('dport','dfilter',color=PURPLE[1],ex=(.5,1),en=(.4,0),dash=True)
    p.edge('cport','cfilter',color=PURPLE[1],ex=(.5,1),en=(.4,0),dash=True)
    p.node('dp','D′ = D + α [LP_low(D) − D]',1215,1030,520,65,BLUE,'rect',22)
    p.node('ra','R_adaptive = β [C* − LP_high(C*)]',1215,1197,520,65,ORANGE,'rect',22)
    p.edge('dfilter','dp',color=BLUE[1]);p.edge('cfilter','ra',color=ORANGE[1])
    p.text('dpnote','D′ → Haar → D′_low → semantic paths\nD′ also goes directly to decoder concatenation.',1210,1097,540,77,20,'fontColor=#4C78A8;')
    p.text('ranote','R_adaptive → sum in panel A',1240,1270,520,40,20,'fontColor=#B57937;')
    p.text('adapt_note','V_low = Haar-low(V); D_low = Haar-low(D). Filter mixtures are predicted per spatial position and channel group.\nAt up2 / up1: D′ = D and R_adaptive = 0; the semantic and detail paths in panel A remain active.',80,1320,1710,65,20)
    p.node('out_group','',50,1440,1780,130,GREEN,'group')
    p.node('outy','Y',500,1450,80,40,GREEN,'port',22)
    p.node('outd','D′',500,1510,80,40,BLUE,'port',22)
    p.node('cat','Channel concat [Y, D′]',800,1460,380,60,GRAY,'rect',23)
    p.node('outconv','Two 3 × 3 convs → decoder feature',1320,1459,440,62,PURPLE,'rect',22)
    p.edge('outy','cat',color=GREEN[1],en=(0,.25));p.edge('outd','cat',color=BLUE[1],en=(0,.75))
    p.edge('cat','outconv',color=GREEN[1])
    return p


def race():
    p=Page('03 · RACE',1880,1490)
    p.title('RACE · Report–Anatomy Consistency and Evidence Fusion','Learned residual routing on C1–C4 · the original CNN encoder stream remains unchanged')
    p.node('report_group','',50,160,1780,365,GOLD,'group')
    p.text('report_title','A. Shared report prior',70,170,1100,42,26,'fontStyle=1;')
    p.node('t','T\nText tokens',90,240,215,90,GOLD,'cube',23)
    p.node('pool','Masked mean',355,248,200,70,GOLD,'rect',22)
    p.node('slot','Slot head\nLinear → LN → GELU → Linear',610,235,335,105,GOLD,'rect',22)
    p.node('z','Sigmoid → z\nSix location scores',1020,235,270,105,GOLD,'rect',22)
    p.node('prior','Spatial prior P\nclamp(Σk zk Bk, 0, 1)',1440,245,290,115,ORANGE,'rect',23)
    p.edge('t','pool',color=GOLD[1]);p.edge('pool','slot',color=GOLD[1]);p.edge('slot','z','6 logits',GOLD[1]);p.edge('z','prior',color=GOLD[1])
    p.node('basis','B · six anatomical zones\nAligned with image transforms',1070,397,350,80,GRAY,'rect',22)
    p.edge('basis','prior',color=GRAY[1],ex=(1,.5),en=(.5,1),points=[(1585,437)])
    p.node('count','3 count logits\nAuxiliary training only',650,400,300,75,GRAY,'rect',21,extra='dashed=1;')
    p.edge('slot','count',color=GRAY[1],ex=(.5,1),en=(.5,0),dash=True)
    p.text('prior_note','P and B are resized to each CNN scale.',1090,480,670,33,21)

    p.node('route_group','',50,565,1780,700,ORANGE,'group')
    p.text('route_title','B. One route per CNN scale',70,575,990,43,26,'fontStyle=1;')
    p.node('c','C\nCNN skip',90,805,220,100,PURPLE,'cube',24)
    p.node('evidence','Visual evidence E\n1 × 1 conv → GN → GELU\n→ 1 × 1 conv → sigmoid',390,670,350,110,BLUE,'rect',22)
    p.node('zonepool','Zone-weighted pooling\nqk = Σ(E ⊙ Bk) / ΣBk',825,662,355,105,BLUE,'rect',22)
    p.node('bport','B',950,610,80,32,GRAY,'port',19)
    p.edge('bport','zonepool',color=GRAY[1],ex=(.5,1),en=(.5,0),dash=True)
    p.node('agreement','Agreement A\nReport-weighted discrepancy\nclamped to [0, 1]',1310,650,360,120,ORANGE,'rect',22)
    p.node('zport','z',1415,602,70,32,GOLD,'port',19)
    p.edge('zport','agreement',color=GOLD[1],ex=(.5,1),en=(.5,0),dash=True)
    p.edge('c','evidence',color=BLUE[1],ex=(.5,0),en=(0,.5),points=[(200,725)])
    p.edge('evidence','zonepool','E',BLUE[1]);p.edge('zonepool','agreement','q',BLUE[1])
    p.node('gate','×\nG = P ⊙ E · A',1290,850,285,100,ORANGE,'rect',25)
    p.node('pport','P',1635,875,85,45,ORANGE,'port',22)
    p.edge('pport','gate',color=ORANGE[1],ex=(0,.5),en=(1,.5),dash=True)
    p.edge('agreement','gate','A',ORANGE[1],ex=(.5,1),en=(.7,0))
    p.edge('evidence','gate','E',BLUE[1],ex=(.5,1),en=(0,.5),points=[(565,810),(1235,810),(1235,900)])
    p.node('residual','Learned residual R(C)\nDWConv 3 × 3 → PWConv 1 × 1\n→ GN → GELU',390,990,425,120,PURPLE,'rect',22)
    p.node('multiply','×',1090,1015,75,75,ORANGE,'circle',33)
    p.node('strength','s = 0.15 tanh(a)\na starts at 0',845,1150,280,70,ORANGE,'rect',22)
    p.node('add','+',1460,1020,70,70,ORANGE,'circle',36)
    p.node('routed','C′\nRouted skip',1595,1010,180,105,ORANGE,'cube',23)
    p.edge('c','residual',color=PURPLE[1],ex=(.5,1),en=(0,.5),points=[(200,1050)])
    p.edge('residual','multiply','R(C)',PURPLE[1]);p.edge('gate','multiply','G',ORANGE[1],ex=(0,.8),en=(.5,0),points=[(1127,930)])
    p.edge('strength','multiply','s',ORANGE[1],ex=(1,.5),en=(.5,1),points=[(1127,1185)])
    p.edge('multiply','add','s G ⊙ R(C)',ORANGE[1]);p.edge('add','routed',color=ORANGE[1])
    p.edge('c','add','Original C identity bypass',PURPLE[1],ex=(0,.7),en=(.5,1),points=[(70,875),(70,1240),(1495,1240)])
    p.text('agreement_formula','A = clamp(1 − Σk |qk − zk| · stopgrad(zk) / max(Σk stopgrad(zk), 1), 0, 1)',370,820,880,75,18,'fontColor=#5B6877;')
    p.text('gate_note','A: sample scalar; P, E: spatial maps',1190,970,595,35,20,'fontColor=#5B6877;')
    p.text('output_formula','C′ = C + s G ⊙ R(C)',350,1155,470,58,26,'fontStyle=1;align=center;')

    p.node('training','',50,1310,1780,130,GRAY,'group')
    p.text('training_title','Complete RACE during training',75,1320,650,38,24,'fontStyle=1;')
    p.text('training_text','Report-slot / count, visual-evidence and agreement auxiliary objectives train the complete module.\nParsed report targets and mask-derived targets supervise training only; they are not inference inputs.',75,1367,1710,61,21)
    return p


def concise(p):
    """Presentation labels: operations and signals, with prose kept in the notes."""
    labels = {
        '01': {
            'subtitle': '', 'vit_title': 'ViT branch', 'cnn_title': 'CNN branch',
            'report': 'Report', 'bert': 'CXR-BERT', 'text_note': '',
            'mask': 'Mask', 'head': 'Conv 1×1 → σ', 'vit_note': '',
            'detail_title': 'Decoder stage', 'prev': 'Decoder input',
            'concat': 'Concat', 'conv': 'Conv 3×3 ×2',
            'legend': 'T: text     B: anatomy     Dashed: matching ports',
            'footer': '',
            **{f'c{i}':f'C{i}' for i in range(1,5)},
            **{f'up{i}':f'up{i}\nFSDR stage' for i in range(1,5)},
        },
        '02': {
            'title':'FSDR', 'subtitle':'', 'main_title':'A. Semantic–detail fusion',
            'c':'C*', 'haar':'Haar split', 'channel':'Channel gate',
            'region':'Text FiLM\nSpatial support', 'rsem':'Semantic fusion',
            'channel_formula':'', 'refine':'Local + dilated\nConv',
            'detail':'Detail fusion', 'support_label':'S', 'main_note':'',
            'yformula':'', 'adapt_title':'B. Adaptive filters · up4 / up3',
            'context':'C_low, V_low, D_low\nContext fusion',
            'wl':'Conv → Softmax\nw_low', 'wh':'Conv → Softmax\nw_high',
            'dfilter':'Filter bank\nLP_low(D)', 'cfilter':'Filter bank\nLP_high(C*)',
            'dp':'Decoder smoothing\nD′', 'ra':'Detail residual\nR_adaptive',
            'dpnote':'D′ → Haar → D′_low', 'ranote':'',
            'adapt_note':'Filter bank: Identity · Blur3 · Blur5',
            'cat':'Concat', 'outconv':'Conv 3×3 ×2',
        },
        '03': {
            'title':'RACE', 'subtitle':'', 'report_title':'A. Report prior',
            't':'T', 'slot':'Slot MLP', 'z':'σ → z', 'prior':'Spatial prior\nP',
            'basis':'Anatomical basis B', 'count':'Count\n(train only)',
            'prior_note':'', 'route_title':'B. Skip routing · C1–C4',
            'c':'C', 'evidence':'Conv ×2 → σ\nE', 'zonepool':'Zone pooling',
            'agreement':'Agreement\nA', 'gate':'P × E × A',
            'residual':'DWConv → PWConv\nR(C)', 'strength':'Scale s',
            'routed':'C′', 'agreement_formula':'', 'gate_note':'',
            'training_title':'Training only',
            'training_text':'Slot / count · visual evidence · agreement',
        },
    }[p.name[:2]]
    for id,label in labels.items():
        cell=p.root.find(f"mxCell[@id='{id}']")
        if label:
            cell.set('value',label)
        else:
            assert not any(id in e for e in p.edges), id
            p.root.remove(cell); p.ids.remove(id); del p.boxes[id]
    # Short labels remain large enough to read on a presentation slide.
    for cell in p.root.findall('mxCell'):
        if cell.get('edge')=='1':
            if cell.get('value','').startswith('Original '): cell.set('value','Identity')
        elif cell.get('vertex')=='1':
            style=cell.get('style','')
            if 'rounded=1;arcSize=12;' in style:
                import re
                cell.set('style',re.sub(r'fontSize=\d+;', 'fontSize=26;',style))
    return p


def export(pages, filename):
    root=ET.Element('mxfile',host='app.diagrams.net',agent='Native editable architecture builder',version='26.0.0',type='device')
    checks=[]
    for i,p in enumerate(pages,1):
        checks.append(p.validate())
        diagram=ET.SubElement(root,'diagram',id=f'lvit-fsdr-race-{i}',name=p.name)
        diagram.append(p.model)
    target=OUT/filename
    ET.indent(root)
    ET.ElementTree(root).write(target,encoding='utf-8',xml_declaration=True)
    chars=sum(len(cell.get('value','')) for p in pages for cell in p.root.findall('mxCell'))
    print(target)
    return {'file':filename,'pages':checks,'visible_characters':chars}


def main():
    full=export([overview(),fsdr(),race()],'FSDR_RACE_architecture_20260922.drawio')
    short=export([concise(p) for p in [overview(),fsdr(),race()]],'FSDR_RACE_architecture_concise.drawio')
    result={'versions':[full,short],'native_editable':True,'embedded_images':0,'source_commit':'b71218b52240c60b2e47c25d584448f0dde2f73d'}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__': main()
