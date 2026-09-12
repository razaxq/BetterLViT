"""Pin the next text interface to actual R2 source; no dataset or GPU access."""
import hashlib,json,subprocess
from remote_ops import HERE,save

ROOT='D:/BetterLViT/recipe_r2s2027_work'
BASE='9eca26de5b301099805530edbf5a1a8718bea662'
FILES=('nets/BetterLViT.py','nets/LViT.py','nets/eppa.py','nets/Vit.py','Config.py','Load_Dataset.py')

def main():
    sources={name:subprocess.check_output(['git','show',BASE+':'+name],cwd=ROOT) for name in FILES}
    text={name:data.decode('utf-8') for name,data in sources.items()}
    anchors={
        'nets/BetterLViT.py':['def encode_text(self, input_ids, attention_mask):','text_mask=attention_mask,'],
        'nets/LViT.py':['0, in_channels * 16, in_channels * 4, \'up4\'','1, in_channels * 8, in_channels * 2, \'up3\'',
            'd3 = self.up3(d4, x3, plam3, text=text)','d2 = self.up2(d3, x2, plam2, text=text)',
            'self.text_module2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)'],
        'nets/eppa.py':['self.text_channel_proj(text[:, 0, :])','self.text_pixel_film(text[:, 0, :])'],
        'Config.py':['text_max_len = 32','config.base_channel = 64'],
        'Load_Dataset.py':["'attention_mask': sample['attention_mask']","padding='max_length'"]}
    positions={}
    for name,needles in anchors.items():
        positions[name]=[]
        for needle in needles:
            lines=[i+1 for i,line in enumerate(text[name].splitlines()) if needle in line]
            assert lines,(name,needle);positions[name].append(dict(anchor=needle,lines=lines))
    assert not subprocess.check_output(['git','diff','--name-only',BASE,'c972a152b5d152ddfa82e034596085aca85dd0ab','--',*FILES],cwd=ROOT,text=True).strip()
    proposed=dict(location='Immediately after the FAM-EPPA d3=up3(...) output and before d2=up2(...)',
        d4=[None,256,28,28],d3=[None,128,56,56],d2=[None,64,112,112],d1=[None,64,224,224],
        projected_text2=[None,32,128],attention_width=32,attention_heads=4,
        adapter_parameters=4*128*32+2*2*128,
        parameter_accounting='Bias-free Q/K/V/output projections plus affine LayerNorm on query and context',
        shapes_are_static_source_inference=True,runtime_shape_verified=False,implemented=False,trained=False)
    proof=dict(verified=True,baseline_source_git_commit=BASE,audited_files={n:hashlib.sha256(v).hexdigest() for n,v in sources.items()},
        anchors=positions,network_and_tokenizer_sources_identical_to_seed2027=True,proposed_interface=proposed,
        constraints=['Preserve original EPPA and token-to-patch path','Only new-path attention keys mask padding',
            'Existing text convolutions can mix neighboring padded states; do not claim end-to-end padding invariance',
            'No fixed image-coordinate prior while legacy geometric augmentation remains',
            'The proposed adapter is not an established innovation'],
        dataset_accessed=False,test_split_accessed=False,formal_training_performed=False,gpu_used=False)
    save('text_interface_audit.json',proof);print(json.dumps({k:v for k,v in proof.items() if k not in ('audited_files','anchors')},indent=2))

if __name__=='__main__':main()
