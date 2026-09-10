"""Train-mask geometry audit for choosing regional supervision; no model inference."""
import hashlib
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np

REPO=Path('/root/autodl-tmp/BetterLViT-visual-aux-s2')
SOURCE='7defc637e36974fabd0f47763275c90f617e5f53'


def quantiles(values):
    return dict(zip(('min','p10','p25','median','p75','p90','max'),
                    np.quantile(values,[0,.1,.25,.5,.75,.9,1]).tolist())) if len(values) else {}


def main():
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==SOURCE
    assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain','--untracked-files=no'],text=True).strip()
    cv2.setNumThreads(1)
    files=sorted((REPO/'datasets/Covid19/Train_Folder/labelcol').iterdir())
    assert len(files)==5716
    areas=[];counts={str(k):[] for k in (1,4,16)};largest=[];small_mass=[];foreground=[];fractions=[]
    patches={str(k):dict(total=0,empty=0,full=0,partial=0,positive_below_5_percent=0,positive_fractions=[]) for k in (28,56,112)}
    examples=[];mask_hash=hashlib.sha256()
    for path in files:
        image=cv2.imread(str(path),0);assert image is not None
        mask=(cv2.resize(image,(224,224))>0).astype(np.uint8)
        mask_hash.update(path.name.encode()+b'\0'+mask.tobytes())
        n,_,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
        a=stats[1:,cv2.CC_STAT_AREA].astype(np.int64);areas.extend(a.tolist())
        fg=int(mask.sum());foreground.append(fg)
        for cutoff in (1,4,16):counts[str(cutoff)].append(int((a>=cutoff).sum()))
        if fg:
            largest.append(float(a.max()/fg));small_mass.append(float(a[a<16].sum()/fg));fractions.extend((a/fg).tolist())
        if len(examples)<12 and len(a)>=3:
            examples.append(dict(name=path.name,component_areas=sorted(a.tolist(),reverse=True)))
        for side in (28,56,112):
            area=mask.reshape(224//side,side,224//side,side).sum((1,3)).ravel()
            row=patches[str(side)];row['total']+=len(area);row['empty']+=int((area==0).sum())
            row['full']+=int((area==side*side).sum());row['partial']+=int(((area>0)&(area<side*side)).sum())
            pos=area[area>0]/(side*side);row['positive_fractions'].extend(pos.tolist())
            row['positive_below_5_percent']+=int((pos<.05).sum())
    for row in patches.values():
        pos=row.pop('positive_fractions');row['positive_fraction_quantiles']=quantiles(pos)
        row['empty_fraction']=row['empty']/row['total']
    result=dict(scope='5716 native Train masks only; no augmentation, weights, GPU, Val or Test',
        source_git_commit=SOURCE,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        normalized_masks_sha256=mask_hash.hexdigest(),opencv_version=cv2.__version__,numpy_version=np.__version__,
        preprocessing='original grayscale cv2.resize default linear to 224x224, then >0, matching existing pipeline',
        samples=len(files),empty_images=sum(x==0 for x in foreground),foreground_pixels_quantiles=quantiles(foreground),
        component_count_quantiles={k:quantiles(v) for k,v in counts.items()},
        images_with_multiple_components={k:sum(x>=2 for x in v) for k,v in counts.items()},
        component_total=len(areas),component_area_quantiles=quantiles(areas),
        components_below_4_pixels=sum(a<4 for a in areas),components_below_16_pixels=sum(a<16 for a in areas),
        largest_component_foreground_share_quantiles=quantiles(largest),
        mean_foreground_share_of_components_below_16_pixels=float(np.mean(small_mass)),
        component_foreground_share_quantiles=quantiles(fractions),patches=patches,examples=examples,
        limitations='2D connected regions are geometric proxies, not clinical lesion instances; area cutoffs are descriptive only; no label removal',
        test_split_accessed=False,validation_split_accessed=False,model_inference=False,training_performed=False)
    print(json.dumps(result))


if __name__=='__main__':main()
