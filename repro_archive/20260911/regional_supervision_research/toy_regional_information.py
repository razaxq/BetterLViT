"""Counterexample for presence/area target information, not a simulated S2 model."""
import json
from pathlib import Path
import numpy as np


def describe(prediction,target):
    tp=int((prediction*target).sum());union=int(prediction.sum()+target.sum()-tp)
    return dict(presence=int(prediction.any()),occupancy=float(prediction.mean()),
                occupancy_squared_error=float((prediction.mean()-target.mean())**2),
                intersection=tp,union=union,iou=tp/union)


if __name__=='__main__':
    target=np.zeros((32,32),dtype=np.uint8);target[2:6,2:6]=1
    correct=target.copy();displaced=np.zeros_like(target);displaced[24:28,24:28]=1
    data=dict(scope='synthetic supervision counterexample; no patient data, no neural model',
              target_presence=1,target_occupancy=float(target.mean()),
              correct=describe(correct,target),displaced=describe(displaced,target))
    assert data['correct']['presence']==data['displaced']['presence']==1
    assert data['correct']['occupancy_squared_error']==data['displaced']['occupancy_squared_error']==0
    assert data['correct']['iou']==1 and data['displaced']['iou']==0
    Path(__file__).with_name('toy_regional_information.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))
