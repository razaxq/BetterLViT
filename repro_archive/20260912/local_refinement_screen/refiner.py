"""Same-parameter coarse-grid and fine-point heads with strictly local edits."""
import torch
from torch import nn
from torch.nn import functional as F
from mass_projection import MassProject,interpolation_matrix
VARIANTS=('coarse_free','coarse_mass','fine_free','fine_mass')

class LocalRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(67,128);self.fc2=nn.Linear(128,64);self.out=nn.Linear(64,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
        self.register_buffer('upsample',interpolation_matrix())
        yy,xx=torch.meshgrid((torch.arange(28)+.5)/28*2-1,(torch.arange(28)+.5)/28*2-1,indexing='ij')
        self.register_buffer('grid_coordinates',torch.stack((xx,yy),-1).reshape(1,784,2))

    def point_signal(self,feature,z,xy):
        x=torch.cat((F.normalize(feature.float(),dim=-1),z.clamp(-8,8).unsqueeze(-1)/8,xy),dim=-1)
        return .5*torch.tanh(self.out(F.gelu(self.fc2(F.gelu(self.fc1(x)))))).squeeze(-1)

    def forward(self,coarse,fine,z,index,variant):
        assert variant in VARIANTS and not z.requires_grad
        if variant.startswith('coarse'):
            feature=coarse.permute(0,2,3,1).reshape(len(z),784,64)
            low_z=F.avg_pool2d(z,8).flatten(1)
            delta=self.point_signal(feature,low_z,self.grid_coordinates.expand(len(z),-1,-1)).reshape(-1,1,28,28)
            full=self.upsample@delta@self.upsample.T
            return full.flatten(1).gather(1,index)
        xy=torch.stack(((index%224+.5)/224*2-1,(index//224+.5)/224*2-1),-1)
        return self.point_signal(fine,z.flatten(1).gather(1,index),xy)

def predict(z,delta,index,variant):
    selected_z=z.flatten(1).gather(1,index)
    selected=(MassProject.apply(selected_z,delta) if variant.endswith('mass') else (selected_z+delta).sigmoid())
    return z.sigmoid().flatten(1).scatter(1,index,selected).reshape_as(z)
