"""Temporarily separate direct LViT text from the four EPPA text inputs."""
from contextlib import contextmanager


@contextmanager
def eppa_text_override(model, text):
    restored=[]; counts=[0]*4
    try:
        for index,name in enumerate(('up4','up3','up2','up1')):
            module=getattr(model,name).eppa
            old=module.forward
            own=module.__dict__.get('forward')
            def wrapped(*args,_old=old,_index=index,**kwargs):
                counts[_index]+=1
                kwargs['text']=text
                return _old(*args,**kwargs)
            module.forward=wrapped
            restored.append((module,own))
        yield counts
    finally:
        for module,own in restored:
            if own is None:del module.forward
            else:module.forward=own


def separate_forward(visual_forward, model, image, main_text, main_mask, eppa_text):
    with eppa_text_override(model,eppa_text) as counts:
        result=visual_forward(model,image,main_text,text_mask=main_mask)
        assert counts==[1,1,1,1],counts
    return result
