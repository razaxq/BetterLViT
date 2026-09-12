"""Small CPU regressions for masking, initialization, gradients and controls."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from nets.decoder_context import DecoderContextAdapter


def main():
    torch.set_num_threads(2)
    torch.manual_seed(1219)
    x = torch.randn(3, 128, 8, 16)
    text = torch.randn(3, 32, 128)
    mask = torch.arange(32)[None, :] < torch.tensor([13, 2, 0])[:, None]
    initial = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        visual = DecoderContextAdapter('visual')
    assert torch.equal(initial, torch.get_rng_state())
    with torch.random.fork_rng(devices=[]):
        language = DecoderContextAdapter('text')
    assert all(torch.equal(v, language.state_dict()[k]) for k, v in visual.state_dict().items())
    for module in (visual, language):
        assert sum(p.numel() for p in module.parameters()) == 16896
        assert torch.equal(module(x, text, mask), x)
    with torch.no_grad():
        language.output.weight.normal_(0, .05)
        visual.output.weight.copy_(language.output.weight)
    masked_text = text.clone().masked_fill(~mask[:, :, None], 10000)
    assert torch.equal(language(x, text, mask), language(x, masked_text, mask))
    assert torch.equal(language(x, text, mask)[1:], x[1:])
    assert torch.equal(visual(x, text, mask), visual(x, text.flip(0), ~mask))
    assert not torch.equal(language(x, text, mask)[0], language(x, text.flip(0), mask)[0])
    grad_records = {}
    for mode in ('visual', 'text'):
        module = DecoderContextAdapter(mode)
        opt = torch.optim.Adam(module.parameters(), lr=.001)
        records = []
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            y = module(x, text, mask)
            (y - torch.ones_like(y)).square().mean().backward()
            grads = {n:float(p.grad.abs().max()) for n,p in module.named_parameters()}
            assert all(torch.isfinite(p.grad).all() for p in module.parameters())
            assert grads['output.weight'] > 0
            if step:
                assert all(grads[n] > 0 for n in ('query.weight','key.weight','value.weight'))
            opt.step()
            records.append(grads)
        grad_records[mode] = records
    empty = DecoderContextAdapter('text')
    empty(x, text, torch.zeros_like(mask)).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in empty.parameters())
    print(json.dumps(dict(status='ok', device='cpu', matched_parameters=16896,
        zero_init_identity=True, padding_key_invariance=True, empty_fallback=True,
        visual_control_ignores_text=True, cpu_rng_preserved=True,
        same_adapter_initialization=True, early_gradients=grad_records)))


if __name__ == '__main__':
    main()
