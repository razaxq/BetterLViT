# FAM-EPPA V4-H: Token-Conditioned Frequency Routing

## Evidence-driven hypothesis

V4-B remains the strongest architecture experiment. Its spatially adaptive,
grouped mixtures over identity, 3x3 blur, and 5x5 blur remain normalized and
active throughout training. V4-C through V4-G instead injected flow, token
residuals, amplitude gates, prototypes, or reliability residuals after feature
formation; none improved complete-test generalization, and V4-F/V4-G exposed
collapse or dead-gradient shortcuts.

V4-H therefore changes only the successful V4-B decision: which normalized
low/high-frequency filter each pixel and channel group should use. It adapts
the token-level linguistic-filter idea from FMISeg while retaining the bounded
frequency path and avoiding direct text-to-visual amplitude injection.

## Token-conditioned normalized route

At `up4` and `up3`, the existing V4-B visual context predicts base ALPF/AHPF
logits. Context pixels query all valid CXR-BERT tokens with four-head masked
cosine attention. The attended token representation interacts with the local
visual context through normalized product and absolute-difference features.
A zero-initialized 1x1 route head predicts bounded logit corrections:

```text
Q = normalize(Wq C)
K = normalize(Wk T)
A = softmax(mask(Q K^T) * temperature)
U = Wo(A Wv T)
I = concat(normalize(C) * normalize(U),
           abs(normalize(C) - normalize(U)))
delta_low, delta_high = split(tanh(Wroute I))
w_low  = softmax(base_low_logits  + delta_low)
w_high = softmax(base_high_logits + delta_high)
```

The correction is bounded to `[-1, 1]` in logit space. Softmax keeps every
filter mixture non-negative and exactly sum-normalized. The route head receives
a gradient on the first backward pass even though it starts at zero; after its
first update, gradients reach token query/key/value projections. This avoids
V4-G's `zero strength * residual` dead-gradient product.

## Controlled scope

- Base architecture: FAM-EPPA V4-B.
- Adaptive ALPF/AHPF: unchanged at `up4` and `up3`.
- Token-conditioned frequency correction: `up4` and `up3` only.
- V4-C flow, V4-D direct token residual, V4-E gate, V4-F prototypes, and V4-G
  reliability residual: disabled.
- Loss: Dice/Focal (`0.5/0.5`, gamma `2.0`).
- Boundary loss: `0.0`.
- Seed `1219`, batch size `16`, maximum `200` epochs.

## Diagnostics and acceptance

The run logs attention entropy/peak, CLS and non-CLS mass, spatial attention
variation, attended-text variation, raw/bounded route delta, and the mean
ALPF/AHPF weight shift from the visual-only V4-B route. Existing kernel sums,
entropies, route strengths, and FAM-EPPA branch diagnostics remain enabled.

Acceptance requires:

1. Exact V4-B output at initialization.
2. Non-zero route-head gradient on the first backward pass.
3. Non-zero route variation and filter-weight shift by approximately Epoch 20.
4. ALPF/AHPF kernel sums remain `1`.
5. Complete test Dice/IoU exceed V4-B `0.844996/0.760273`, after choosing the
   threshold using only all 1,429 validation images and fixing it for all 2,113
   test images.
