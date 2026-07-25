# Born float32 normalization fix

The full model-ready campaign exposed a valid Born distribution that failed the tensor contract after batching. A complete audit of all 33,540 train/validation Born artifacts found no malformed distributions. The maximum source mass deviation was `4.47e-08`, and the maximum direct float32 `torch.sum` deviation was `1.19e-07`.

The failure came from sequential float32 `index_add_` accumulation across a large outcome support. That accumulation can drift beyond `1e-6` even when the source distribution is normalized.

This revision:

- changes the Born tensor normalization tolerance from `1e-6` to `1e-5`, matching the Hilbert normalization contract;
- keeps materially unnormalized distributions rejected;
- does not renormalize or mutate the immutable model-ready source;
- bumps the training tensor-adapter identity to `v2`, producing a new deterministic run identity;
- adds a regression test reproducing the float32 accumulation case;
- adds a permanent tmux launcher with deterministic cuBLAS configuration.
