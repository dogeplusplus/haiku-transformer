# haiku-transformer

- Implementation of the basic Vision Transformer using `jax` and `dm-haiku`
- Model and optimizer serialization/deserialization with pickle
- Uses tensor manipulation operations using `einops`
- Re-implementation of Attention Rollout with percentile based filtering from `https://github.com/jacobgil/vit-explain` (WIP)
- Inference script with attention rollout (WIP)
