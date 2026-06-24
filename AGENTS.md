# Equimo Contributor Instructions

## Dimension Naming

Use axis names that match the operation domain:

- Linear layers and heads use `features`, `in_features`, and `out_features`.
- Token or embedding-width modules use `dim` only when that width is preserved.
- Token or embedding-width modules that change width use `in_dim` and `out_dim`.
- Image/audio-to-token projection modules use `embed_dim` for the token output width.
- Convolution modules use `channels` when the channel count is preserved.
- Convolution modules that can change channel count use `in_channels` and `out_channels`.
- Mixed-domain modules name both sides explicitly, for example `in_channels` to `embed_dim`.

Do not mix input `dim` with `out_channels` in one constructor. Do not add
compatibility aliases for old parameter names unless explicitly requested.
