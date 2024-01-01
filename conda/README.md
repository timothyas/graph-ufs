# Environment

Only the packages listed in [gpu.yaml](gpu.yaml) should be necessary, but there
are currently issues using GraphCast with recent versions of JAX, see e.g.
[#41](https://github.com/google-deepmind/graphcast/issues/41) or
[#30](https://github.com/google-deepmind/graphcast/issues/30).
To avoid these issues, use [gpu-workaround.yaml](gpu-workaround.yaml).
