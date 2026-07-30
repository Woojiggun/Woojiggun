# HSL / HoLo compute research brief

**Researcher:** Jinhyun Woo, independent AI researcher  
**Public portfolio:** [GitHub](https://github.com/Woojiggun) · [Hugging Face](https://huggingface.co/collections/ggunio/hsl-holo-open-feasibility-portfolio-6a6b38f6016eb6fe23bcc9ea) · [Zenodo DOI](https://doi.org/10.5281/zenodo.20581805)

## Research question

Can deterministic, byte-native structural signals replace part of the learned tokenizer and embedding stack while preserving useful model behavior and improving parameter, memory, or data economics?

HSL is the zero-learned-parameter structural input substrate. HoLo is the model family used to test it in language, speech, diffusion, and multimodal settings. The current claim is intentionally narrow: the mechanism works and deserves matched scaling tests. It is not a SOTA claim.

## Evidence already produced

The public portfolio includes:

- a fully open and tested byte-to-HSL implementation on PyPI;
- zero-learned-input-parameter transformer integration;
- working language, STT, TTS, diffusion, and multimodal prototypes;
- released code, weights, hashes, controls, and documented negative results;
- single-GPU matched studies on an RTX 4070.

The latest controlled Korean language gate trained two matched 73.33M-parameter HoLo arms for two exact no-replacement epochs and 3.303B target bytes each. Exact previous-byte feedback improved validation/test BPB from `0.602928/0.597821` to `0.601155/0.596147` while adding 864 parameters and approximately 0.58% wall-throughput cost. It also improved mean self-rollout BPB from `6.847124` to `6.743871`.

This is a small, economically cheap gain, not a solved model. Late-slot error propagation and weak free generation remain the principal measured bottlenecks. Full evidence and limitations: [Tiny-Ko matched two-epoch report](https://github.com/Woojiggun/holo-hsl/blob/master/EVIDENCE_TINY_KO_20260730.md).

## Why additional compute is decision-critical

The two-epoch curves were still improving, so the current endpoints cannot answer whether HSL is competitive at convergence. A larger, controlled allocation would answer four falsifiable questions:

1. Do HSL-input models retain their parameter and memory advantage at a predefined validation plateau?
2. Does the result survive matched conventional byte and learned-BPE baselines?
3. Which quality loss comes from representation, optimization, context mixing, or insufficient scale?
4. Are the findings stable across independent seeds and English/Korean corpora?

## Initial 100 H100-hour plan

| Work package | Budget | Measurable output |
|---|---:|---|
| Reproducibility and accelerator port | 10 h | Strict checkpoint parity, deterministic resume, throughput and memory baseline |
| Matched convergence study | 35 h | HSL baseline and exact-byte arms trained to preregistered plateau |
| Reference-model comparisons | 35 h | Parameter- and byte-exposure-matched byte and learned-BPE baselines |
| Replication and final evaluation | 20 h | Independent seeds, fixed generation, BPB, rollout, VRAM, throughput, and confidence intervals |

For TPU support, the same matrix would be resized only after an XLA port and measured pilot; GPU-hour and TPU-core-hour claims would not be treated as interchangeable.

## Success and rejection criteria

Success is not simply a lower loss. HSL must show a reproducible parameter, VRAM, throughput, or data-efficiency benefit at matched validation quality and convergence. A quality gain without economic benefit, or an economic gain that disappears under a conventional matched baseline, will narrow or reject the corresponding design claim.

All comparisons will pair teacher-forced BPB with autoregressive rollout and direct fixed-prompt inspection. Negative and inconclusive results will remain public.

## Outputs

- Open experiment contracts, code, and evaluation scripts
- Checkpoint and dataset provenance hashes where licensing permits
- Matched tables for BPB, parameters, VRAM, throughput, byte exposure, and convergence
- Public technical report and archival DOI update
- Reproducible Hugging Face models or demos where licensing permits
- A co-authored technical blog for a compute sponsor, if requested

## Relevant links

- [HoLo / HSL](https://github.com/Woojiggun/holo-hsl)
- [hsl-embedding](https://github.com/Woojiggun/hsl-embedding)
- [hsl-embedding-zero](https://github.com/Woojiggun/hsl-embedding-zero)
- [HoLo-ToLk-STT](https://github.com/Woojiggun/HoLo-ToLk-STT)
- [HoLo-ToLk-TTS](https://github.com/Woojiggun/HoLo-ToLk-TTS)
- [HoLo-FuSe](https://github.com/Woojiggun/HoLo-FuSe)

This project is independent, open, and evidence-first. Sponsored compute would be used to determine rigorously where the approach works, where it fails, and whether its economics survive controlled scaling.
