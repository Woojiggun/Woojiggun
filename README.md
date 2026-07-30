# Jinhyun Woo

**Independent AI researcher building HSL and HoLo**

I study whether deterministic, byte-native structural signals can replace part of the learned tokenizer and embedding stack while preserving useful model behavior. HoLo is the integration and evaluation vehicle; HSL is the core research product.

This is an open, evidence-first project. I publish code, weights, matched controls, implementation limits, and negative results. The current claim is **works and is worth measuring**, not SOTA.

## Research portfolio

| Project | What it tests | Public evidence |
| --- | --- | --- |
| [HSL](https://github.com/Woojiggun/holo-hsl) | A deterministic, modality-agnostic structural signal substrate | Core repository, specifications, experiments, and reports |
| [hsl-embedding](https://github.com/Woojiggun/hsl-embedding) | Reproducible byte-to-HSL feature generation | Python package, tests, and CI |
| [hsl-embedding-zero](https://github.com/Woojiggun/hsl-embedding-zero) | Transformer input with zero learned input parameters | Python package, tests, CI, and released weights |
| [HoLo_ZeRo](https://github.com/Woojiggun/HoLo_ZeRo) | Language-model integration of the zero-input path | Training code, checkpoints, and documented limitations |
| [HoLo-ToLk-STT](https://github.com/Woojiggun/HoLo-ToLk-STT) | HSL-based speech recognition | Four-seed matched experiments and released artifacts |
| [HoLo-ToLk-TTS](https://github.com/Woojiggun/HoLo-ToLk-TTS) | HSL-based text-to-speech | Working prototype, code, and weights; perceptual evaluation remains open |
| [HoLo-FuSe](https://github.com/Woojiggun/HoLo-FuSe) | Frozen HSL features in diffusion conditioning | Reproducible prototype and retained negative-control evidence |

## Current research

- Controlled language-model studies on clean English and Korean corpora
- Byte-level BPB, memory, throughput, parameter, and convergence economics
- Exact long-context and autoregressive rollout diagnostics
- Matched ablations that separate data quality, input representation, optimization, and scale
- Speech and diffusion prototypes as cross-modal feasibility studies
- A private exploratory autonomous-system prototype, not presented as public evidence

The language path remains harder than conventional learned BPE embeddings: token IDs are not used as learned semantic inputs. That makes direct quality comparisons incomplete unless parameter cost, byte exposure, compute, and convergence are matched.

## Research principles

- **Works-not-wins:** feasibility before superiority claims
- **Matched controls:** same data, exposure, seed, and evaluation whenever possible
- **Negative results stay public:** rejected paths are evidence, not hidden failures
- **Reproducibility:** code, weights, hashes, and run contracts where practical
- **Economy matters:** quality is evaluated together with parameters, VRAM, throughput, and data cost

## Links

- [GitHub repositories](https://github.com/Woojiggun?tab=repositories)
- [Curated Hugging Face research collection](https://huggingface.co/collections/ggunio/hsl-holo-open-feasibility-portfolio-6a6b38f6016eb6fe23bcc9ea)
- [All Hugging Face models and demos](https://huggingface.co/ggunio)
- [HSL on PyPI](https://pypi.org/project/hsl-embedding/)
- [Zero-input package on PyPI](https://pypi.org/project/hsl-embedding-zero/)
- [HSL paper on Zenodo](https://doi.org/10.5281/zenodo.20581805)
- [ORCID](https://orcid.org/0009-0002-3273-1169)
- [LinkedIn](https://www.linkedin.com/in/namuneup)
- Contact: `ggunio5782@gmail.com`

## Compute collaboration

I am seeking research-compute sponsorship or credits for reproducible scaling studies, especially controlled HSL/HoLo language, speech, and diffusion experiments. The immediate goal is not to declare a breakthrough; it is to determine rigorously where the approach works, where it fails, and whether its parameter and memory economics survive at larger scale.

---

## 한국어 요약

HSL은 바이트에서 고정 구조 신호를 만드는 입력 기판이며, HoLo는 이를 언어·음성·이미지 모델에서 검증하는 통합 연구입니다. 학습된 토큰 임베딩 일부를 고정 신호로 대체할 수 있는지, 그리고 그 과정에서 파라미터·메모리·처리량 측면의 경제성이 유지되는지를 공개 실험으로 확인하고 있습니다.

현재 공개된 결과는 가능성과 작동성을 보여 주지만 완성형이나 성능 우위를 뜻하지 않습니다. 코드, 가중치, 대조군, 실패 사례를 함께 공개하며 더 큰 규모의 재현 가능한 검증을 위한 컴퓨팅 협력을 찾고 있습니다.
