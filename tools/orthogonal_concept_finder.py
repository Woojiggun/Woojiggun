"""
직교 개념 탐색기 (Orthogonal Concept Finder)
---------------------------------------------
대립 개념 A, B가 만드는 축을 구하고, 후보 개념들 중
그 축과는 무관하게(직교하게) 서 있는 "제3의 개념"을 찾아낸다.

설치:
    pip install sentence-transformers numpy

사용:
    python tools/orthogonal_concept_finder.py                      # 기본 데모
    python tools/orthogonal_concept_finder.py -a 초전도체 -b 저항체 \
        -c 쿠퍼쌍 -c 그래핀 -c 냉각 시스템 --score relevant --json
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional, Protocol, Sequence

import numpy as np

# 한국어 포함 다국어 지원 모델. 한국어 품질을 더 올리고 싶으면
# "jhgan/ko-sroberta-multitask" 같은 한국어 전용 모델로 교체 가능.
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# --backend spacy 일 때 쓰는 기본 패키지 (offline_demo_backend 참고).
SPACY_PACKAGE = "ko_core_news_md"

SCORE_MODES = ("raw", "relevant")


class Encoder(Protocol):
    """SentenceTransformer와 같은 인코더가 만족해야 하는 최소 인터페이스."""

    def encode(self, sentences: Sequence[str], normalize_embeddings: bool = ...) -> np.ndarray:
        ...


@dataclass
class ConceptScore:
    concept: str
    axis_projection: float   # 대립축 위의 위치, 0에 가까울수록 축과 무관
    orthogonal_ratio: float  # 0~1, 1에 가까울수록 순수하게 축 바깥
    distinctiveness: float   # A·B 중점에서 얼마나 먼 개념인지
    relevance: float         # A, B와의 평균 유사도 = 이 영역에 얼마나 붙어 있는지
    sim_to_a: float
    sim_to_b: float
    score: float             # 정렬 기준, --score 모드에 따라 정의가 달라진다


class OrthogonalConceptFinder:
    def __init__(self, model_name: str = MODEL_NAME, model: Optional[Encoder] = None,
                 backend: str = "sentence-transformers"):
        """model을 직접 주면 그것을 쓰고, 아니면 backend/model_name을 지연 로딩한다."""
        self._model_name = model_name
        self._model = model
        self._backend = backend

    @property
    def model(self) -> Encoder:
        if self._model is None:
            if self._backend == "spacy":
                from offline_demo_backend import SpacyVectorEncoder

                self._model = SpacyVectorEncoder(self._model_name)
            else:
                from sentence_transformers import SentenceTransformer  # 무거워서 지연 import

                self._model = SentenceTransformer(self._model_name)
        return self._model

    def find(self, concept_a: str, concept_b: str, candidates: Sequence[str],
             score_mode: str = "raw", eps: float = 1e-8) -> List[ConceptScore]:
        if score_mode not in SCORE_MODES:
            raise ValueError(f"score_mode는 {SCORE_MODES} 중 하나여야 합니다: {score_mode!r}")
        if not candidates:
            return []

        vecs = np.asarray(
            self.model.encode([concept_a, concept_b, *candidates], normalize_embeddings=True),
            dtype=np.float64,
        )
        a, b, cand_vecs = vecs[0], vecs[1], vecs[2:]

        axis = b - a
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < eps:
            raise ValueError("두 개념이 임베딩상 거의 동일합니다.")
        axis_unit = axis / axis_norm
        mid = (a + b) / 2

        # 중점 기준 잔차를 축 성분과 직교 성분으로 분해한다 (후보 전체 한 번에).
        r = cand_vecs - mid                                  # (N, D)
        r_norm = np.linalg.norm(r, axis=1)                   # (N,)
        axis_proj = r @ axis_unit                            # (N,)
        orth = r - np.outer(axis_proj, axis_unit)            # (N, D)
        orth_ratio = np.linalg.norm(orth, axis=1) / (r_norm + eps)

        sim_a = cand_vecs @ a
        sim_b = cand_vecs @ b
        relevance = (sim_a + sim_b) / 2

        # raw:      원래 정의. 중점에서 멀면서 축 밖에 있는 개념을 고른다.
        # relevant: A·B 영역에 붙어 있으면서 축 밖에 있는 개념을 고른다.
        base = r_norm if score_mode == "raw" else relevance
        score = base * orth_ratio

        results = [
            ConceptScore(
                concept=concept,
                axis_projection=float(axis_proj[i]),
                orthogonal_ratio=float(orth_ratio[i]),
                distinctiveness=float(r_norm[i]),
                relevance=float(relevance[i]),
                sim_to_a=float(sim_a[i]),
                sim_to_b=float(sim_b[i]),
                score=float(score[i]),
            )
            for i, concept in enumerate(candidates)
        ]
        return sorted(results, key=lambda x: -x.score)


def _display_width(text: str) -> int:
    """한글·한자처럼 터미널에서 두 칸을 먹는 글자를 감안한 폭."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _display_width(text))


def pretty_print(results: List[ConceptScore], top_n: int = 10) -> None:
    header = (f"{_pad('개념', 18)}{'score':>8}{'직교율':>8}{'축위치':>9}"
              f"{'거리':>8}{'관련도':>8}{'simA':>8}{'simB':>8}")
    print(header)
    print("-" * _display_width(header))
    for r in results[:top_n]:
        print(f"{_pad(r.concept, 18)}{r.score:8.3f}{r.orthogonal_ratio:8.3f}"
              f"{r.axis_projection:9.3f}{r.distinctiveness:8.3f}"
              f"{r.relevance:8.3f}{r.sim_to_a:8.3f}{r.sim_to_b:8.3f}")


def generate_candidates(concept_a: str, concept_b: str, n: int = 20,
                        model: str = "claude-sonnet-5") -> List[str]:
    """후보를 손으로 넣지 않고 LLM으로 뽑아낸다. ANTHROPIC_API_KEY 필요."""
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model, max_tokens=1000,
        messages=[{"role": "user", "content":
                   f"'{concept_a}'와 '{concept_b}' 둘 다와 관련 있지만 그 대립축 자체와는 "
                   f"무관한 개념을 {n}개, 설명 없이 한 줄에 하나씩 나열해줘."}],
    )
    lines: Iterable[str] = msg.content[0].text.splitlines()
    return [stripped for stripped in (line.strip("-•*0123456789. \t") for line in lines) if stripped]


DEFAULT_CANDIDATES = [
    "임계온도", "마이스너 효과", "쿠퍼쌍", "결정격자 진동",
    "에너지 손실", "자기부상", "위상절연체", "반도체",
    "열전소자", "그래핀", "양자 얽힘", "냉각 시스템",
]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="대립축과 직교하는 제3의 개념을 찾는다.")
    p.add_argument("-a", "--concept-a", default="초전도체")
    p.add_argument("-b", "--concept-b", default="저항체")
    p.add_argument("-c", "--candidate", action="append", dest="candidates",
                   help="후보 개념 (여러 번 지정 가능)")
    p.add_argument("--candidates-file", help="후보 개념을 한 줄에 하나씩 담은 파일")
    p.add_argument("--generate", type=int, metavar="N",
                   help="후보를 Claude로 N개 자동 생성 (ANTHROPIC_API_KEY 필요)")
    p.add_argument("--backend", choices=("sentence-transformers", "spacy"),
                   default="sentence-transformers",
                   help="spacy는 HF 허브를 못 쓰는 환경용 정적 단어벡터 대체 백엔드")
    p.add_argument("--model", help="임베딩 모델/패키지 이름 (백엔드별 기본값 사용)")
    p.add_argument("--score", choices=SCORE_MODES, default="raw",
                   help="raw=거리×직교율(기본), relevant=관련도×직교율")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--json", action="store_true", help="결과를 JSON으로 출력")
    args = p.parse_args(argv)

    candidates: List[str] = list(args.candidates or [])
    if args.candidates_file:
        with open(args.candidates_file, encoding="utf-8") as f:
            candidates += [line.strip() for line in f if line.strip()]
    if args.generate:
        candidates += generate_candidates(args.concept_a, args.concept_b, args.generate)
    if not candidates:
        candidates = DEFAULT_CANDIDATES

    model_name = args.model or (SPACY_PACKAGE if args.backend == "spacy" else MODEL_NAME)
    finder = OrthogonalConceptFinder(model_name=model_name, backend=args.backend)
    try:
        results = finder.find(args.concept_a, args.concept_b, candidates, score_mode=args.score)
    except ValueError as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([asdict(r) for r in results[:args.top]], ensure_ascii=False, indent=2))
    else:
        print(f"축: {args.concept_a} ↔ {args.concept_b}  |  후보 {len(candidates)}개  "
              f"|  score={args.score}  |  backend={args.backend}\n")
        pretty_print(results, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
