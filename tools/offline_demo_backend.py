"""
HF 허브를 못 쓰는 환경(사내망, 에어갭, egress 차단)에서 쓰는 대체 인코더.

SentenceTransformer 대신 spaCy가 GitHub 릴리스로 배포하는 정적 단어벡터를
쓴다. 품질은 문장 임베딩 모델보다 확실히 낮다. 특히 학습 코퍼스에 거의
없던 전문용어(예: "쿠퍼쌍")는 floret 서브워드 해싱으로 떨어져 사실상 잡음이
되므로, 이 백엔드는 "일상 어휘로 파이프라인을 확인하는 용도"로만 쓴다.

준비:
    pip install spacy
    pip install https://github.com/explosion/spacy-models/releases/download/\
ko_core_news_md-3.8.0/ko_core_news_md-3.8.0-py3-none-any.whl
"""

from __future__ import annotations

import pathlib
from typing import Sequence

import numpy as np


class SpacyVectorEncoder:
    """OrthogonalConceptFinder가 요구하는 Encoder 프로토콜의 오프라인 구현."""

    def __init__(self, package: str = "ko_core_news_md"):
        import importlib

        from spacy.vocab import Vocab

        mod = importlib.import_module(package)
        root = pathlib.Path(mod.__file__).parent
        # 패키지 안의 <이름>-<버전>/vocab 만 읽는다. 형태소 분석기(mecab)는 필요 없다.
        (model_dir,) = sorted(root.glob(f"{package}-*"))
        self.vocab = Vocab().from_disk(model_dir / "vocab")

    def encode(self, sentences: Sequence[str], normalize_embeddings: bool = True) -> np.ndarray:
        # 구("마이스너 효과")는 공백 단위로 쪼개 평균낸다.
        vecs = np.array(
            [np.mean([self.vocab.get_vector(tok) for tok in s.split()] or [0.0], axis=0)
             for s in sentences],
            dtype=np.float64,
        )
        if normalize_embeddings:
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs


if __name__ == "__main__":
    from orthogonal_concept_finder import OrthogonalConceptFinder, pretty_print

    finder = OrthogonalConceptFinder(model=SpacyVectorEncoder())

    cases = [
        ("도시", "시골",
         ["교통", "공기", "인구", "학교", "병원", "농업",
          "아파트", "자연", "인터넷", "음악", "시간", "소음"]),
        ("여름", "겨울",
         ["온도", "눈", "해변", "난방", "습도", "옷",
          "여행", "음악", "사진", "시간", "바람", "감기"]),
    ]

    for a, b, candidates in cases:
        for mode in ("raw", "relevant"):
            print(f"\n=== 축: {a} ↔ {b}  (score={mode}) ===")
            pretty_print(finder.find(a, b, candidates, score_mode=mode), top_n=6)
