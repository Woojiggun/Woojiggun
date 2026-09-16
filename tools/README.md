# 직교 개념 탐색기 (Orthogonal Concept Finder)

대립 개념 A, B가 임베딩 공간에서 만드는 축을 구하고, 후보 개념들 중 **그 축과
무관하게(직교하게) 서 있는 제3의 개념**을 찾아낸다.

```
r         = v_candidate - (v_A + v_B)/2      # 중점 기준 잔차
axis_proj = r · axis_unit                    # 대립축 위의 위치
orth      = r - axis_proj · axis_unit        # 축 밖 성분
직교율     = |orth| / |r|                     # 1에 가까울수록 축과 무관
```

## 사용

```bash
pip install sentence-transformers numpy

python tools/orthogonal_concept_finder.py                     # 기본 데모 (초전도체 ↔ 저항체)
python tools/orthogonal_concept_finder.py -a 이성 -b 감성 \
    -c 직관 -c 언어 -c 신뢰 --score relevant --top 5
python tools/orthogonal_concept_finder.py --candidates-file cands.txt --json
python tools/orthogonal_concept_finder.py --generate 20        # 후보를 Claude로 자동 생성
```

## 점수 모드

| 모드 | 정의 | 고르는 것 |
| --- | --- | --- |
| `raw` (기본) | `거리 × 직교율` | A·B 중점에서 멀면서 축 밖에 있는 개념 |
| `relevant` | `관련도 × 직교율` | A·B 영역에 붙어 있으면서 축 밖에 있는 개념 |

`raw`는 원래 정의를 그대로 유지한 것이지만, 거리(`distinctiveness`)가 크다는 것은
"A와도 B와도 멀다"는 뜻이라 **주제에서 완전히 벗어난 개념이 상위로 올라오기 쉽다.**
실제 실행에서 `도시 ↔ 시골` 축의 `raw` 1위는 관련도 0.16의 "인터넷"이었고, `relevant`
1위는 관련도 0.40의 "아파트"였다. 원하는 것이 "둘 다와 관련은 있는데 그 대립과는
상관없는 제3의 축"이라면 `--score relevant` 쪽이 의도에 맞는다.

## 임베딩 백엔드

`OrthogonalConceptFinder`는 `encode(sentences, normalize_embeddings=True) -> ndarray`
하나만 만족하면 무엇이든 받는다(`Encoder` 프로토콜).

```python
finder = OrthogonalConceptFinder(model=MyEncoder())     # 직접 주입
finder = OrthogonalConceptFinder(backend="spacy")       # HF 허브 없이 정적 단어벡터
```

`offline_demo_backend.py`는 허브 접근이 막힌 환경(사내망, 에어갭)에서 파이프라인을
확인하려고 붙인 spaCy 정적 단어벡터 백엔드다. 문장 임베딩 모델보다 품질이 낮고,
학습 코퍼스에 드문 전문용어는 서브워드 해싱으로 떨어져 사실상 잡음이 되므로
**본 용도로는 `sentence-transformers` 기본 백엔드를 쓴다.**

## 테스트

임베딩 모델 없이 numpy만으로 수식을 검증한다.

```bash
cd tools && python test_orthogonal_concept_finder.py
```
