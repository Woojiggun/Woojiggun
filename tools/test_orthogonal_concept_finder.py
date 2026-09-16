"""임베딩 모델 없이 수식만 검증하는 테스트 (numpy만 있으면 돈다)."""

import numpy as np

from orthogonal_concept_finder import OrthogonalConceptFinder


class FakeEncoder:
    """개념 이름 -> 미리 정해둔 단위벡터. 3차원이면 손으로 검산할 수 있다."""

    def __init__(self, table):
        self.table = table

    def encode(self, sentences, normalize_embeddings=True):
        vecs = np.array([self.table[s] for s in sentences], dtype=np.float64)
        if normalize_embeddings:
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs


def _finder(table):
    return OrthogonalConceptFinder(model=FakeEncoder(table))


def test_pure_orthogonal_beats_on_axis():
    # A=+x, B=-x 이므로 축은 x. z축 개념은 완전 직교, x축 개념은 완전 정렬.
    table = {
        "A": [1, 0, 0],
        "B": [-1, 0, 0],
        "축_위의_개념": [1, 0, 0],
        "직교_개념": [0, 0, 1],
    }
    results = _finder(table).find("A", "B", ["축_위의_개념", "직교_개념"])
    assert results[0].concept == "직교_개념"
    assert np.isclose(results[0].orthogonal_ratio, 1.0)
    assert abs(results[0].axis_projection) < 1e-9
    assert results[1].orthogonal_ratio < 1e-9


def test_scores_match_hand_computation():
    table = {"A": [1, 0, 0], "B": [-1, 0, 0], "C": [0, 0, 1]}
    (r,) = _finder(table).find("A", "B", ["C"])
    # 중점은 원점이므로 잔차 = C 자신, 길이 1, 축(x) 성분 0.
    assert np.isclose(r.distinctiveness, 1.0)
    assert np.isclose(r.score, 1.0)
    assert np.isclose(r.sim_to_a, 0.0) and np.isclose(r.sim_to_b, 0.0)
    assert np.isclose(r.relevance, 0.0)


def test_relevant_mode_prefers_in_domain_concept():
    # 먼 개념과 가까운 개념 둘 다 축과 직교. raw는 먼 쪽, relevant는 가까운 쪽을 고른다.
    table = {
        "A": [1, 1, 0],
        "B": [-1, 1, 0],       # 축 = x, 둘 다 y성분을 공유
        "영역_안_개념": [0, 1, 0.2],
        "영역_밖_개념": [0, -1, 0.2],
    }
    raw = _finder(table).find("A", "B", ["영역_안_개념", "영역_밖_개념"], score_mode="raw")
    rel = _finder(table).find("A", "B", ["영역_안_개념", "영역_밖_개념"], score_mode="relevant")
    assert raw[0].concept == "영역_밖_개념"
    assert rel[0].concept == "영역_안_개념"


def test_identical_concepts_raise():
    table = {"A": [1, 0, 0], "B": [1, 0, 0], "C": [0, 1, 0]}
    try:
        _finder(table).find("A", "B", ["C"])
    except ValueError as e:
        assert "거의 동일" in str(e)
    else:
        raise AssertionError("동일한 두 개념에 대해 ValueError가 나와야 한다")


def test_empty_candidates_and_bad_mode():
    table = {"A": [1, 0, 0], "B": [-1, 0, 0]}
    assert _finder(table).find("A", "B", []) == []
    try:
        _finder(table).find("A", "B", ["A"], score_mode="없는모드")
    except ValueError:
        pass
    else:
        raise AssertionError("잘못된 score_mode는 ValueError여야 한다")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\n전부 통과")
