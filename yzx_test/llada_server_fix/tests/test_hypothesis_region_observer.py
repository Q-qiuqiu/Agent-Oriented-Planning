import torch

from hypothesis_region_observer import HypothesisRegionObserver


class CharacterTokenizer:
    mask_token_id = 250
    all_special_ids = [0, 250]

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(token)) for token in ids)


def make_observer(**kwargs):
    return HypothesisRegionObserver(
        tokenizer=CharacterTokenizer(),
        catalog=["search_agent", "calculation_agent"],
        prompt_length=3,
        gen_length=128,
        mask_id=250,
        score_chunk_size=16,
        **kwargs,
    )


def snapshot(position):
    return {"refined_value_start": position}


def observation(number):
    return {
        "observation": number,
        "iteration": number * 32,
        "wall_time": float(number),
    }


def update(observer, number, positions):
    snapshots = {
        track_id: snapshot(position) for track_id, position in positions
    }
    current = observation(number)
    active = observer._assign_active_tracks(snapshots, current)
    centers = observer._update_hypotheses(active, current)
    observer._suppress_duplicates(centers)
    return active, centers


def update_with_merge(observer, number, positions):
    snapshots = {
        track_id: snapshot(position) for track_id, position in positions
    }
    current = observation(number)
    active = observer._assign_active_tracks(snapshots, current)
    centers = observer._update_hypotheses(active, current)
    centers, active = observer._merge_close_hypotheses(
        centers, active, current
    )
    observer._suppress_duplicates(centers)
    return active, centers


def test_nearby_tracks_merge_into_one_validated_hypothesis():
    observer = make_observer()
    canvas = torch.full((1, 131), 250, dtype=torch.long)
    observer.initialize(canvas)

    update(observer, 1, [(0, 100), (1, 103)])
    update(observer, 2, [(0, 101), (1, 104), (2, 102)])

    assert len(observer._hypotheses) == 1
    hypothesis = observer._hypotheses[0]
    assert hypothesis["member_track_ids"] == [0, 1, 2]
    assert hypothesis["seen_count"] == 2
    assert hypothesis["support_ratio"] == 1.0
    assert hypothesis["validated"] is True
    assert hypothesis["center_history"] == [101, 102]


def test_short_lived_hypothesis_fails_final_support_validation():
    observer = make_observer()
    canvas = torch.full((1, 131), 250, dtype=torch.long)
    observer.initialize(canvas)

    update(observer, 1, [(0, 100)])
    update(observer, 2, [])
    update(observer, 3, [])

    hypothesis = observer._hypotheses[0]
    assert hypothesis["seen_count"] == 1
    assert hypothesis["support_ratio"] == 1 / 3
    assert hypothesis["validated"] is False


def test_established_tracks_merge_after_their_median_positions_converge():
    observer = make_observer(hypothesis_max_center_jump=100)
    canvas = torch.full((1, 131), 250, dtype=torch.long)
    observer.initialize(canvas)

    update_with_merge(observer, 1, [(0, 100), (1, 112)])
    update_with_merge(observer, 2, [(0, 104), (1, 108)])
    update_with_merge(observer, 3, [(0, 104), (1, 108)])

    roots = [
        row for row in observer._hypotheses if row["active_root"]
    ]
    assert len(roots) == 1
    assert sorted(roots[0]["member_track_ids"]) == [0, 1]


def test_nearby_successor_track_merges_across_two_observation_gap():
    observer = make_observer()
    canvas = torch.full((1, 131), 250, dtype=torch.long)
    observer.initialize(canvas)

    update_with_merge(observer, 1, [(0, 100)])
    update_with_merge(observer, 2, [])
    update_with_merge(observer, 3, [])
    update_with_merge(observer, 4, [(1, 104)])

    assert len(observer._hypotheses) == 1
    assert observer._hypotheses[0]["member_track_ids"] == [0, 1]


def test_converging_hypotheses_suppress_weaker_duplicate():
    observer = make_observer(hypothesis_max_center_jump=100)
    canvas = torch.full((1, 131), 250, dtype=torch.long)
    observer.initialize(canvas)

    update(observer, 1, [(0, 100), (1, 110)])
    update(observer, 2, [(0, 100), (1, 110)])
    update(observer, 3, [(0, 100), (1, 110)])
    update(observer, 4, [(0, 102), (1, 106)])
    update(observer, 5, [(0, 103), (1, 107)])

    assert len(observer._hypotheses) == 2
    duplicates = [row for row in observer._hypotheses if row["duplicate"]]
    assert len(duplicates) == 1
    assert duplicates[0]["suppressed_by"] is not None


def test_large_center_jump_starts_a_new_unvalidated_lifecycle():
    observer = make_observer()
    canvas = torch.full((1, 131), 250, dtype=torch.long)
    observer.initialize(canvas)

    update(observer, 1, [(0, 100)])
    update(observer, 2, [(0, 130)])

    assert len(observer._hypotheses) == 1
    assert observer._hypothesis_by_track[0] == 0
    hypothesis = observer._hypotheses[0]
    assert hypothesis["validated"] is False
    assert hypothesis["lifecycle_reset_count"] == 1
    assert len(hypothesis["archived_lifecycles"]) == 1
    assert hypothesis["archived_lifecycles"][0]["center_history"] == [100]
    assert hypothesis["center_history"] == [130]
