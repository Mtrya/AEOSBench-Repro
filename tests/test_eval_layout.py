import json

from aeosbench.evaluation.layout import (
    AnnotationSelection,
    load_annotations,
    raw_scenario_ids,
    scenario_refs,
    trajectory_metrics_path,
    trajectory_payload_path,
)


def test_load_annotations_supports_dict_payload(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text('{"ids":[1,2],"epochs":[3,4]}', encoding="utf-8")

    selection = load_annotations(path)

    assert selection == AnnotationSelection(ids=[1, 2], epochs=[3, 4])
    assert selection.epoch_at(1) == 4


def test_load_annotations_supports_legacy_list(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    selection = load_annotations(path)

    assert selection == AnnotationSelection(ids=[1, 2, 3], epochs=None)
    assert selection.epoch_at(0, default=1) == 1


def test_scenario_refs_follow_annotation_order_for_real_data():
    refs = scenario_refs("val_seen", limit=3)

    assert [ref.id_ for ref in refs] == [238, 267, 425]
    assert [ref.epoch for ref in refs] == [1, 1, 1]


def test_test_official64_uses_annotation_ids():
    refs = scenario_refs("test_official64", limit=3)

    assert [ref.id_ for ref in refs] == [194, 718, 819]
    assert [ref.epoch for ref in refs] == [1, 1, 1]
    assert [ref.split for ref in refs] == ["test", "test", "test"]


def test_test_random64_is_deterministic():
    refs = scenario_refs("test_random64")

    assert len(refs) == 64
    assert [ref.id_ for ref in refs[:10]] == [6, 25, 27, 30, 32, 44, 89, 94, 95, 99]
    assert refs[-1].id_ == 996
    assert all(ref.split == "test" for ref in refs)
    assert all(ref.epoch == 1 for ref in refs)


def test_test_all_discovers_raw_test_pool():
    ids = raw_scenario_ids("test")
    refs = scenario_refs("test_all")

    assert len(ids) == 1000
    assert ids[:3] == [0, 1, 2]
    assert ids[-3:] == [997, 998, 999]
    assert [ref.id_ for ref in refs[:3]] == [0, 1, 2]
    assert [ref.id_ for ref in refs[-3:]] == [997, 998, 999]
    assert all(ref.split == "test" for ref in refs)


def test_trajectory_paths_resolve_epoch_roots():
    metrics_path = trajectory_metrics_path("train", 42, epoch=3)
    payload_path = trajectory_payload_path("train", 42, epoch=2)

    assert metrics_path.as_posix().endswith("data/trajectories.3/train/00/00042.json")
    assert payload_path.as_posix().endswith("data/trajectories.2/train/00/00042.pth")
