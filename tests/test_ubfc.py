import numpy as np
import pytest

from pulsecheck.datasets.ubfc import list_subjects, load_ground_truth


def _subject(root, name, video=True, gt=True):
    d = root / name
    d.mkdir()
    if video:
        (d / "vid.avi").touch()
    if gt:
        (d / "ground_truth.txt").touch()
    return d


def test_list_subjects_sorts_numerically(tmp_path):
    for n in (10, 3, 1):
        _subject(tmp_path, f"subject{n}")
    subjects = list_subjects(tmp_path)
    assert [s.subject_id for s in subjects] == ["subject1", "subject3", "subject10"]
    assert subjects[0].video == tmp_path / "subject1" / "vid.avi"
    assert subjects[0].ground_truth == tmp_path / "subject1" / "ground_truth.txt"


def test_list_subjects_skips_incomplete_and_unrelated_dirs(tmp_path):
    _subject(tmp_path, "subject2")
    _subject(tmp_path, "subject4", gt=False)
    _subject(tmp_path, "subject5", video=False)
    _subject(tmp_path, "notes")
    (tmp_path / "subject9.txt").touch()
    assert [s.subject_id for s in list_subjects(tmp_path)] == ["subject2"]


def test_list_subjects_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope"):
        list_subjects(tmp_path / "nope")


def test_load_ground_truth_three_lines(tmp_path):
    p = tmp_path / "ground_truth.txt"
    p.write_text("0.1 0.2 0.3\n70 71 72\n0.0 0.033 0.066\n")
    t, ppg, hr = load_ground_truth(p)
    np.testing.assert_allclose(ppg, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(hr, [70, 71, 72])
    np.testing.assert_allclose(t, [0.0, 0.033, 0.066])


def test_load_ground_truth_accepts_scientific_notation_and_blank_lines(tmp_path):
    p = tmp_path / "gt.txt"
    p.write_text("1e-1 2.0E-1\n\n70  71\n0 3.3e-2\n\n")
    t, ppg, _hr = load_ground_truth(p)
    np.testing.assert_allclose(ppg, [0.1, 0.2])
    np.testing.assert_allclose(t, [0.0, 0.033])


@pytest.mark.parametrize(
    "text",
    [
        "",  # empty
        "1 2 3\n4 5 6\n",  # two lines
        "1 2 3\n4 5 6\n7 8 9\n1 2 3\n",  # four lines
        "1 2 3\n4 5\n7 8 9\n",  # unequal lengths
        "1 2 3\n4 x 6\n7 8 9\n",  # non-numeric
        "1 2 nan\n4 5 6\n7 8 9\n",  # non-finite
        "1\n2\n3\n",  # a single sample per line
    ],
)
def test_load_ground_truth_is_strict(tmp_path, text):
    p = tmp_path / "ground_truth.txt"
    p.write_text(text)
    with pytest.raises(ValueError, match="ground_truth.txt"):
        load_ground_truth(p)
