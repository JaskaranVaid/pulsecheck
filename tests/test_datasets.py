"""Video manifest, FaceForensics++ and Celeb-DF loaders, and the two scripts around them.

Everything runs on fake trees of empty files in ``tmp_path``; nothing is decoded and nothing
touches the network.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from pulsecheck.datasets.celebdf import list_celebdf_test
from pulsecheck.datasets.ffpp import FFPP_METHODS, list_ffpp, load_splits
from pulsecheck.datasets.manifest import VideoItem, read_manifest, write_manifest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _write_splits(directory: Path, train, val, test) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, pairs in (("train", train), ("val", val), ("test", test)):
        (directory / f"{name}.json").write_text(json.dumps(pairs))
    return directory


def _splits(train=(("001", "002"),), val=(("005", "006"),), test=(("003", "004"),)):
    return {
        "train": [tuple(p) for p in train],
        "val": [tuple(p) for p in val],
        "test": [tuple(p) for p in test],
    }


def _ffpp_original(root: Path, c: str, vid: str) -> Path:
    return _touch(root / "original_sequences" / "youtube" / c / "videos" / f"{vid}.mp4")


def _ffpp_fake(root: Path, m: str, c: str, name: str) -> Path:
    return _touch(root / "manipulated_sequences" / m / c / "videos" / f"{name}.mp4")


# ---- splits ---------------------------------------------------------------------------------


def test_load_splits_validates_shape_and_disjointness(tmp_path):
    good = _write_splits(
        tmp_path / "good", [["953", "974"], ["012", "026"]], [["100", "101"]], [["200", "201"]]
    )
    splits = load_splits(good)
    assert splits == {
        "train": [("953", "974"), ("012", "026")],
        "val": [("100", "101")],
        "test": [("200", "201")],
    }

    nested = _write_splits(tmp_path / "nested", [[["953", "974"]]], [["100", "101"]], [])
    with pytest.raises(ValueError, match="train.json"):
        load_splits(nested)

    triple = _write_splits(tmp_path / "triple", [["953", "974", "001"]], [], [])
    with pytest.raises(ValueError, match="train.json"):
        load_splits(triple)

    bad_id = _write_splits(tmp_path / "bad_id", [["95", "974"]], [], [])
    with pytest.raises(ValueError, match="95"):
        load_splits(bad_id)

    not_a_list = tmp_path / "not_a_list"
    not_a_list.mkdir()
    for name in ("train", "val", "test"):
        (not_a_list / f"{name}.json").write_text('{"a": 1}')
    with pytest.raises(ValueError, match="JSON array"):
        load_splits(not_a_list)

    leak = _write_splits(tmp_path / "leak", [["001", "002"]], [["003", "004"]], [["004", "005"]])
    with pytest.raises(ValueError, match="004"):
        load_splits(leak)

    twice = _write_splits(tmp_path / "twice", [["001", "002"], ["002", "003"]], [], [])
    with pytest.raises(ValueError, match="002"):
        load_splits(twice)


# ---- FaceForensics++ ------------------------------------------------------------------------


def test_list_ffpp_pairs_both_directions_groups_and_missing(tmp_path):
    root = tmp_path / "ffpp"
    for vid in ("001", "002", "003", "004"):
        _ffpp_original(root, "c23", vid)
    for name in ("001_002", "002_001", "003_004"):  # 004_003 is missing
        _ffpp_fake(root, "Deepfakes", "c23", name)
    splits = _splits(val=())

    items, missing = list_ffpp(root, "c23", splits, methods=("Deepfakes",))

    fakes = {i.rel_path for i in items if i.label == 1}
    assert fakes == {
        "manipulated_sequences/Deepfakes/c23/videos/001_002.mp4",
        "manipulated_sequences/Deepfakes/c23/videos/002_001.mp4",
        "manipulated_sequences/Deepfakes/c23/videos/003_004.mp4",
    }
    assert [p.name for p in missing] == ["004_003.mp4"]
    assert {i.group for i in items if i.split == "test"} == {"ffpp:003_004"}
    assert {i.group for i in items if i.split == "train"} == {"ffpp:001_002"}

    originals = [i for i in items if i.label == 0]
    assert {i.rel_path for i in originals} == {
        f"original_sequences/youtube/c23/videos/{v}.mp4" for v in ("001", "002", "003", "004")
    }
    assert {i.kind for i in originals} == {"original"}
    assert {i.kind for i in items if i.label == 1} == {"Deepfakes"}
    assert {i.compression for i in items} == {"c23"}
    assert {i.dataset for i in items} == {"ffpp"}
    for item in items:
        assert item.path.is_absolute() and item.path.is_file()
        assert item.root == root


def test_list_ffpp_group_is_order_independent(tmp_path):
    root = tmp_path / "ffpp"
    _ffpp_original(root, "c40", "974")
    _ffpp_fake(root, "Face2Face", "c40", "953_974")
    items, _ = list_ffpp(
        root, "c40", _splits(train=(("974", "953"),), val=(), test=()), methods=("Face2Face",)
    )
    assert {i.group for i in items} == {"ffpp:953_974"}


def test_list_ffpp_missing_originals_and_other_methods(tmp_path):
    root = tmp_path / "ffpp"
    _ffpp_original(root, "c23", "001")  # 002 missing
    items, missing = list_ffpp(root, "c23", _splits(val=(), test=()))
    assert [i.rel_path for i in items] == ["original_sequences/youtube/c23/videos/001.mp4"]
    # 1 original + 4 methods x 2 directions
    assert len(missing) == 1 + len(FFPP_METHODS) * 2
    assert sum(p.name == "002.mp4" for p in missing) == 1


def test_list_ffpp_max_pairs(tmp_path):
    root = tmp_path / "ffpp"
    train = (("001", "002"), ("011", "012"), ("021", "022"))
    for a, b in train:
        _ffpp_original(root, "c23", a)
        _ffpp_original(root, "c23", b)
        _ffpp_fake(root, "FaceSwap", "c23", f"{a}_{b}")
        _ffpp_fake(root, "FaceSwap", "c23", f"{b}_{a}")
    splits = _splits(train=train, val=(), test=())
    items, missing = list_ffpp(root, "c23", splits, methods=("FaceSwap",), max_pairs={"train": 2})
    assert not missing
    assert {i.group for i in items} == {"ffpp:001_002", "ffpp:011_012"}  # the first two, JSON order
    assert len(items) == 2 * (2 + 2)


def test_list_ffpp_rejects_bad_arguments(tmp_path):
    with pytest.raises(ValueError, match="compression"):
        list_ffpp(tmp_path, "c99", _splits())
    with pytest.raises(ValueError, match="method"):
        list_ffpp(tmp_path, "c23", _splits(), methods=("Nope",))
    with pytest.raises(ValueError, match="max_pairs"):
        list_ffpp(tmp_path, "c23", _splits(), max_pairs={"trian": 1})


# ---- Celeb-DF -------------------------------------------------------------------------------


def _celebdf_tree(root: Path, lines: list[str], files: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "List_of_testing_videos.txt").write_text("\n".join(lines) + "\n")
    for rel in files:
        _touch(root / rel)


def test_celebdf_parses_and_self_checks_labels(tmp_path):
    root = tmp_path / "celebdf"
    rels = [
        "Celeb-real/id0_0000.mp4",
        "Celeb-synthesis/id0_id3_0000.mp4",
        "YouTube-real/00170.mp4",
    ]
    _celebdf_tree(
        root,
        [f"1 {rels[0]}", f"0 {rels[1]}", "", f"1 {rels[2]}", "1 Celeb-real/id9_0001.mp4"],
        rels,  # the last listed file does not exist
    )
    items, missing = list_celebdf_test(root)

    assert [i.rel_path for i in items] == rels
    assert [i.label for i in items] == [0, 1, 0]
    assert [i.group for i in items] == ["id0", "id0", "yt:00170"]
    assert [i.kind for i in items] == ["Celeb-real", "Celeb-synthesis", "YouTube-real"]
    assert {i.split for i in items} == {"test"}
    assert {i.compression for i in items} == {""}
    assert {i.dataset for i in items} == {"celebdf"}
    assert all(i.root == root for i in items)
    assert [p.name for p in missing] == ["id9_0001.mp4"]


@pytest.mark.parametrize(
    "line",
    [
        "0 Celeb-real/id0_0000.mp4",  # opposite convention for a real folder
        "0 YouTube-real/00170.mp4",
        "1 Celeb-synthesis/id0_id3_0000.mp4",
        "2 Celeb-real/id0_0000.mp4",  # not a label
        "1 Elsewhere/id0_0000.mp4",  # unknown folder
        "Celeb-real/id0_0000.mp4",  # no label
        "1 Celeb-real/../id0_0000.mp4",  # escapes the dataset root
    ],
)
def test_celebdf_bad_lines_raise_naming_the_line(tmp_path, line):
    root = tmp_path / "celebdf"
    _celebdf_tree(root, ["1 Celeb-real/id1_0000.mp4", line], ["Celeb-real/id1_0000.mp4"])
    with pytest.raises(ValueError, match=r"List_of_testing_videos\.txt:2"):
        list_celebdf_test(root)


def test_celebdf_missing_list_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_celebdf_test(tmp_path)


# ---- manifest -------------------------------------------------------------------------------


def _item(**kw) -> VideoItem:
    fields = {
        "dataset": "ffpp",
        "path": Path("/data/ffpp/original_sequences/youtube/c23/videos/001.mp4"),
        "rel_path": "original_sequences/youtube/c23/videos/001.mp4",
        "label": 0,
        "group": "ffpp:001_002",
        "kind": "original",
        "compression": "c23",
        "split": "train",
    }
    fields.update(kw)
    return VideoItem(**fields)


def test_video_item_root_is_path_minus_rel_path():
    item = _item()
    assert item.root == Path("/data/ffpp")
    with pytest.raises(ValueError, match="rel_path"):
        _item(rel_path="elsewhere/001.mp4").root  # noqa: B018


def test_manifest_roundtrip_and_bad_label(tmp_path):
    items = [
        _item(),
        _item(
            path=Path("/data/ffpp/manipulated_sequences/Deepfakes/c23/videos/001_002.mp4"),
            rel_path="manipulated_sequences/Deepfakes/c23/videos/001_002.mp4",
            label=1,
            kind="Deepfakes",
        ),
        _item(
            dataset="celebdf",
            path=Path("/data/celebdf/YouTube-real/00170.mp4"),
            rel_path="YouTube-real/00170.mp4",
            group="yt:00170",
            kind="YouTube-real",
            compression="",
            split="test",
        ),
    ]
    path = tmp_path / "out" / "m.csv"  # parent directory is created
    write_manifest(items, path)
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == "dataset,path,rel_path,label,group,kind,compression,split"
    assert read_manifest(path) == items

    text = path.read_text(encoding="utf-8")
    bad = tmp_path / "bad_label.csv"
    bad.write_text(text.replace(",0,ffpp:001_002", ",2,ffpp:001_002"), encoding="utf-8")
    with pytest.raises(ValueError, match="label"):
        read_manifest(bad)
    text_label = tmp_path / "text_label.csv"
    text_label.write_text(text.replace(",0,ffpp:001_002", ",fake,ffpp:001_002"), encoding="utf-8")
    with pytest.raises(ValueError, match="label"):
        read_manifest(text_label)

    no_col = tmp_path / "no_col.csv"
    no_col.write_text("dataset,path,rel_path,label\nffpp,/x,y,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="group"):
        read_manifest(no_col)

    bad_split = tmp_path / "bad_split.csv"
    bad_split.write_text(text.replace(",train", ",training"), encoding="utf-8")
    with pytest.raises(ValueError, match="split"):
        read_manifest(bad_split)


# ---- scripts/build_manifest.py ---------------------------------------------------------------


def _build_trees(tmp_path: Path):
    ffpp = tmp_path / "ffpp"
    for c in ("c23", "c40"):
        for vid in ("001", "002", "003", "004", "005", "006"):
            _ffpp_original(ffpp, c, vid)
        for name in ("001_002", "002_001", "003_004", "005_006", "006_005"):
            _ffpp_fake(ffpp, "Deepfakes", c, name)  # 004_003 missing
    splits_dir = _write_splits(
        tmp_path / "splits", [["001", "002"]], [["005", "006"]], [["003", "004"]]
    )
    celeb = tmp_path / "celebdf"
    rels = ["Celeb-real/id0_0000.mp4", "Celeb-synthesis/id0_id3_0000.mp4"]
    _celebdf_tree(celeb, [f"1 {rels[0]}", f"0 {rels[1]}"], rels)
    return ffpp, celeb, splits_dir


def test_build_manifest_main_writes_csvs_and_report(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("PULSECHECK_DATA", str(tmp_path / "data"))
    ffpp, celeb, splits_dir = _build_trees(tmp_path)
    script = _load_script("build_manifest")
    out = tmp_path / "manifests"

    code = script.main(
        [
            "--ffpp-root", str(ffpp),
            "--celebdf-root", str(celeb),
            "--splits-dir", str(splits_dir),
            "--compression", "c23", "c40",
            "--max-train-pairs", "5",
            "--methods", "Deepfakes",
            "--out-dir", str(out),
        ]
    )  # fmt: skip
    text = capsys.readouterr().out
    assert code == 0
    assert {p.name for p in out.iterdir()} == {"ffpp_c23.csv", "ffpp_c40.csv", "celebdf_test.csv"}

    c23 = read_manifest(out / "ffpp_c23.csv")
    assert {i.compression for i in c23} == {"c23"}
    assert len(c23) == 6 + 5  # six originals, five fakes
    assert {i.split for i in c23} == {"train", "val", "test"}
    celeb_items = read_manifest(out / "celebdf_test.csv")
    assert [(i.kind, i.label) for i in celeb_items] == [("Celeb-real", 0), ("Celeb-synthesis", 1)]

    # split report: counts per split x label x kind, and missing per kind
    assert "ffpp_c23" in text and "celebdf_test" in text
    assert any(ln.split() == ["test", "1", "Deepfakes", "1"] for ln in text.splitlines()), text
    assert any(ln.split() == ["train", "0", "original", "2"] for ln in text.splitlines()), text
    assert "missing: Deepfakes 1" in text
    assert "missing: none" in text  # celebdf


def test_build_manifest_defaults_to_data_dir_and_skips_missing_roots(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("PULSECHECK_DATA", str(tmp_path / "data"))
    _, celeb, _ = _build_trees(tmp_path)
    script = _load_script("build_manifest")

    code = script.main(["--ffpp-root", str(tmp_path / "nope"), "--celebdf-root", str(celeb)])
    captured = capsys.readouterr()
    assert code == 0
    assert (tmp_path / "data" / "manifests" / "celebdf_test.csv").is_file()
    assert not (tmp_path / "data" / "manifests" / "ffpp_c23.csv").exists()
    assert "skipping" in (captured.out + captured.err)

    # nothing could be built at all -> non-zero
    assert script.main(["--ffpp-root", str(tmp_path / "nope")]) == 1


def test_build_manifest_needs_splits_for_ffpp(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("PULSECHECK_DATA", str(tmp_path / "data"))
    ffpp, _, _ = _build_trees(tmp_path)
    script = _load_script("build_manifest")
    assert script.main(["--ffpp-root", str(ffpp)]) == 2
    assert "fetch_ffpp_splits" in capsys.readouterr().err


# ---- scripts/fetch_ffpp_splits.py ------------------------------------------------------------


def _fake_fetch(payloads: dict[str, object]):
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        name = url.rsplit("/", 1)[1].removesuffix(".json")
        return json.dumps(payloads[name]).encode()

    fetch.calls = calls
    return fetch


def test_fetch_ffpp_splits_downloads_validates_and_writes(tmp_path):
    script = _load_script("fetch_ffpp_splits")
    fetch = _fake_fetch(
        {"train": [["001", "002"]], "val": [["005", "006"]], "test": [["003", "004"]]}
    )
    out = tmp_path / "splits"
    assert script.main(["--out", str(out)], fetch=fetch) == 0
    assert sorted(p.name for p in out.iterdir()) == ["test.json", "train.json", "val.json"]
    assert load_splits(out)["test"] == [("003", "004")]
    assert fetch.calls == [
        f"https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/{n}.json"
        for n in ("train", "val", "test")
    ]


def test_fetch_ffpp_splits_default_dir_and_invalid_download(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PULSECHECK_DATA", str(tmp_path / "data"))
    script = _load_script("fetch_ffpp_splits")

    bad = _fake_fetch(
        {"train": [["001", "002"]], "val": [["002", "003"]], "test": [["003", "004"]]}
    )  # ids shared between splits
    assert script.main([], fetch=bad) == 1
    assert "002" in capsys.readouterr().err
    assert not (
        tmp_path / "data" / "ffpp" / "splits" / "train.json"
    ).exists()  # nothing left behind

    good = _fake_fetch(
        {"train": [["001", "002"]], "val": [["005", "006"]], "test": [["003", "004"]]}
    )
    assert script.main([], fetch=good) == 0
    assert (tmp_path / "data" / "ffpp" / "splits" / "val.json").is_file()
