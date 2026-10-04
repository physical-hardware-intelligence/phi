"""The file view: roots bound what can be read, secrets stay out, search skips heavy folders."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from phi.studio import files as F
from phi.studio.files import FileError, Files, Root


def tree(tmp: Path) -> Files:
    code = tmp / "code"
    (code / "src").mkdir(parents=True)
    (code / "src" / "rig.py").write_text("PORT = '/dev/tty.usbmodem123'\n")
    (code / "robot-config.yaml").write_text("robot:\n  port: /dev/tty.usbmodem123\n")
    (code / ".env").write_text("HF_TOKEN=hf_secret\n")
    (code / "hf_token.txt").write_text("usbmodem hf_secret\n")
    (code / ".git").mkdir()
    (code / ".git" / "config").write_text("usbmodem\n")
    (code / "node_modules" / "x").mkdir(parents=True)
    (code / "node_modules" / "x" / "a.js").write_text("usbmodem\n")
    (code / "datasets").mkdir()
    (code / "datasets" / "meta.json").write_text("usbmodem\n")
    (code / "src" / "data").mkdir()
    (code / "src" / "data" / "loader.py").write_text("usbmodem in a code package called data\n")
    (code / "blob.bin").write_bytes(b"usbmodem\0\0\0")
    (code / "big.log").write_text("usbmodem\n" * 200_000)  # > 1 MB
    cal = tmp / "cal"
    (cal / "robots" / "so_follower").mkdir(parents=True)
    (cal / "robots" / "so_follower" / "phi_follower.json").write_text("{}")
    (tmp / "outside.txt").write_text("not yours")
    return Files([Root("code", "Code", code.resolve()), Root("calibration", "Cal", cal.resolve())])


@pytest.mark.parametrize("root,rel,why", [
    ("code", "../outside.txt", "outside"),
    ("code", "/etc/hosts", "outside"),
    ("code", ".env", "secrets"),
    ("code", "hf_token.txt", "secrets"),
    ("code", ".git/config", "secrets"),
    ("nope", "src/rig.py", "no file root"),
    ("code", "", "path"),
    ("code", "blob.bin", "binary"),
    ("code", "src", "not a file"),
])  # fmt: skip
def test_reads_outside_a_root_or_of_secrets_are_refused(tmp_path: Path, root, rel, why) -> None:
    with pytest.raises(FileError, match=why):
        tree(tmp_path).read(root, rel)


def test_a_symlink_out_of_a_root_is_refused(tmp_path: Path) -> None:
    f = tree(tmp_path)
    (tmp_path / "code" / "link.txt").symlink_to(tmp_path / "outside.txt")
    with pytest.raises(FileError, match="outside"):
        f.read("code", "link.txt")


def test_read_returns_text_and_cuts_large_files(tmp_path: Path) -> None:
    f = tree(tmp_path)
    got = f.read("code", "robot-config.yaml")
    assert "usbmodem123" in got["text"] and got["path"] == "robot-config.yaml"
    assert not got["truncated"]
    big = f.read("code", "big.log")
    assert big["truncated"] and len(big["text"]) == F.MAX_READ


def test_search_skips_secrets_vcs_dependencies_heavy_and_binary_files(tmp_path: Path) -> None:
    res = tree(tmp_path).search("USBMODEM")  # case-insensitive
    found = {(h["root"], h["path"]) for h in res["hits"]}
    assert found == {("code", "src/rig.py"), ("code", "robot-config.yaml"),
                     ("code", "src/data/loader.py")}  # fmt: skip
    assert not res["stopped"]


def test_search_stops_at_its_limit_and_rejects_tiny_queries(tmp_path: Path) -> None:
    f = tree(tmp_path)
    res = f.search("usbmodem", limit=2)
    assert len(res["hits"]) == 2 and res["stopped"]
    with pytest.raises(FileError):
        f.search("u")


def test_notes_and_calibrations_are_listed(tmp_path: Path) -> None:
    ix = tree(tmp_path).index()
    assert [n["path"] for n in ix["notes"]] == ["robot-config.yaml"]
    assert ix["calibrations"][0]["id"] == "phi_follower"
    assert ix["calibrations"][0]["kind"] == "so_follower"


def test_rig_config_is_the_first_note_even_from_the_second_root(tmp_path: Path) -> None:
    code, repo = tmp_path / "code", tmp_path / "repo"
    (code / "docs" / "robots" / "so-arm101").mkdir(parents=True)
    (code / "docs" / "robots" / "so-arm101" / "02-setup.md").write_text("setup\n")
    repo.mkdir()
    (repo / "robot-config.yaml").write_text("robot: {}\n")
    f = Files([Root("code", "Code", code.resolve()), Root("repo", "Repo", repo.resolve())])
    assert [(n["root"], n["path"]) for n in f.notes()] == [
        ("repo", "robot-config.yaml"),
        ("code", "docs/robots/so-arm101/02-setup.md"),
    ]


def test_locate_maps_a_path_to_its_root(tmp_path: Path) -> None:
    f = tree(tmp_path)
    assert f.locate("src/rig.py") == ("code", "src/rig.py")
    assert f.locate(str(tmp_path / "code" / "src" / "rig.py")) == ("code", "src/rig.py")
    assert f.locate(str(tmp_path / "outside.txt")) is None


def test_main_checkout_of_a_worktree_is_found(tmp_path: Path) -> None:
    main = tmp_path / "main"
    main.mkdir()
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "-C", str(main), "init", "-q"], check=True)
    subprocess.run([*git, "-C", str(main), "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    wt = tmp_path / "wt"
    subprocess.run([*git, "-C", str(main), "worktree", "add", "-q", str(wt)], check=True)
    assert F.main_checkout(wt).resolve() == main.resolve()
    assert F.main_checkout(tmp_path / "not-a-repo") is None


def test_ports_name_the_arm_on_each(monkeypatch: pytest.MonkeyPatch) -> None:
    import serial.tools.list_ports as lp

    fake = [SimpleNamespace(device="/dev/cu.usbmodem5B7B0096441", description="USB Single Serial",
                            vid=0x1A86, pid=0x55D3, serial_number="5B7B009644", manufacturer=None),
            SimpleNamespace(device="/dev/cu.debug-console", description="n/a", vid=None, pid=None,
                            serial_number=None, manufacturer=None)]  # fmt: skip
    monkeypatch.setattr(lp, "comports", lambda: fake)
    got = F.list_ports([{"name": "left_follower", "port": "/dev/tty.usbmodem5B7B0096441"}])
    first, second = got["ports"]
    assert first["arm"] == "left_follower" and first["tty"] == "/dev/tty.usbmodem5B7B0096441"
    assert first["usb"] and not second["usb"] and second["description"] is None
