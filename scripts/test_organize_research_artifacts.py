"""Filesystem safety tests; all mutations are confined to temporary Git repos."""

import csv
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).with_name("organize_research_artifacts.py")
SPEC = importlib.util.spec_from_file_location("organizer", SCRIPT)
organizer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = organizer
SPEC.loader.exec_module(organizer)


class OrganizerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)

    def put(self, relative, content=b"sample\r\n\x00\xff\n"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def snapshot(self):
        return {
            str(p.relative_to(self.root)): p.read_bytes()
            for p in self.root.rglob("*")
            if p.is_file() and ".git" not in p.relative_to(self.root).parts
        }

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--root", str(self.root), *args],
            text=True, capture_output=True,
        )

    def apply(self):
        result = self.run_cli("--apply")
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_default_dry_run_does_not_write(self):
        self.put("rte/tmp_old.csv")
        before = self.snapshot()
        for args in ((), ("--dry-run",), ("--inventory-only",)):
            result = self.run_cli(*args)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("DRY RUN", result.stdout)
            self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.root / organizer.ARCHIVE).exists())

    def test_move_preserves_bytes_mode_manifest_and_idempotence(self):
        original = self.put("rte/tmp_old.csv")
        original.chmod(0o640)
        content = original.read_bytes()
        self.put("rte/TMP_TABLE.tex", b"historical table\n")
        self.put("rte/core.py", b"pass\n")
        result = self.apply()
        self.assertIn("APPLIED: 2 moves", result.stdout)
        destination = self.root / organizer.SMOKE / original.name
        self.assertEqual(destination.read_bytes(), content)
        self.assertEqual(destination.stat().st_mode & 0o777, 0o640)
        self.assertFalse(original.exists())
        moves = organizer.verify_archive(self.root)
        self.assertEqual(len(moves), 2)
        self.assertEqual(self.run_cli("--verify").returncode, 0)
        before = self.snapshot()
        self.assertIn("APPLIED: 0 moves", self.apply().stdout)
        self.assertEqual(self.snapshot(), before)

    def test_references_self_references_stems_and_escaped_names_are_retained(self):
        self.put("rte/tmp_data.csv")
        self.put("rte/tmp_data.tex")
        self.put("rte/TMP_SELF.md", b"original output: rte/TMP_SELF.md\n")
        self.put("rte/tmp_url.csv")
        self.put("rte/tmp_free.csv")
        report = self.put("rte/report.tex", b"\\input{tmp\\_data}\nlink: tmp%5Furl.csv\n")
        before = report.read_bytes()
        result = self.apply()
        self.assertIn("APPLIED: 1 moves", result.stdout)
        self.assertTrue((self.root / "rte/tmp_data.csv").exists())
        self.assertTrue((self.root / "rte/tmp_data.tex").exists())
        self.assertTrue((self.root / "rte/TMP_SELF.md").exists())
        self.assertTrue((self.root / "rte/tmp_url.csv").exists())
        self.assertEqual(report.read_bytes(), before)

    def test_ignored_configs_protect_references_but_are_not_indexed(self):
        self.put(".gitignore", b"*.json\n*.log\n")
        self.put("local.json", b'{"result": "rte/tmp_configured.csv"}\n')
        self.put("rte/tmp_configured.csv")
        self.put("rte/tmp_ignored.log", b"generated log\n")
        self.put("rte/tmp_free.csv")
        self.apply()
        self.assertTrue((self.root / "rte/tmp_configured.csv").exists())
        inventory = (self.root / organizer.INVENTORY).read_text()
        self.assertNotIn("local.json", inventory)
        self.assertNotIn("tmp_ignored.log", inventory)

    def test_only_allowlisted_root_scratch_files_move(self):
        retained = ["rte/tmp_code.py", "rte/TMP_RUN.sh", "rte/tmp_build.aux",
                    "rte/main_results.csv", "rte/MAIN_REPORT.md", "rte/nested/tmp_old.csv"]
        for path in retained:
            self.put(path)
        self.put("rte/tmp_old.csv")
        self.apply()
        for path in retained:
            self.assertTrue((self.root / path).exists(), path)

    def test_existing_destination_aborts_entire_plan(self):
        self.put("rte/tmp_a.csv")
        self.put("rte/tmp_z.csv")
        self.put("rte/archive/smoke/tmp_z.csv", b"other data")
        before = self.snapshot()
        result = self.run_cli("--apply")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Destination already exists", result.stderr)
        self.assertEqual(self.snapshot(), before)

    def test_symlink_source_and_parent_are_refused(self):
        target = self.put("rte/real.csv")
        link = self.root / "rte/tmp_link.csv"
        link.symlink_to(target)
        self.assertNotEqual(self.run_cli("--apply").returncode, 0)
        link.unlink()
        self.put("rte/tmp_old.csv")
        (self.root / "elsewhere").mkdir()
        (self.root / "rte/archive").symlink_to(self.root / "elsewhere")
        self.assertNotEqual(self.run_cli("--apply").returncode, 0)
        self.assertTrue((self.root / "rte/tmp_old.csv").exists())

    def test_changed_source_or_new_reference_invalidates_plan(self):
        source = self.put("rte/tmp_old.csv")
        moves, _ = organizer.make_plan(self.root)
        source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Plan changed"):
            organizer.apply_moves(self.root, moves)
        moves, _ = organizer.make_plan(self.root)
        self.put("rte/report.md", b"tmp_old.csv\n")
        with self.assertRaisesRegex(ValueError, "Plan changed"):
            organizer.apply_moves(self.root, moves)
        self.assertFalse((self.root / organizer.ARCHIVE).exists())

    def test_link_failure_preserves_every_original(self):
        self.put("rte/tmp_a.csv")
        self.put("rte/tmp_b.csv")
        moves, _ = organizer.make_plan(self.root)
        real_link = organizer.os.link

        def fail_second(source, destination):
            if Path(source).name == "tmp_b.csv":
                raise OSError("simulated interruption")
            return real_link(source, destination)

        with mock.patch.object(organizer.os, "link", side_effect=fail_second):
            with self.assertRaises(OSError):
                organizer.apply_moves(self.root, moves)
        for move in moves:
            self.assertEqual(organizer.fingerprint(self.root, move.source), (move.size, move.sha256))
        self.assertEqual((self.root / organizer.SMOKE / "tmp_a.csv").read_bytes(), (self.root / "rte/tmp_a.csv").read_bytes())

    def test_exclusive_link_handles_destination_race_without_clobber(self):
        self.put("rte/tmp_old.csv")
        moves, _ = organizer.make_plan(self.root)
        real_link = organizer.os.link

        def race(source, destination):
            Path(destination).write_bytes(b"concurrent file")
            return real_link(source, destination)

        with mock.patch.object(organizer.os, "link", side_effect=race):
            with self.assertRaises(FileExistsError):
                organizer.apply_moves(self.root, moves)
        self.assertEqual((self.root / organizer.SMOKE / "tmp_old.csv").read_bytes(), b"concurrent file")
        self.assertTrue((self.root / "rte/tmp_old.csv").exists())

    def test_interruption_after_manifest_keeps_all_archive_bytes(self):
        self.put("rte/tmp_a.csv")
        self.put("rte/tmp_b.csv")
        moves, _ = organizer.make_plan(self.root)
        real_unlink = Path.unlink

        def interrupt_second(path, *args, **kwargs):
            if path == self.root / "rte/tmp_b.csv":
                raise OSError("simulated interruption after first original removal")
            return real_unlink(path, *args, **kwargs)

        with mock.patch.object(Path, "unlink", interrupt_second):
            with self.assertRaises(OSError):
                organizer.apply_moves(self.root, moves)
        self.assertEqual(organizer.manifest_moves(self.root), moves)
        for move in moves:
            self.assertEqual(
                organizer.fingerprint(self.root, move.destination),
                (move.size, move.sha256),
            )
        self.assertFalse((self.root / "rte/tmp_a.csv").exists())
        self.assertTrue((self.root / "rte/tmp_b.csv").exists())
        self.assertIn("Original path still exists", self.run_cli("--verify").stderr)

    def test_verify_detects_corruption_recreated_source_and_stale_index(self):
        self.put("rte/tmp_old.csv")
        self.apply()
        destination = self.root / organizer.SMOKE / "tmp_old.csv"
        original = destination.read_bytes()
        destination.write_bytes(b"corrupt")
        self.assertIn("Archived content differs", self.run_cli("--verify").stderr)
        destination.write_bytes(original)
        source = self.put("rte/tmp_old.csv", original)
        self.assertIn("Original path still exists", self.run_cli("--verify").stderr)
        source.unlink()
        self.put("new_notes.md", b"new notes")
        self.assertIn("Inventory missing or stale", self.run_cli("--verify").stderr)

    def test_inventory_only_is_reproducible_and_tracks_ignored_committed_files(self):
        self.put(".gitignore", b"*.json\n")
        self.put("config.json", b"{}\n")
        subprocess.run(["git", "-C", str(self.root), "add", "-f", "config.json"], check=True)
        self.put("rte/tmp_old.csv")
        self.put("rte/history_trace.csv")
        result = self.run_cli("--inventory-only", "--apply")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "rte/tmp_old.csv").exists())
        self.assertFalse((self.root / organizer.SMOKE).exists())
        before = self.snapshot()
        self.assertEqual(self.run_cli("--inventory-only", "--apply").returncode, 0)
        self.assertEqual(self.snapshot(), before)
        with (self.root / organizer.INVENTORY).open() as handle:
            rows = {r["path"]: r["group"] for r in csv.DictReader(handle, delimiter="\t")}
        self.assertEqual(rows["config.json"], "core code")
        self.assertEqual(rows["rte/tmp_old.csv"], "scratch")
        self.assertEqual(rows["rte/history_trace.csv"], "traces")

    def test_non_generated_index_is_not_overwritten(self):
        self.put("rte/tmp_old.csv")
        self.put(str(organizer.INDEX), b"user-owned index\n")
        before = self.snapshot()
        result = self.run_cli("--apply")
        self.assertIn("Refusing to overwrite", result.stderr)
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
