"""End-to-end checks against an independent scalar Euclidean oracle."""
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import prepare


def write_fvecs(path, rows):
    with path.open("wb") as out:
        for row in rows:
            out.write(struct.pack(f"<i{len(row)}f", len(row), *row))


def read_ivecs(path):
    rows = []
    with path.open("rb") as stream:
        while header := stream.read(4):
            k, = struct.unpack("<i", header)
            rows.append(list(struct.unpack(f"<{k}i", stream.read(k * 4))))
    return rows


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = prepare.build_backend()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Different modality dimensions, deliberate equal-distance ties, and a
        # pair where mixing squared distances REVERSES the Euclidean ranking.
        self.b1 = [(0., 0.), (1., 2.)] + [(10. + i, float(i % 3)) for i in range(198)]
        self.b2 = [(4., 0., 0.), (1., 2., 0.)] + [(10. + i, 0., 1.) for i in range(198)]
        self.b1[8], self.b2[8] = self.b1[7], self.b2[7]
        self.q1 = [(0., 0.), (30., 1.), (90., 0.)]
        self.q2 = [(0., 0., 0.), (32., 0., 1.), (95., 0., 2.)]
        for name, rows in zip(prepare.NAMES, (self.b1, self.b2, self.q1, self.q2)):
            write_fvecs(self.root / name, rows)

    def run_pipeline(self, name="out", *extra, success=True):
        result = subprocess.run([sys.executable, str(prepare.HERE / "prepare.py"), str(self.root),
                                 "--output", str(self.root / name), "--backend", str(self.backend),
                                 "--k", "5", *extra], capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return self.root / name

    def check_all_states(self, out):
        manifest = json.loads((out / "manifest.json").read_text())
        alpha_bytes = (out / "query_alpha.fvecs").read_bytes()
        alphas = []
        for dim, value in struct.iter_unpack("<if", alpha_bytes):
            self.assertEqual(dim, 1)
            alphas.append(value)
        for name in prepare.NAMES[:2]:
            self.assertEqual((out / "base_for_insert" / name).read_bytes(),
                             (self.root / name).read_bytes()[:100 * (12 if "img" in name else 16)])
        jobs = [manifest["static_gt"]]
        for mode, experiment in manifest["experiments"].items():
            self.assertEqual(len(experiment["traces"]), 10)
            self.assertEqual(len(experiment["states"]), 11)
            active = set(range(100 if mode == "insert" else 200))
            for r, trace in enumerate(experiment["traces"]):
                integers = list(struct.unpack("<3i", (out / trace["file"]).read_bytes()))
                start = (100 if mode == "insert" else 0) + 2 * r
                self.assertEqual(integers, [2, start, start + 1])
                before = experiment["states"][r]["active_range"]
                self.assertEqual(active, set(range(*before)))
                if mode == "insert":
                    self.assertTrue(set(integers[1:]).isdisjoint(active))
                    active.update(integers[1:])
                else:
                    active.difference_update(integers[1:])
                self.assertEqual(active, set(range(*experiment["states"][r + 1]["active_range"])))
            jobs.extend(experiment["states"])
        for job in jobs:
            rows = read_ivecs(out / job["file"])
            self.assertEqual(len(rows), len(self.q1))
            begin, end = job["active_range"]
            for qi, actual in enumerate(rows):
                a = alphas[qi]
                expected = sorted(range(begin, end), key=lambda j: (
                    a * math.dist(self.q1[qi], self.b1[j]) +
                    (1-a) * math.dist(self.q2[qi], self.b2[j]), j))[:5]
                self.assertEqual(actual, expected, (job["file"], qi))
        for path, entry in manifest["outputs"].items():
            self.assertEqual(hashlib.sha256((out / path).read_bytes()).hexdigest(), entry["sha256"])

    def test_sqrt_and_every_update_state(self):
        out = self.run_pipeline("out", "--alpha-min", "0.5", "--alpha-max", "0.5")
        self.check_all_states(out)
        self.assertEqual(read_ivecs(out / "static_gt.ivecs")[0][:2], [0, 1])
        # A squared-distance implementation incorrectly ranks ID 1 before ID 0.

    def test_random_alpha_and_thread_reproducibility(self):
        first = self.run_pipeline("one", "--threads", "1")
        second = self.run_pipeline("three", "--threads", "3")
        self.check_all_states(first)
        self.assertEqual(json.loads((first / "manifest.json").read_text())["outputs"],
                         json.loads((second / "manifest.json").read_text())["outputs"])

    def test_invalid_inputs_and_existing_output(self):
        for flag, value in (("--rounds", "100"), ("--k", "201"), ("--alpha-min", "2"),
                            ("--update-rate", "0.001"), ("--update-rate", "nan")):
            self.run_pipeline("bad", flag, value, success=False)
            self.assertFalse((self.root / "bad").exists())
        (self.root / "exists").mkdir()
        self.run_pipeline("exists", success=False)
        write_fvecs(self.root / prepare.NAMES[1], self.b2[:-1])
        self.run_pipeline("mismatch", success=False)
        self.assertFalse((self.root / "mismatch").exists())

    def test_corrupted_fvecs(self):
        path = self.root / prepare.NAMES[0]
        with path.open("ab") as stream:
            stream.write(b"x")
        self.run_pipeline("broken", success=False)
        self.assertFalse((self.root / "broken").exists())


if __name__ == "__main__":
    unittest.main()
