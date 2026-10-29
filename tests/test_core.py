import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from stats_quick.cli import main
from stats_quick.core import (betainc, describe, mann_whitney_u, one_sample_t_test, paired_t_test, student_t_test,
                              t_cdf, t_ppf, welch_t_test)


class DistributionTests(unittest.TestCase):
    def test_t_cdf_known_values(self):
        self.assertAlmostEqual(t_cdf(0, 5), 0.5)
        self.assertAlmostEqual(t_cdf(2.228, 10), 0.975, places=3)      # tabulated critical value
        self.assertAlmostEqual(t_cdf(-1.812, 10), 0.05, places=3)
        self.assertAlmostEqual(t_cdf(1.96, 1e6), 0.975, places=3)      # approaches the normal

    def test_ppf_inverts_cdf(self):
        self.assertAlmostEqual(t_ppf(0.975, 10), 2.228, places=3)
        self.assertAlmostEqual(t_ppf(0.995, 3), 5.841, places=3)

    def test_betainc_symmetry(self):
        self.assertAlmostEqual(betainc(2, 3, 0.4) + betainc(3, 2, 0.6), 1.0)


class TestTests(unittest.TestCase):
    X = [5.1, 4.9, 6.2, 5.8, 6.0, 5.5, 5.3]
    Y = [6.8, 7.1, 6.5, 7.4, 6.9, 7.0, 7.3]

    def test_welch_and_student_agree_in_sign(self):
        w, s = welch_t_test(self.X, self.Y), student_t_test(self.X, self.Y)
        self.assertLess(w["t"], -6)
        self.assertLess(w["p"], 1e-4)
        self.assertEqual(s["df"], 12)
        self.assertLess(w["df"], 12)
        self.assertLess(w["ci"][1], 0)

    def test_identical_groups(self):
        r = welch_t_test([1, 2, 3, 4], [1, 2, 3, 4])
        self.assertAlmostEqual(r["t"], 0)
        self.assertAlmostEqual(r["p"], 1)

    def test_one_sample_and_paired(self):
        r = one_sample_t_test([2, 4, 6, 8], mu=5)   # mean 5 -> t 0
        self.assertAlmostEqual(r["p"], 1.0)
        p = paired_t_test([10, 12, 14, 16], [9, 10, 12, 13])
        self.assertAlmostEqual(p["mean_diff"], 2.0)
        self.assertEqual(p["df"], 3)
        with self.assertRaises(ValueError):
            paired_t_test([1, 2], [1])

    def test_mann_whitney(self):
        r = mann_whitney_u(self.X, self.Y)
        self.assertEqual(r["U"], 0)
        self.assertAlmostEqual(r["rank_biserial"], -1.0)
        self.assertLess(r["p"], 0.01)
        tie = mann_whitney_u([1, 1, 1], [1, 1, 1])
        self.assertEqual(tie["p"], 1.0)

    def test_describe(self):
        d = describe([1, 2, 3, 4, 5])
        self.assertEqual((d["median"], d["q1"], d["q3"]), (3, 2, 4))
        self.assertAlmostEqual(d["ci95"][0], 3 - 2.776 * d["se"], places=2)

    def test_cli_groups(self):
        with tempfile.TemporaryDirectory() as dd:
            p = os.path.join(dd, "d.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("group,y\n" + "".join("a,%s\n" % v for v in self.X) + "".join("b,%s\n" % v for v in self.Y))
            with redirect_stdout(io.StringIO()) as buf:
                main([p, "-v", "y", "-g", "group"])
            self.assertIn("Welch two-sample t-test", buf.getvalue())
            self.assertIn("Mann-Whitney U: U = 0.0", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
