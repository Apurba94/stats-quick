# stats-quick

**Descriptive statistics, t-tests (Welch, Student, paired, one-sample) and the Mann-Whitney U
test in pure Python, from a CSV file, with p-values and 95% confidence intervals.**

Part of the *Research Automation Toolkit*: 100 open tools for daily research work.

## Why

Many everyday checks ("is this batch different from the last one?", "did the pilot change the
score?") do not justify starting R or installing SciPy, especially on locked-down lab and
instrument PCs. `stats-quick` needs nothing but Python, and its implementation is short enough to audit.

## Features

- Descriptives: n, mean, SD, SE, quartiles, median, 95% CI of the mean
- Welch (default for unequal variances), Student, paired and one-sample t-tests with CIs of the difference
- Student t distribution via the regularized incomplete beta function (accurate to ~1e-10)
- Mann-Whitney U with tie correction, continuity correction and rank-biserial correlation
- CSV input in long format (`-v value -g group`) or wide paired format (`-v before --paired after`)

## Usage

```bash
pip install .
stats-quick data.csv -v reaction_time -g condition
stats-quick pre_post.csv -v post --paired pre
stats-quick weights.csv -v weight --mu 70 --json
```

## Testing

Validated against tabulated critical values of the t distribution:

```bash
python -m unittest discover -s tests -t .
```

## License

MIT (c) JANIN A APURBA
