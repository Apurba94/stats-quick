"""Student t distribution via the regularized incomplete beta function, t-tests and Mann-Whitney U."""
import math
import statistics

_NORMAL = statistics.NormalDist()


def _betacf(a, b, x, max_iter=300, eps=3e-14):
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c if abs(1 + aa / c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c if abs(1 + aa / c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def t_cdf(t, df):
    """CDF of Student's t distribution."""
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2, 0.5, x)
    return 1 - tail if t > 0 else tail


def t_ppf(p, df, tol=1e-10):
    """Quantile of the t distribution by bisection (adequate for CIs)."""
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def _p_two_sided(t, df):
    return 2 * (1 - t_cdf(abs(t), df))


def describe(x):
    x = sorted(x)
    n = len(x)
    if n == 0:
        raise ValueError("no data")
    mean = statistics.fmean(x)
    sd = statistics.stdev(x) if n > 1 else 0.0
    q = statistics.quantiles(x, n=4, method="inclusive") if n > 1 else [x[0]] * 3
    res = {"n": n, "mean": mean, "sd": sd, "se": sd / math.sqrt(n), "min": x[0], "q1": q[0], "median": q[1],
           "q3": q[2], "max": x[-1]}
    if n > 1:
        tcrit = t_ppf(0.975, n - 1)
        res["ci95"] = (mean - tcrit * res["se"], mean + tcrit * res["se"])
    return res


def _result(name, t, df, diff, se, alpha=0.05):
    tcrit = t_ppf(1 - alpha / 2, df)
    return {"test": name, "t": t, "df": df, "p": _p_two_sided(t, df), "mean_diff": diff,
            "ci": (diff - tcrit * se, diff + tcrit * se)}


def welch_t_test(x, y):
    nx, ny = len(x), len(y)
    vx, vy = statistics.variance(x), statistics.variance(y)
    se = math.sqrt(vx / nx + vy / ny)
    diff = statistics.fmean(x) - statistics.fmean(y)
    df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    return _result("Welch two-sample t-test", diff / se, df, diff, se)


def student_t_test(x, y):
    nx, ny = len(x), len(y)
    sp2 = ((nx - 1) * statistics.variance(x) + (ny - 1) * statistics.variance(y)) / (nx + ny - 2)
    se = math.sqrt(sp2 * (1 / nx + 1 / ny))
    diff = statistics.fmean(x) - statistics.fmean(y)
    return _result("Student two-sample t-test (equal variances)", diff / se, nx + ny - 2, diff, se)


def one_sample_t_test(x, mu=0.0):
    n = len(x)
    se = statistics.stdev(x) / math.sqrt(n)
    diff = statistics.fmean(x) - mu
    return _result("one-sample t-test", diff / se, n - 1, diff, se)


def paired_t_test(x, y):
    if len(x) != len(y):
        raise ValueError("paired samples must have equal length")
    r = one_sample_t_test([a - b for a, b in zip(x, y)])
    r["test"] = "paired t-test"
    return r


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    ties = []
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        if j > i:
            ties.append(j - i + 1)
        i = j + 1
    return ranks, ties


def mann_whitney_u(x, y, continuity=True):
    """Two-sided Mann-Whitney U with normal approximation and tie correction.

    Also returns the rank-biserial correlation as an effect size."""
    nx, ny = len(x), len(y)
    ranks, ties = _ranks(list(x) + list(y))
    r1 = sum(ranks[:nx])
    u1 = r1 - nx * (nx + 1) / 2
    u2 = nx * ny - u1
    n = nx + ny
    tie_term = sum(t ** 3 - t for t in ties) / (n * (n - 1)) if n > 1 else 0
    sigma = math.sqrt(nx * ny / 12 * ((n + 1) - tie_term))
    mu = nx * ny / 2
    if sigma == 0:
        z, p = 0.0, 1.0
    else:
        diff = abs(u1 - mu) - (0.5 if continuity else 0)
        z = max(diff, 0) / sigma * (1 if u1 >= mu else -1)
        p = 2 * (1 - _NORMAL.cdf(abs(z)))
    return {"test": "Mann-Whitney U", "U": min(u1, u2), "U1": u1, "z": z, "p": min(1.0, p),
            "rank_biserial": 1 - 2 * u2 / (nx * ny) if nx * ny else 0.0}
