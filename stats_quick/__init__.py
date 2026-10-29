"""Pure-Python descriptive statistics and common two-group tests."""
from .core import describe, t_cdf, welch_t_test, student_t_test, paired_t_test, one_sample_t_test, mann_whitney_u

__version__ = "0.1.0"
__all__ = ["describe", "t_cdf", "welch_t_test", "student_t_test", "paired_t_test", "one_sample_t_test",
           "mann_whitney_u"]
