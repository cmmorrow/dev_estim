from math import copysign, fabs, log, sqrt


def brent_root(f, a, b, maxiter=100, tol=1e-12):
    """Find a root of f in [a, b] using Brent's method.
    Requires f(a) and f(b) to have opposite signs.
    tol is an absolute tolerance on the root position.
    """
    fa = f(a)
    fb = f(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if copysign(1.0, fa) == copysign(1.0, fb):
        raise ValueError("f(a) and f(b) must have opposite signs")

    c, fc = a, fa
    d = e = b - a  # step sizes
    for _ in range(maxiter):
        if fb == 0:
            return b
        # Ensure |f(b)| <= |f(c)|
        if fabs(fc) < fabs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol_act = 2.0 * tol * max(1.0, fabs(b))
        m = 0.5 * (c - b)

        # Check convergence
        if fabs(m) <= tol_act:
            return b

        if fabs(e) < tol_act or fabs(fa) <= fabs(fb):
            # Bisection
            d = e = m
        else:
            # Inverse quadratic interpolation or secant
            s = fb / fa
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            p = fabs(p)
            # Accept interpolation only if it stays within bounds
            if 2.0 * p < min(3.0 * m * q - fabs(tol_act * q), fabs(e * q)):
                e = d
                d = p / q
            else:
                d = e = m

        a, fa = b, fb
        if fabs(d) > tol_act:
            b += d
        else:
            b += copysign(tol_act, m)
        fb = f(b)
        # Update c if sign changes
        if copysign(1.0, fb) == copysign(1.0, fc):
            c, fc = a, fa
            d = e = b - a
        # Otherwise keep c and fc

    raise RuntimeError("Maximum iterations exceeded")


def normal_quantile(p: float) -> float:
    """Return z such that normal_cdf(z) == p (0 < p < 1) using Acklam's approximation."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")

    # Coefficients for Acklam's approximation
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow, phigh = 0.02425, 1 - 0.02425

    if p < plow:
        q = sqrt(-2 * log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if phigh < p:
        q = sqrt(-2 * log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )
