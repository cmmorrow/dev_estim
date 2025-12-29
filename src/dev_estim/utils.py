from math import copysign, fabs


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
