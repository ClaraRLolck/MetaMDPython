import numpy as np

def compute_rmsd_and_grad(coord_ref, coord_cur, W=None, translate=True):
    # Avoid unnecessary copies unless we need to translate/center.
    x = np.asarray(coord_ref, dtype=float)  # (n, 3)
    y = np.asarray(coord_cur, dtype=float)  # (n, 3)
    n = x.shape[0]

    if translate:
        x = x - x.mean(axis=0)
        y = y - y.mean(axis=0)

    x_norm = np.sum(x * x)
    y_norm = np.sum(y * y)
    R = x.T @ y

    # Build the symmetric 4x4 K (or S) matrix for quaternion-based optimal rotation
    r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
    r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
    r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]
    S = np.empty((4, 4), dtype=float)
    S[0, 0] = r00 + r11 + r22
    S[0, 1] = r12 - r21
    S[0, 2] = r20 - r02
    S[0, 3] = r01 - r10
    S[1, 0] = S[0, 1]
    S[1, 1] = r00 - r11 - r22
    S[1, 2] = r01 + r10
    S[1, 3] = r02 + r20
    S[2, 0] = S[0, 2]
    S[2, 1] = S[1, 2]
    S[2, 2] = -r00 + r11 - r22
    S[2, 3] = r12 + r21
    S[3, 0] = S[0, 3]
    S[3, 1] = S[1, 3]
    S[3, 2] = S[2, 3]
    S[3, 3] = -r00 - r11 + r22

    evals, evecs = np.linalg.eigh(S)
    # For symmetric matrices, eigh returns sorted eigenvalues.
    lam = evals[-1]
    q = evecs[:, -1]

    # RMSD: sqrt( (x_norm + y_norm - 2*lam) / n )
    val = (x_norm + y_norm) - 2.0 * lam
    if val < 0.0:
        val = 0.0
    error = np.sqrt(val / n) + 1e-9

    # Rotate y directly from quaternion coefficients to avoid building U / BLAS call overhead.
    q0, q1, q2, q3 = q
    b0 = 2.0 * q0
    b1 = 2.0 * q1
    b2 = 2.0 * q2
    b3 = 2.0 * q3
    q00 = b0 * q0 - 1.0
    q01 = b0 * q1
    q02 = b0 * q2
    q03 = b0 * q3
    q11 = b1 * q1
    q12 = b1 * q2
    q13 = b1 * q3
    q22 = b2 * q2
    q23 = b2 * q3
    q33 = b3 * q3

    u00 = q00 + q11
    u01 = q12 - q03
    u02 = q13 + q02
    u10 = q12 + q03
    u11 = q00 + q22
    u12 = q23 - q01
    u20 = q13 - q02
    u21 = q23 + q01
    u22 = q00 + q33

    yy0 = y[:, 0]
    yy1 = y[:, 1]
    yy2 = y[:, 2]
    g = np.empty_like(y)
    g[:, 0] = yy0 * u00 + yy1 * u10 + yy2 * u20
    g[:, 1] = yy0 * u01 + yy1 * u11 + yy2 * u21
    g[:, 2] = yy0 * u02 + yy1 * u12 + yy2 * u22
    g -= x
    g *= -1.0 / (n * error)

    return error, g
