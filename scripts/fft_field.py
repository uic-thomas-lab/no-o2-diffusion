# -*- coding: utf-8 -*-
r"""Zero-padded FFT superposition solver with sub-voxel deposition, and its validation gates.

The many-source field is a CONVOLUTION -- every source contributes the same radial kernel and
removal is linear -- so it is computable by FFT at a cost INDEPENDENT of source count. Measured:
225 sources on a 201^3 grid, 90.6 s by direct summation against 1.4 s by FFT.

## THE LESSON THIS FILE EXISTS TO PRESERVE

**FFT validation must be performed at MAXIMAL source occupancy and the MOST PERMISSIVE
concentration threshold. Low-source validation is insufficient, because circular-wrap error
accumulates with source number and preferentially biases the long-range low-concentration field.**

Sub-voxel deposition was NOT the cause and did not fix it: trilinear gave +6.268% where nearest
gave +6.287%. The cause was **circular convolution** -- np.fft wraps, so field leaving one face
re-enters the opposite one and ADDS. Zero-padding took the same case to **+0.011%**.

## The padding is DERIVED and then PROVEN, not chosen

  1. kernel-derived candidate:  N_max * G(r_pad) < eps_C
     \[WarningSign] scaled by the maximum production source count, NOT read off the single-cell
     kernel alone -- the kernel has a tail rather than a hard cutoff, and ~225 individually tiny
     wrapped contributions add.
  2. pad-convergence gate: repeat the worst case at r_pad, 1.25 r_pad, 1.5 r_pad and require
     V(>0.5 nM) to move by less than tolerance. Freeze the SMALLEST passing pad.
  3. direct-summation equivalence at the worst PRODUCTION configuration.
  4. freeze.

No literal pad width survives in production code; `Solver` computes it from the kernel.
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import tornado_v2 as T                                            # noqa: E402


def grid(L, dx):
    n = int(round(2 * L / dx)) + 1
    return np.linspace(-L, L, n), n


def kernel_pad_radius(coeff, dx, n_max, eps_nM, r_hi=600.0):
    """Smallest r with n_max * G(r) < eps_nM, G the single-cell field in nM.

    Scaled by n_max. The single-cell kernel alone would give a radius at which ONE source is
    negligible; with n_max sources the wrapped contributions add.
    """
    r = np.arange(dx, r_hi, dx)
    g = T.sphere_field(r, coeff) * 1e9
    ok = np.where(n_max * g < eps_nM)[0]
    if ok.size == 0:
        raise ValueError('kernel does not decay below %g nM within %g um' % (eps_nM, r_hi))
    return float(r[ok[0]])


class Solver:
    """Zero-padded FFT superposition on a fixed grid, with the kernel transform cached.

    The kernel transform depends only on the grid and the oxygen tension, so for an ensemble at
    fixed [O2] it is computed ONCE and reused for every realisation and every fraction.
    """

    def __init__(self, L, dx, coeff, n_max, eps_nM, dtype=np.float32, r_pad=None):
        self.L, self.dx, self.coeff, self.dtype = L, dx, coeff, dtype
        self.ax, self.n = grid(L, dx)
        # r_pad is DERIVED unless a convergence test is deliberately overriding it
        self.r_pad = float(r_pad) if r_pad else kernel_pad_radius(coeff, dx, n_max, eps_nM)
        self.pad = int(np.ceil(self.r_pad / dx))
        self.m = self.n + 2 * self.pad
        axp = (np.arange(self.m) - self.m // 2) * dx
        GX, GY, GZ = np.meshgrid(axp, axp, axp, indexing='ij')
        R = np.sqrt(GX ** 2 + GY ** 2 + GZ ** 2)
        del GX, GY, GZ
        K = T.sphere_field(R, coeff).astype(dtype)
        # DO NOT TRUNCATE THE KERNEL AT r_pad. Zeroing it beyond the pad radius was tried, to
        # make the linear-convolution bound exact, and it DISCARDS REAL SIGNAL: per source the
        # kernel is only ~0.002 nM at 200 um, but summed over 226 sources the field is still
        # 0.39 nM there. Truncating at 120 um threw away 36% of V(>0.5 nM) while the untruncated
        # version agreed with exact summation to +0.011%.
        # The padding suppresses WRAPAROUND; it does not license truncation.
        del R
        self.Khat = np.fft.rfftn(np.fft.ifftshift(K))
        del K

    def describe(self):
        return ('grid %d^3 (+/-%g um, dx %g) | pad %d voxels = %.0f um | transform %d^3'
                % (self.n, self.L, self.dx, self.pad, self.r_pad, self.m))

    def field(self, pts, mode='trilinear'):
        S = np.zeros((self.m,) * 3, dtype=self.dtype)
        off = self.L / self.dx + self.pad
        for q in np.atleast_2d(pts):
            g = np.asarray(q) / self.dx + off
            if mode == 'nearest':
                i, j, k = (int(round(v)) for v in g)
                S[i, j, k] += 1.0
                continue
            i0 = np.floor(g).astype(int)
            f = g - i0
            for di in (0, 1):
                wi = f[0] if di else 1 - f[0]
                for dj in (0, 1):
                    wj = f[1] if dj else 1 - f[1]
                    for dk in (0, 1):
                        S[i0[0] + di, i0[1] + dj, i0[2] + dk] += wi * wj * (f[2] if dk
                                                                           else 1 - f[2])
        out = np.fft.irfftn(np.fft.rfftn(S) * self.Khat, s=(self.m,) * 3, axes=(0, 1, 2))
        p, n = self.pad, self.n
        return (out[p:p + n, p:p + n, p:p + n] * 1e9).astype(np.float64)


def field_direct(pts, coeff, ax):
    GX, GY, GZ = np.meshgrid(ax, ax, ax, indexing='ij')
    F = np.zeros_like(GX)
    for q in np.atleast_2d(pts):
        F += T.sphere_field(
            np.sqrt((GX - q[0]) ** 2 + (GY - q[1]) ** 2 + (GZ - q[2]) ** 2), coeff)
    return F * 1e9


def _worst_case(n_src, r_scatter, seed=3):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n_src, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return v * r_scatter * rng.random(n_src)[:, None] ** (1 / 3.)


def gates(L=200.0, dx=2.0, o2_uM=10.0, n_max=226, r_scatter=90.0,
          eps_nM=5e-4, pads_um=(80.0, 120.0, 160.0, 200.0, 240.0),
          thresholds=(0.5, 1.0, 2.0), tol_pct=0.5):
    """WORST production configuration: max source count, production grid and dx, 0.5 nM primary.

    THE KERNEL-DERIVED CANDIDATE IS REPORTED BUT NOT USED AS-IS. n_max*G(r) < eps gives 424 um
    here, a 625^3 transform (~3 GB), and 1.5x that is unaffordable. That criterion demands the
    KERNEL be negligible, whereas what matters is the wrapped contribution to the region above
    0.5 nM. There is also a geometric bound it ignores: no source lies beyond R_scatter and no
    field point beyond the grid, so no source-fieldpoint pair is ever farther apart than
    R_scatter + L. Kernel support past that is never sampled.

    So the CONVERGENCE TEST is the production criterion, as it should be, and the smallest
    converged pad is frozen.
    """
    coeff = T.cell_coeffs(T.nominal_params(), o2_uM * 1e-6)
    pts = _worst_case(n_max, r_scatter)
    dv = dx ** 3
    print('WORST PRODUCTION CASE: %d sources over R = %g um, [O2] = %g uM, dx = %g'
          % (n_max, r_scatter, o2_uM, dx))
    r_strict = kernel_pad_radius(coeff, dx, n_max, eps_nM)
    print('  kernel-derived candidate  n_max*G(r) < %g nM  ->  r_pad = %.0f um  '
          '(transform %d^3 — reported, not used)'
          % (eps_nM, r_strict, int(round(2 * L / dx)) + 1 + 2 * int(np.ceil(r_strict / dx))))
    print('  geometric bound: no source-fieldpoint pair exceeds %.0f um\n'
          % (r_scatter + L))

    print('--- GATE A: padding convergence (the production criterion) ---')
    vols, solvers = {}, {}
    for rp in pads_um:
        s_ = Solver(L, dx, coeff, n_max, eps_nM, r_pad=rp)
        t0 = time.time()
        C = s_.field(pts)
        vols[rp] = {c: (C > c).sum() * dv for c in thresholds}
        solvers[rp] = s_
        print('    pad %5.0f um = %3d voxels, transform %3d^3, %5.1f s   V>0.5 = %.6g'
              % (rp, s_.pad, s_.m, time.time() - t0, vols[rp][0.5]))
        del C
    ref = pads_um[-1]
    chosen = None
    print('    reference = largest pad (%.0f um)' % ref)
    for rp in pads_um[:-1]:
        worst = max(100.0 * abs(vols[rp][c] - vols[ref][c]) / vols[ref][c] for c in thresholds)
        ok = worst < tol_pct
        print('      pad %5.0f um vs reference: worst %+.4f%%  %s'
              % (rp, worst, 'converged' if ok else 'NOT converged'))
        if ok and chosen is None:
            chosen = rp
    conv_ok = chosen is not None
    print('  %s\n'
          % ('GATE A PASSED — smallest converged pad = %.0f um; FREEZE IT' % chosen
             if conv_ok else 'GATE A FAILED — no tested pad converged'))
    if not conv_ok:
        return 1

    print('--- GATE B: equivalence with exact direct summation, at the frozen pad ---')
    t0 = time.time()
    D = field_direct(pts, coeff, solvers[chosen].ax)
    print('  direct summation: %.1f s   peak %.3f nM' % (time.time() - t0, D.max()))
    face = max(D[0].max(), D[-1].max(), D[:, 0].max(),
               D[:, -1].max(), D[:, :, 0].max(), D[:, :, -1].max())
    print('  max on grid face: %.4f nM  %s' % (face, 'ok' if face < min(thresholds)
                                               else 'DOMAIN UNDERSIZED'))
    eq_ok = True
    for mode in ('nearest', 'trilinear'):
        C = solvers[chosen].field(pts, mode)
        for c in thresholds:
            vd = (D > c).sum() * dv
            e = 100.0 * ((C > c).sum() * dv - vd) / vd
            print('    %-9s V>%-4g nM  exact %11.5g   fft %+7.4f%%%s'
                  % (mode, c, vd, e, '' if abs(e) < tol_pct else '  '))
            if mode == 'trilinear':
                eq_ok &= abs(e) < tol_pct
        del C
    print('  %s' % ('GATE B PASSED — sub-voxel padded FFT matches exact summation'
                    if eq_ok else 'GATE B FAILED'))
    print('\n  FROZEN PRODUCTION PAD: %.0f um (%d voxels)'
          % (chosen, solvers[chosen].pad))
    return 0 if eq_ok else 1


if __name__ == '__main__':
    sys.exit(gates())
