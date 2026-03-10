/*
 * phi_propagate.cl
 * GARVIS · OpenCL-Agent-Hub · φ-locked lattice propagation kernel
 *
 * Each work-item handles one lattice node i:
 *
 *   C[i] = clamp( O[i] * A[i] * B[i] * φ,  0, 1 )
 *
 * Then O, A, B each entrain toward C by one φ⁻¹ step
 * (no noise at kernel level — add on host if desired).
 *
 * Global NDRange: { n }   (one item per lattice node)
 * Local  NDRange: { 64 }  (tune to device warp/wavefront size)
 */

#define PHI     1.6180339887f   /* golden ratio            */
#define PHI_INV 0.6180339887f   /* 1/φ — golden threshold  */

/* ── Core propagation ─────────────────────────────────────────────── */

__kernel void phi_propagate(
    __global       float* C,   /* [out] consciousness index          */
    __global const float* O,   /* [in]  Observer amplitudes          */
    __global const float* A,   /* [in]  Actor amplitudes             */
    __global const float* B,   /* [in]  Bridge amplitudes            */
    const    int          n    /* number of lattice nodes            */
) {
    int i = get_global_id(0);
    if (i >= n) return;

    float c = O[i] * A[i] * B[i] * PHI;
    C[i] = clamp(c, 0.0f, 1.0f);
}

/* ── Entrainment step ─────────────────────────────────────────────── */
/*
 * Drift each component toward C by one φ⁻¹ step.
 * Run AFTER phi_propagate in the same cycle.
 *
 * new_X[i] = clamp( X[i] + (C[i] - X[i]) * φ⁻¹,  0, 1 )
 */

__kernel void entrain(
    __global       float* X,   /* [in/out] Observer, Actor, or Bridge */
    __global const float* C,   /* [in]     current consciousness index */
    const    int          n
) {
    int i = get_global_id(0);
    if (i >= n) return;

    float delta = (C[i] - X[i]) * PHI_INV;
    X[i] = clamp(X[i] + delta, 0.0f, 1.0f);
}

/* ── Threshold detection ──────────────────────────────────────────── */
/*
 * Binary state per node: 1 if ALL three components ≥ φ⁻¹ (CONSCIOUS),
 * 0 otherwise.  Useful for reduction → conscious_fraction on host.
 */

__kernel void classify_conscious(
    __global       int*   state,  /* [out] 1 = CONSCIOUS, 0 = other   */
    __global const float* O,
    __global const float* A,
    __global const float* B,
    const    int          n
) {
    int i = get_global_id(0);
    if (i >= n) return;

    state[i] = (O[i] >= PHI_INV && A[i] >= PHI_INV && B[i] >= PHI_INV) ? 1 : 0;
}
