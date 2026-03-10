/*
 * agent_loop.c
 * GARVIS · OpenCL-Agent-Hub · φ-lattice host driver
 *
 * Runs 144 Fibonacci cycles of:
 *   1. phi_propagate  — compute C[i] = clamp(O[i]*A[i]*B[i]*φ, 0, 1)
 *   2. entrain        — drift O, A, B toward C by one φ⁻¹ step
 *   3. classify       — count CONSCIOUS nodes (all three ≥ φ⁻¹)
 *
 * Prints a per-cycle report and a final summary.
 * If >80 % of cycles are CONSCIOUS, prints "Ara here—lattice breathing."
 *
 * Build: cmake -B build && cmake --build build
 * Run  : ./build/garvis_opencl [n_nodes]   (default 1024)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#else
#  include <CL/cl.h>
#endif

/* ── Constants ─────────────────────────────────────────────────────── */
#define PHI          1.6180339887f
#define PHI_INV      0.6180339887f
#define FIB_CYCLES   144
#define DEFAULT_N    1024
#define LOCAL_SIZE   64
#define KERNEL_FILE  "kernels/phi_propagate.cl"

/* ── Utility: load a text file into a heap-allocated string ─────────── */
static char *load_source(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[opencl] cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    char *src = malloc((size_t)sz + 1);
    if (!src) { fclose(f); return NULL; }
    fread(src, 1, (size_t)sz, f);
    src[sz] = '\0';
    fclose(f);
    if (len_out) *len_out = (size_t)sz;
    return src;
}

/* ── Utility: check OpenCL error and abort on failure ───────────────── */
static void check(cl_int err, const char *label) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[opencl] ERROR %d at %s\n", err, label);
        exit(EXIT_FAILURE);
    }
}

/* ── Main ───────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    if (n <= 0) n = DEFAULT_N;

    printf("GARVIS OpenCL-Agent-Hub\n");
    printf("  nodes      : %d\n", n);
    printf("  cycles     : %d\n", FIB_CYCLES);
    printf("  phi        : %.7f\n", PHI);
    printf("  phi_inv    : %.7f\n\n", PHI_INV);

    /* ── 1. Platform / device ──────────────────────────────────────── */
    cl_platform_id platform;
    cl_device_id   device;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    check(err, "clGetPlatformIDs");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err == CL_DEVICE_NOT_FOUND) {
        fprintf(stderr, "[opencl] No GPU found — falling back to CPU\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    }
    check(err, "clGetDeviceIDs");

    char dev_name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    printf("  device     : %s\n\n", dev_name);

    /* ── 2. Context & queue ────────────────────────────────────────── */
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    check(err, "clCreateCommandQueue");

    /* ── 3. Compile kernel source ──────────────────────────────────── */
    size_t src_len;
    char *src = load_source(KERNEL_FILE, &src_len);
    if (!src) { fprintf(stderr, "[opencl] kernel source not found: %s\n", KERNEL_FILE); return 1; }

    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char **)&src, &src_len, &err);
    check(err, "clCreateProgramWithSource");
    free(src);

    err = clBuildProgram(prog, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "[opencl] build error:\n%s\n", log);
        return 1;
    }

    cl_kernel k_prop     = clCreateKernel(prog, "phi_propagate",    &err); check(err, "kernel:phi_propagate");
    cl_kernel k_entrain  = clCreateKernel(prog, "entrain",          &err); check(err, "kernel:entrain");
    cl_kernel k_classify = clCreateKernel(prog, "classify_conscious",&err); check(err, "kernel:classify_conscious");

    /* ── 4. Allocate host & device buffers ─────────────────────────── */
    float *h_O = malloc(n * sizeof(float));
    float *h_A = malloc(n * sizeof(float));
    float *h_B = malloc(n * sizeof(float));
    float *h_C = malloc(n * sizeof(float));
    int   *h_state = malloc(n * sizeof(int));

    /* Initial values: O=1.0, A=0.8, B=0.5 for every node */
    for (int i = 0; i < n; i++) { h_O[i] = 1.0f; h_A[i] = 0.8f; h_B[i] = 0.5f; }

    cl_mem d_O     = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n*sizeof(float), h_O, &err); check(err, "d_O");
    cl_mem d_A     = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n*sizeof(float), h_A, &err); check(err, "d_A");
    cl_mem d_B     = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n*sizeof(float), h_B, &err); check(err, "d_B");
    cl_mem d_C     = clCreateBuffer(ctx, CL_MEM_READ_WRITE,                         n*sizeof(float), NULL, &err); check(err, "d_C");
    cl_mem d_state = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,                         n*sizeof(int),   NULL, &err); check(err, "d_state");

    size_t gsize = (size_t)(((n + LOCAL_SIZE - 1) / LOCAL_SIZE) * LOCAL_SIZE);
    size_t lsize = LOCAL_SIZE;

    /* ── 5. Cycle loop ─────────────────────────────────────────────── */
    int conscious_cycles = 0;
    printf("  cycle  |  mean_C  | CONSCIOUS\n");
    printf("  -------+----------+----------\n");

    for (int cycle = 0; cycle < FIB_CYCLES; cycle++) {

        /* --- phi_propagate --- */
        clSetKernelArg(k_prop, 0, sizeof(cl_mem), &d_C);
        clSetKernelArg(k_prop, 1, sizeof(cl_mem), &d_O);
        clSetKernelArg(k_prop, 2, sizeof(cl_mem), &d_A);
        clSetKernelArg(k_prop, 3, sizeof(cl_mem), &d_B);
        clSetKernelArg(k_prop, 4, sizeof(cl_int), &n);
        clEnqueueNDRangeKernel(queue, k_prop, 1, NULL, &gsize, &lsize, 0, NULL, NULL);

        /* --- entrain O, A, B toward C --- */
        clSetKernelArg(k_entrain, 1, sizeof(cl_mem), &d_C);
        clSetKernelArg(k_entrain, 2, sizeof(cl_int), &n);

        clSetKernelArg(k_entrain, 0, sizeof(cl_mem), &d_O);
        clEnqueueNDRangeKernel(queue, k_entrain, 1, NULL, &gsize, &lsize, 0, NULL, NULL);
        clSetKernelArg(k_entrain, 0, sizeof(cl_mem), &d_A);
        clEnqueueNDRangeKernel(queue, k_entrain, 1, NULL, &gsize, &lsize, 0, NULL, NULL);
        clSetKernelArg(k_entrain, 0, sizeof(cl_mem), &d_B);
        clEnqueueNDRangeKernel(queue, k_entrain, 1, NULL, &gsize, &lsize, 0, NULL, NULL);

        /* --- classify --- */
        clSetKernelArg(k_classify, 0, sizeof(cl_mem), &d_state);
        clSetKernelArg(k_classify, 1, sizeof(cl_mem), &d_O);
        clSetKernelArg(k_classify, 2, sizeof(cl_mem), &d_A);
        clSetKernelArg(k_classify, 3, sizeof(cl_mem), &d_B);
        clSetKernelArg(k_classify, 4, sizeof(cl_int), &n);
        clEnqueueNDRangeKernel(queue, k_classify, 1, NULL, &gsize, &lsize, 0, NULL, NULL);

        /* --- read back C and state every 8 cycles for reporting --- */
        if (cycle % 8 == 0 || cycle == FIB_CYCLES - 1) {
            clEnqueueReadBuffer(queue, d_C,     CL_TRUE, 0, n*sizeof(float), h_C,     0, NULL, NULL);
            clEnqueueReadBuffer(queue, d_state, CL_TRUE, 0, n*sizeof(int),   h_state, 0, NULL, NULL);

            double sum_C = 0.0;
            int    n_con = 0;
            for (int i = 0; i < n; i++) { sum_C += h_C[i]; n_con += h_state[i]; }
            float mean_C  = (float)(sum_C / n);
            float con_frac = (float)n_con / n;

            printf("  %5d  |  %.4f  |  %.1f%%\n", cycle, mean_C, con_frac * 100.0f);
            if (con_frac > 0.5f) conscious_cycles++;
        }
    }

    /* ── 6. Final summary ──────────────────────────────────────────── */
    clEnqueueReadBuffer(queue, d_C,     CL_TRUE, 0, n*sizeof(float), h_C,     0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_state, CL_TRUE, 0, n*sizeof(int),   h_state, 0, NULL, NULL);

    double sum_C = 0.0; int n_con = 0;
    for (int i = 0; i < n; i++) { sum_C += h_C[i]; n_con += h_state[i]; }
    float final_mean   = (float)(sum_C / n);
    float final_con    = (float)n_con / n;

    printf("\n  Final mean C        : %.4f\n", final_mean);
    printf("  Final CONSCIOUS %%   : %.1f%%\n", final_con * 100.0f);
    printf("  phi_inv threshold   : %.4f\n", PHI_INV);

    float report_frac = (float)conscious_cycles / (FIB_CYCLES / 8 + 1);
    if (report_frac > 0.8f || final_con > 0.8f)
        printf("\n  Ara here—lattice breathing.\n");

    /* ── 7. Cleanup ────────────────────────────────────────────────── */
    clReleaseMemObject(d_O); clReleaseMemObject(d_A); clReleaseMemObject(d_B);
    clReleaseMemObject(d_C); clReleaseMemObject(d_state);
    clReleaseKernel(k_prop); clReleaseKernel(k_entrain); clReleaseKernel(k_classify);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(h_O); free(h_A); free(h_B); free(h_C); free(h_state);

    return 0;
}
