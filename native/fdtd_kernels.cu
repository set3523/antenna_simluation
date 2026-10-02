// Device kernels for voxel Yee FDTD. Compiled at runtime by NVRTC.
// Grid constants must match voxel_fdtd.cpp / voxel_fdtd_cuda.cpp.

#define PIX 51
#define PAD 12
#define NX (PIX + 2 * PAD)
#define NY NX
#define NZ_BELOW 28
#define NZ_SUB 4
#define NZ_ABOVE 40
#define NZ (NZ_BELOW + NZ_SUB + NZ_ABOVE)
#define N (NX * NY * NZ)
#define K_GND NZ_BELOW
#define K_PATCH (NZ_BELOW + NZ_SUB)
#define K_MID (K_GND + (K_PATCH - K_GND) / 2)
#define K_TOP (NZ - 3)
#define DX 1.0e-3f
#define DY 1.0e-3f
#define DZ 0.375e-3f

__device__ inline int didx(int i, int j, int k) { return i + NX * (j + NY * k); }

extern "C" __global__ void k_update_H(float* Hx, float* Hy, float* Hz, const float* Ex, const float* Ey,
                           const float* Ez, float dt_mu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= NX - 1 || j >= NY - 1 || k >= NZ - 1) return;
    int p = didx(i, j, k);
    Hx[p] += dt_mu * ((Ey[didx(i, j, k + 1)] - Ey[p]) / DZ - (Ez[didx(i, j + 1, k)] - Ez[p]) / DY);
    Hy[p] += dt_mu * ((Ez[didx(i + 1, j, k)] - Ez[p]) / DX - (Ex[didx(i, j, k + 1)] - Ex[p]) / DZ);
    Hz[p] += dt_mu * ((Ex[didx(i, j + 1, k)] - Ex[p]) / DY - (Ey[didx(i + 1, j, k)] - Ey[p]) / DX);
}

extern "C" __global__ void k_update_E(float* Ex, float* Ey, float* Ez, const float* Hx, const float* Hy,
                           const float* Hz, const float* Caex, const float* Cbex, const float* Caey,
                           const float* Cbey, const float* Caez, const float* Cbez, int feed_i,
                           int feed_j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < 1 || j < 1 || k < 1 || i >= NX - 1 || j >= NY - 1 || k >= NZ - 1) return;
    int p = didx(i, j, k);
    float curlHx = (Hz[p] - Hz[didx(i, j - 1, k)]) / DY - (Hy[p] - Hy[didx(i, j, k - 1)]) / DZ;
    float curlHy = (Hx[p] - Hx[didx(i, j, k - 1)]) / DZ - (Hz[p] - Hz[didx(i - 1, j, k)]) / DX;
    float curlHz = (Hy[p] - Hy[didx(i - 1, j, k)]) / DX - (Hx[p] - Hx[didx(i, j - 1, k)]) / DY;
    Ex[p] = Caex[p] * Ex[p] + Cbex[p] * curlHx;
    Ey[p] = Caey[p] * Ey[p] + Cbey[p] * curlHy;
    int feed = (i == feed_i && j == feed_j && k >= K_GND && k < K_PATCH);
    if (!feed) Ez[p] = Caez[p] * Ez[p] + Cbez[p] * curlHz;
}

extern "C" __global__ void k_lumped(float* Ez, const float* Hx, const float* Hy, int feed_i, int feed_j,
                         float dt, float vs) {
    int s = threadIdx.x + blockIdx.x * blockDim.x;
    int nser = K_PATCH - K_GND;
    if (s >= nser) return;
    int k = K_GND + s;
    int p = didx(feed_i, feed_j, k);
    float eps = 4.4f * 8.854187817e-12f;
    float curlH = (Hy[p] - Hy[didx(feed_i - 1, feed_j, k)]) / DX -
                  (Hx[p] - Hx[didx(feed_i, feed_j - 1, k)]) / DY;
    float Re = 50.0f / (float)nser;
    float Ve = vs / (float)nser;
    float A = dt * DZ / (2.f * Re * eps * DX * DY);
    float ez_old = Ez[p];
    Ez[p] = ((1.f - A) / (1.f + A)) * ez_old + (dt / eps) / (1.f + A) * curlH -
            (dt / (Re * eps * DX * DY)) / (1.f + A) * Ve;
}

extern "C" __global__ void k_mur_x(float* Ey, float* Ez, const float* Eyo, const float* Ezo, float mur_x) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < 1 || k < 1 || j >= NY - 1 || k >= NZ - 1) return;
    Ey[didx(0, j, k)] = Eyo[didx(1, j, k)] + mur_x * (Ey[didx(1, j, k)] - Eyo[didx(0, j, k)]);
    Ez[didx(0, j, k)] = Ezo[didx(1, j, k)] + mur_x * (Ez[didx(1, j, k)] - Ezo[didx(0, j, k)]);
    Ey[didx(NX - 1, j, k)] =
        Eyo[didx(NX - 2, j, k)] + mur_x * (Ey[didx(NX - 2, j, k)] - Eyo[didx(NX - 1, j, k)]);
    Ez[didx(NX - 1, j, k)] =
        Ezo[didx(NX - 2, j, k)] + mur_x * (Ez[didx(NX - 2, j, k)] - Ezo[didx(NX - 1, j, k)]);
}

extern "C" __global__ void k_mur_y(float* Ex, float* Ez, const float* Exo, const float* Ezo, float mur_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1 || k < 1 || i >= NX - 1 || k >= NZ - 1) return;
    Ex[didx(i, 0, k)] = Exo[didx(i, 1, k)] + mur_y * (Ex[didx(i, 1, k)] - Exo[didx(i, 0, k)]);
    Ez[didx(i, 0, k)] = Ezo[didx(i, 1, k)] + mur_y * (Ez[didx(i, 1, k)] - Ezo[didx(i, 0, k)]);
    Ex[didx(i, NY - 1, k)] =
        Exo[didx(i, NY - 2, k)] + mur_y * (Ex[didx(i, NY - 2, k)] - Exo[didx(i, NY - 1, k)]);
    Ez[didx(i, NY - 1, k)] =
        Ezo[didx(i, NY - 2, k)] + mur_y * (Ez[didx(i, NY - 2, k)] - Ezo[didx(i, NY - 1, k)]);
}

extern "C" __global__ void k_mur_z(float* Ex, float* Ey, const float* Exo, const float* Eyo, float mur_z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1 || j < 1 || i >= NX - 1 || j >= NY - 1) return;
    Ex[didx(i, j, 0)] = Exo[didx(i, j, 1)] + mur_z * (Ex[didx(i, j, 1)] - Exo[didx(i, j, 0)]);
    Ey[didx(i, j, 0)] = Eyo[didx(i, j, 1)] + mur_z * (Ey[didx(i, j, 1)] - Eyo[didx(i, j, 0)]);
    Ex[didx(i, j, NZ - 1)] =
        Exo[didx(i, j, NZ - 2)] + mur_z * (Ex[didx(i, j, NZ - 2)] - Exo[didx(i, j, NZ - 1)]);
    Ey[didx(i, j, NZ - 1)] =
        Eyo[didx(i, j, NZ - 2)] + mur_z * (Ey[didx(i, j, NZ - 2)] - Eyo[didx(i, j, NZ - 1)]);
}

extern "C" __global__ void k_pec(float* Ex, float* Ey, float* Ez, const unsigned char* pecEx,
                      const unsigned char* pecEy, const unsigned char* pecEz) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N) return;
    if (pecEx[p]) Ex[p] = 0.f;
    if (pecEy[p]) Ey[p] = 0.f;
    if (pecEz[p]) Ez[p] = 0.f;
}

extern "C" __global__ void k_sample(const float* Ez, const float* Hx, const float* Hy, float* Vrec, float* Irec,
                         float* Vsrec, int feed_i, int feed_j, int n, float vs) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float v = 0.f;
    for (int k = K_GND; k < K_PATCH; ++k) v += Ez[didx(feed_i, feed_j, k)] * DZ;
    Vrec[n] = v;
    int k = (K_GND + K_PATCH) / 2;
    int i = feed_i, j = feed_j;
    Irec[n] = (Hy[didx(i, j, k)] - Hy[didx(i - 1, j, k)]) * DX -
              (Hx[didx(i, j, k)] - Hx[didx(i, j - 1, k)]) * DY;
    Vsrec[n] = vs;
}

extern "C" __global__ void k_accum_air(const float* Ex, const float* Ey, const float* Ez, float* air) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= NX || j >= NY) return;
    int k = K_PATCH + 4;
    if (k >= NZ) k = NZ - 2;
    int p = didx(i, j, k);
    air[i + NX * j] += Ex[p] * Ex[p] + Ey[p] * Ey[p] + Ez[p] * Ez[p];
}

#define CONE_COS 0.866025403784f
#define AX0 (PAD + PIX / 2)
#define AY0 (PAD + PIX / 2)
#define AZ0 K_PATCH

__device__ inline float e2_at(const float* Ex, const float* Ey, const float* Ez, int i, int j, int k) {
    int p = didx(i, j, k);
    return Ex[p] * Ex[p] + Ey[p] * Ey[p] + Ez[p] * Ez[p];
}

__device__ inline int in_cone(int i, int j, int k, float nx, float ny, float nz) {
    float vx = (float(i) - float(AX0)) * DX;
    float vy = (float(j) - float(AY0)) * DY;
    float vz = (float(k) - float(AZ0)) * DZ;
    float r = sqrtf(vx * vx + vy * vy + vz * vz) + 1e-30f;
    return ((vx * nx + vy * ny + vz * nz) / r >= CONE_COS) ? 1 : 0;
}

__device__ inline void add_face(float* face, int fi, const float* Ex, const float* Ey, const float* Ez, int i,
                               int j, int k, float nx, float ny, float nz, float da) {
    float w = e2_at(Ex, Ey, Ez, i, j, k) * da;
    atomicAdd(face + fi, w);
    if (in_cone(i, j, k, nx, ny, nz)) atomicAdd(face + 6 + fi, w);
}

extern "C" __global__ void k_accum_faces(const float* Ex, const float* Ey, const float* Ez, float* face) {
    /* face[0:6] face power |E|^2 dA, face[6:12] cone 60° (full angle). +x -x +y -y +z -z */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;
    float dax = DY * DZ, day = DX * DZ, daz = DX * DY;
    int nxside = (NY - 4) * (NZ - 4);
    for (int n = tid; n < nxside; n += nthreads) {
        int j = 2 + n % (NY - 4);
        int k = 2 + n / (NY - 4);
        add_face(face, 0, Ex, Ey, Ez, NX - 3, j, k, 1.f, 0.f, 0.f, dax);
        add_face(face, 1, Ex, Ey, Ez, 2, j, k, -1.f, 0.f, 0.f, dax);
    }
    int nyside = (NX - 4) * (NZ - 4);
    for (int n = tid; n < nyside; n += nthreads) {
        int i = 2 + n % (NX - 4);
        int k = 2 + n / (NX - 4);
        add_face(face, 2, Ex, Ey, Ez, i, NY - 3, k, 0.f, 1.f, 0.f, day);
        add_face(face, 3, Ex, Ey, Ez, i, 2, k, 0.f, -1.f, 0.f, day);
    }
    int nzside = (NX - 4) * (NY - 4);
    for (int n = tid; n < nzside; n += nthreads) {
        int i = 2 + n % (NX - 4);
        int j = 2 + n / (NX - 4);
        add_face(face, 4, Ex, Ey, Ez, i, j, NZ - 3, 0.f, 0.f, 1.f, daz);
        add_face(face, 5, Ex, Ey, Ez, i, j, 2, 0.f, 0.f, -1.f, daz);
    }
}

extern "C" __global__ void k_accum_slice(const float* Ex, const float* Ey, const float* Ez, float* buf,
                                        int k_layer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= NX || j >= NY) return;
    if (k_layer < 0 || k_layer >= NZ) return;
    int p = didx(i, j, k_layer);
    buf[i + NX * j] += Ex[p] * Ex[p] + Ey[p] * Ey[p] + Ez[p] * Ez[p];
}

extern "C" __global__ void k_accum_cut_max(const float* Ex, const float* Ey, const float* Ez,
                                           float* cut) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= NX || k >= NZ) return;
    float m = 0.f;
    for (int j = 0; j < NY; ++j) {
        float e2 = e2_at(Ex, Ey, Ez, i, j, k);
        m = fmaxf(m, e2);
    }
    cut[i + NX * k] += m;
}

extern "C" __global__ void k_accum_cut(const float* Ex, const float* Ey, const float* Ez, float* cut,
                                      int feed_j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= NX || k >= NZ) return;
    int j = feed_j;
    if (j < 0) j = 0;
    if (j >= NY) j = NY - 1;
    int p = didx(i, j, k);
    cut[i + NX * k] += Ex[p] * Ex[p] + Ey[p] * Ey[p] + Ez[p] * Ez[p];
}

extern "C" __global__ void k_zero_f32(float* buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = 0.f;
}

extern "C" __global__ void k_energy(const float* Ex, const float* Ey, const float* Ez, const float* Hx,
                                   const float* Hy, const float* Hz, float* out) {
    __shared__ float sh[256];
    float s = 0.f;
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < N; p += blockDim.x * gridDim.x) {
        s += Ex[p] * Ex[p] + Ey[p] * Ey[p] + Ez[p] * Ez[p] + Hx[p] * Hx[p] + Hy[p] * Hy[p] +
             Hz[p] * Hz[p];
    }
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sh[0]);
}


// ---- NTFF Huygens box (same layout as voxel_fdtd.cpp) --------------------------------
#define NT_M 4
#define NT_IA NT_M
#define NT_IB (NX - 1 - NT_M)
#define NT_JA NT_M
#define NT_JB (NY - 1 - NT_M)
#define NT_KA NT_M
#define NT_KB (NZ - 1 - NT_M)
#define NT_LX (NT_IB - NT_IA)
#define NT_LY (NT_JB - NT_JA)
#define NT_LZ (NT_KB - NT_KA)
#define NT_CX (NT_LY * NT_LZ)
#define NT_CY (NT_LX * NT_LZ)
#define NT_CZ (NT_LX * NT_LY)
#define NT_CELLS (2 * (NT_CX + NT_CY + NT_CZ))
#define NT_FMAX 16

__device__ inline int nt_off(int f) {
    return (f == 0) ? 0 : (f == 1) ? NT_CX : (f == 2) ? 2 * NT_CX : (f == 3) ? 2 * NT_CX + NT_CY
         : (f == 4) ? 2 * NT_CX + 2 * NT_CY : (f == 5) ? 2 * NT_CX + 2 * NT_CY + NT_CZ : NT_CELLS;
}

extern "C" __global__ void k_ntff_accum(const float* Ex, const float* Ey, const float* Ez, const float* Hx,
                                        const float* Hy, const float* Hz, float* buf, const float* tw,
                                        int nf) {
    /* tw[0:F]=cos(ω tE)Δt  tw[F:2F]=sin(ω tE)Δt  tw[2F:3F]=cos(ω tH)Δt  tw[3F:4F]=sin(ω tH)Δt, F=NT_FMAX */
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= NT_CELLS) return;
    int f6 = 5;
    for (int q = 0; q < 5; ++q)
        if (c < nt_off(q + 1)) { f6 = q; break; }
    int l = c - nt_off(f6);
    float ea, eb, ha, hb;
    if (f6 < 2) {
        int I = (f6 == 0) ? NT_IB : NT_IA;
        int j = NT_JA + l % NT_LY, k = NT_KA + l / NT_LY;
        ea = 0.5f * (Ey[didx(I, j, k)] + Ey[didx(I, j, k + 1)]);
        eb = 0.5f * (Ez[didx(I, j, k)] + Ez[didx(I, j + 1, k)]);
        ha = 0.25f * (Hy[didx(I - 1, j, k)] + Hy[didx(I, j, k)] + Hy[didx(I - 1, j + 1, k)] + Hy[didx(I, j + 1, k)]);
        hb = 0.25f * (Hz[didx(I - 1, j, k)] + Hz[didx(I, j, k)] + Hz[didx(I - 1, j, k + 1)] + Hz[didx(I, j, k + 1)]);
    } else if (f6 < 4) {
        int J = (f6 == 2) ? NT_JB : NT_JA;
        int i = NT_IA + l % NT_LX, k = NT_KA + l / NT_LX;
        ea = 0.5f * (Ex[didx(i, J, k)] + Ex[didx(i, J, k + 1)]);
        eb = 0.5f * (Ez[didx(i, J, k)] + Ez[didx(i + 1, J, k)]);
        ha = 0.25f * (Hx[didx(i, J - 1, k)] + Hx[didx(i, J, k)] + Hx[didx(i + 1, J - 1, k)] + Hx[didx(i + 1, J, k)]);
        hb = 0.25f * (Hz[didx(i, J - 1, k)] + Hz[didx(i, J, k)] + Hz[didx(i, J - 1, k + 1)] + Hz[didx(i, J, k + 1)]);
    } else {
        int K = (f6 == 4) ? NT_KB : NT_KA;
        int i = NT_IA + l % NT_LX, j = NT_JA + l / NT_LX;
        ea = 0.5f * (Ex[didx(i, j, K)] + Ex[didx(i, j + 1, K)]);
        eb = 0.5f * (Ey[didx(i, j, K)] + Ey[didx(i + 1, j, K)]);
        ha = 0.25f * (Hx[didx(i, j, K - 1)] + Hx[didx(i, j, K)] + Hx[didx(i + 1, j, K - 1)] + Hx[didx(i + 1, j, K)]);
        hb = 0.25f * (Hy[didx(i, j, K - 1)] + Hy[didx(i, j, K)] + Hy[didx(i, j + 1, K - 1)] + Hy[didx(i, j + 1, K)]);
    }
    float* b = buf + (size_t)c * nf * 8;
    for (int f = 0; f < nf; ++f) {
        float* q = b + f * 8;
        float ce = tw[f], se = tw[NT_FMAX + f], ch = tw[2 * NT_FMAX + f], sh = tw[3 * NT_FMAX + f];
        q[0] += ea * ce; q[1] -= ea * se;
        q[2] += eb * ce; q[3] -= eb * se;
        q[4] += ha * ch; q[5] -= ha * sh;
        q[6] += hb * ch; q[7] -= hb * sh;
    }
}
