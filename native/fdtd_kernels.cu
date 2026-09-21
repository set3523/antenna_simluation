// Device kernels for voxel Yee FDTD. Compiled at runtime by NVRTC.
// Grid constants must match voxel_fdtd.cpp / voxel_fdtd_cuda.cpp.

#define PIX 51
#define PAD 12
#define NX (PIX + 2 * PAD)
#define NY NX
#define NZ_BELOW 12
#define NZ_SUB 4
#define NZ_ABOVE 20
#define NZ (NZ_BELOW + NZ_SUB + NZ_ABOVE)
#define N (NX * NY * NZ)
#define K_GND NZ_BELOW
#define K_PATCH (NZ_BELOW + NZ_SUB)
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
