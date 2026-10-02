// Voxel FDTD: 2-layer metal — fixed rectangle on K_GND (lower), 51x51 bitmap on K_PATCH (upper).
// Probe feed is a z-directed 50 ohm lumped port through the substrate.
// OpenMP CPU. GPU path is voxel_fdtd_cuda.cpp (NVRTC + Driver API).
#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif

namespace {

constexpr double C0 = 299792458.0;
constexpr double MU0 = 4e-7 * M_PI;
constexpr double EPS0 = 1.0 / (MU0 * C0 * C0);
constexpr double Z0 = 376.730313461;

constexpr int PIX = 51;
constexpr int PAD = 12;
constexpr int NX = PIX + 2 * PAD;
constexpr int NY = NX;
constexpr int NZ_BELOW = 28;  // v4: larger air region for far field (NTFF) (4.5->10.5 mm)
constexpr int NZ_SUB = 4;
constexpr int NZ_ABOVE = 40;  // v4: 7.5→15 mm
constexpr int NZ = NZ_BELOW + NZ_SUB + NZ_ABOVE;
constexpr int N = NX * NY * NZ;

constexpr int K_GND = NZ_BELOW;
constexpr int K_PATCH = NZ_BELOW + NZ_SUB;
constexpr float PATCH_LX_MM = 9.0f;
constexpr float PATCH_LY_MM = 11.7f;

void fill_rect_mask(std::vector<uint8_t>& g, float lx_mm, float ly_mm) {
    std::fill(g.begin(), g.end(), 0);
    const int cx = PIX / 2;
    const int cy = PIX / 2;
    const int nx = std::max(1, (int)std::lround(lx_mm));
    const int ny = std::max(1, (int)std::lround(ly_mm));
    const int x0 = cx - nx / 2;
    const int y0 = cy - ny / 2;
    for (int q = 0; q < ny; ++q) {
        for (int p = 0; p < nx; ++p) {
            const int i = x0 + p;
            const int j = y0 + q;
            if (i >= 0 && i < PIX && j >= 0 && j < PIX)
                g[i + PIX * j] = 1;
        }
    }
}
constexpr int K_AIR = K_PATCH + 4;  // ~1.5 mm above patch, in air
constexpr int K_MID = K_GND + (K_PATCH - K_GND) / 2;  // FR4 mid ~5.25 mm
constexpr int K_TOP = NZ - 3;                         // +z box inner face

constexpr float DX = 1.0e-3f;
constexpr float DY = 1.0e-3f;
constexpr float DZ = 0.375e-3f;  // 1.5 mm / 4
constexpr float EPS_R = 4.4f;
#ifndef VOXEL_TAN_D
#define VOXEL_TAN_D 0.02f
#endif
constexpr float TAN_D = VOXEL_TAN_D;
constexpr float F0_DEFAULT = 6.0e9f;
constexpr float FC_DEFAULT = 3.0e9f;
constexpr float RPORT = 50.0f;
constexpr int NSTEPS = 12000;
constexpr float END_CRIT = 1e-4f;

inline int idx(int i, int j, int k) { return i + NX * (j + NY * k); }

// like a camera FOV: cone 60° full angle -> 30° from the axis
constexpr float CONE_COS = 0.866025403784f;
constexpr float AX = float(PAD + PIX / 2);
constexpr float AY = float(PAD + PIX / 2);
constexpr float AZ = float(K_PATCH);

// ---- NTFF (near-to-far-field transform) Huygens box -------------------------------------------
// Box faces = E grid planes. NT_M cells inside the boundary, fully enclosing the substrate (FR4) and copper, in air.
// Face order 0:+x 1:-x 2:+y 3:-y 4:+z 5:-z. DFT of 2 tangential E + 2 tangential H at cell centers.
//   x face (a,b)=(y,z), y face (a,b)=(x,z), z face (a,b)=(x,y)
constexpr int NT_M = 4;
constexpr int NT_IA = NT_M, NT_IB = NX - 1 - NT_M;
constexpr int NT_JA = NT_M, NT_JB = NY - 1 - NT_M;
constexpr int NT_KA = NT_M, NT_KB = NZ - 1 - NT_M;
constexpr int NT_LX = NT_IB - NT_IA, NT_LY = NT_JB - NT_JA, NT_LZ = NT_KB - NT_KA;
constexpr int NT_CX = NT_LY * NT_LZ, NT_CY = NT_LX * NT_LZ, NT_CZ = NT_LX * NT_LY;
constexpr int NT_CELLS = 2 * (NT_CX + NT_CY + NT_CZ);
constexpr int NT_FMAX = 16;
constexpr int NT_EVERY = 2;
static_assert(NT_IA < PAD && NT_KA < NZ_BELOW && NT_KB > NZ_BELOW + NZ_SUB, "NTFF box must enclose substrate");
inline int nt_face_off(int f) {
    const int off[6] = {0, NT_CX, 2 * NT_CX, 2 * NT_CX + NT_CY, 2 * NT_CX + 2 * NT_CY,
                        2 * NT_CX + 2 * NT_CY + NT_CZ};
    return off[f];
}

inline bool in_cone(int i, int j, int k, float nx, float ny, float nz) {
    const float vx = (float(i) - AX) * DX;
    const float vy = (float(j) - AY) * DY;
    const float vz = (float(k) - AZ) * DZ;
    const float r = std::sqrt(vx * vx + vy * vy + vz * vz) + 1e-30f;
    return (vx * nx + vy * ny + vz * nz) / r >= CONE_COS;
}

inline double e2_at(const std::vector<float>& Ex, const std::vector<float>& Ey, const std::vector<float>& Ez,
                   int i, int j, int k) {
    const int p = idx(i, j, k);
    return double(Ex[p]) * Ex[p] + double(Ey[p]) * Ey[p] + double(Ez[p]) * Ez[p];
}

struct Solver {
    std::vector<float> Ex, Ey, Ez, Hx, Hy, Hz;
    std::vector<float> Caex, Cbex, Caey, Cbey, Caez, Cbez;
    std::vector<uint8_t> pecEx, pecEy, pecEz;
    std::vector<uint8_t> metal;     // upper layer @ K_PATCH
    std::vector<uint8_t> metal_lo;  // lower fixed rect @ K_GND
    bool lower_full_gnd = false;
    std::vector<float> Vrec, Irec;
    std::vector<float> airE;
    std::vector<float> patchE;
    std::vector<float> midE;
    std::vector<float> topE;
    std::vector<float> cutE;
    std::vector<float> cutMaxE;
    double faceE[6]{};
    double coneE[6]{};
    int air_n = 0;
    int feed_i = PAD + PIX / 2;
    int feed_j = PAD + PIX / 2;
    float dt = 0.f;
    float mur_x = 0.f, mur_y = 0.f, mur_z = 0.f;
    float t_sec = 3.5e-9f;
    float cfl = 0.99f;
    float f0_hz = F0_DEFAULT;
    float fc_hz = FC_DEFAULT;
    float leave_ratio = 0.f;
    // NTFF
    int nt_nf = 7;
    double nt_freq[NT_FMAX] = {3e9, 4e9, 5e9, 6e9, 7e9, 8e9, 9e9};
    std::vector<float> nt_buf;  // [cell][f][comp 4][re,im]
    double nt_pinc[NT_FMAX]{};
    double nt_prad[NT_FMAX]{};

    Solver()
        : Ex(N, 0), Ey(N, 0), Ez(N, 0), Hx(N, 0), Hy(N, 0), Hz(N, 0),
          Caex(N, 0), Cbex(N, 0), Caey(N, 0), Cbey(N, 0), Caez(N, 0), Cbez(N, 0),
          pecEx(N, 0), pecEy(N, 0), pecEz(N, 0),
          metal(PIX * PIX, 0), metal_lo(PIX * PIX, 0) {
        fill_rect_mask(metal_lo, PATCH_LX_MM, PATCH_LY_MM);
        apply_cfl();
    }

    void apply_cfl() {
        cfl = std::clamp(cfl, 0.10f, 0.99f);
        t_sec = std::max(t_sec, 1e-12f);
        const float inv = std::sqrt(1.f / (DX * DX) + 1.f / (DY * DY) + 1.f / (DZ * DZ));
        dt = cfl / (float(C0) * inv);
        mur_x = (float(C0) * dt - DX) / (float(C0) * dt + DX);
        mur_y = (float(C0) * dt - DY) / (float(C0) * dt + DY);
        mur_z = (float(C0) * dt - DZ) / (float(C0) * dt + DZ);
        build_materials();
    }

    int nplan() const {
        int n = (int)std::ceil(double(t_sec) / double(dt));
        return std::clamp(n, 1, NSTEPS);
    }

    float eps_at(int i, int j, int k) const {
        const int i0 = PAD, i1 = PAD + PIX;
        if (k >= K_GND && k < K_PATCH && i >= i0 && i < i1 && j >= i0 && j < i1)
            return EPS_R * float(EPS0);
        return float(EPS0);
    }

    float sig_at(int i, int j, int k) const {
        const int i0 = PAD, i1 = PAD + PIX;
        if (k >= K_GND && k < K_PATCH && i >= i0 && i < i1 && j >= i0 && j < i1)
            return TAN_D * 2.f * float(M_PI) * f0_hz * EPS_R * float(EPS0);
        return 0.f;
    }

    static void cab(float eps, float sig, float dt, float& Ca, float& Cb) {
        const float denom = 1.f + sig * dt / (2.f * eps);
        Ca = (1.f - sig * dt / (2.f * eps)) / denom;
        Cb = (dt / eps) / denom;
    }

    void build_materials() {
        for (int k = 0; k < NZ; ++k)
            for (int j = 0; j < NY; ++j)
                for (int i = 0; i < NX; ++i) {
                    const int p = idx(i, j, k);
                    float e = eps_at(i, j, k);
                    float s = sig_at(i, j, k);
                    cab(e, s, dt, Caex[p], Cbex[p]);
                    cab(e, s, dt, Caey[p], Cbey[p]);
                    cab(e, s, dt, Caez[p], Cbez[p]);
                }
    }

    void apply_metal_bitmap() {
        std::fill(pecEx.begin(), pecEx.end(), 0);
        std::fill(pecEy.begin(), pecEy.end(), 0);
        std::fill(pecEz.begin(), pecEz.end(), 0);
        const int i0 = PAD, j0 = PAD;
        const int i1 = PAD + PIX, j1 = PAD + PIX;
        if (lower_full_gnd) {
            for (int j = j0; j <= j1; ++j)
                for (int i = i0; i < i1; ++i)
                    pecEx[idx(i, j, K_GND)] = 1;
            for (int j = j0; j < j1; ++j)
                for (int i = i0; i <= i1; ++i)
                    pecEy[idx(i, j, K_GND)] = 1;
        }
        for (int q = 0; q < PIX; ++q) {
            for (int p = 0; p < PIX; ++p) {
                const int i = i0 + p;
                const int j = j0 + q;
                if (!lower_full_gnd && metal_lo[p + PIX * q]) {
                    pecEx[idx(i, j, K_GND)] = 1;
                    pecEx[idx(i, j + 1, K_GND)] = 1;
                    pecEy[idx(i, j, K_GND)] = 1;
                    pecEy[idx(i + 1, j, K_GND)] = 1;
                }
                if (!metal[p + PIX * q])
                    continue;
                pecEx[idx(i, j, K_PATCH)] = 1;
                pecEx[idx(i, j + 1, K_PATCH)] = 1;
                pecEy[idx(i, j, K_PATCH)] = 1;
                pecEy[idx(i + 1, j, K_PATCH)] = 1;
            }
        }
    }

    void apply_pec() {
#pragma omp parallel for schedule(static)
        for (int p = 0; p < N; ++p) {
            if (pecEx[p]) Ex[p] = 0.f;
            if (pecEy[p]) Ey[p] = 0.f;
            if (pecEz[p]) Ez[p] = 0.f;
        }
    }

    float source(float t) const {
        const float fc = std::max(fc_hz, 1e6f);
        const float f0 = std::max(f0_hz, 1e6f);
        const float t0 = 3.0f / fc;
        const float tau = 0.55f / fc;
        const float x = (t - t0) / tau;
        return std::exp(-x * x) * std::sin(2.f * float(M_PI) * f0 * (t - t0));
    }

    void update_H() {
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < NZ - 1; ++k) {
            for (int j = 0; j < NY - 1; ++j) {
                for (int i = 0; i < NX - 1; ++i) {
                    const int p = idx(i, j, k);
                    Hx[p] += float(dt / MU0) * ((Ey[idx(i, j, k + 1)] - Ey[p]) / DZ -
                                                (Ez[idx(i, j + 1, k)] - Ez[p]) / DY);
                    Hy[p] += float(dt / MU0) * ((Ez[idx(i + 1, j, k)] - Ez[p]) / DX -
                                                (Ex[idx(i, j, k + 1)] - Ex[p]) / DZ);
                    Hz[p] += float(dt / MU0) * ((Ex[idx(i, j + 1, k)] - Ex[p]) / DY -
                                                (Ey[idx(i + 1, j, k)] - Ey[p]) / DX);
                }
            }
        }
    }

    void update_E(float vs) {
        const int nser = K_PATCH - K_GND;
        float ez_old[8];
        for (int s = 0; s < nser; ++s)
            ez_old[s] = Ez[idx(feed_i, feed_j, K_GND + s)];

#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 1; k < NZ - 1; ++k) {
            for (int j = 1; j < NY - 1; ++j) {
                for (int i = 1; i < NX - 1; ++i) {
                    const int p = idx(i, j, k);
                    const float curlHx = (Hz[p] - Hz[idx(i, j - 1, k)]) / DY -
                                         (Hy[p] - Hy[idx(i, j, k - 1)]) / DZ;
                    const float curlHy = (Hx[p] - Hx[idx(i, j, k - 1)]) / DZ -
                                         (Hz[p] - Hz[idx(i - 1, j, k)]) / DX;
                    const float curlHz = (Hy[p] - Hy[idx(i - 1, j, k)]) / DX -
                                         (Hx[p] - Hx[idx(i, j - 1, k)]) / DY;
                    Ex[p] = Caex[p] * Ex[p] + Cbex[p] * curlHx;
                    Ey[p] = Caey[p] * Ey[p] + Cbey[p] * curlHy;
                    Ez[p] = Caez[p] * Ez[p] + Cbez[p] * curlHz;
                }
            }
        }

        // Series 50 ohm + voltage source (Taflove lumped element). Do not reuse the
        // already-updated Ez — that would apply curl(H) twice.
        const float Re = RPORT / float(nser);
        const float Ve = vs / float(nser);
        for (int s = 0; s < nser; ++s) {
            const int k = K_GND + s;
            const int p = idx(feed_i, feed_j, k);
            const float eps = eps_at(feed_i, feed_j, k);
            const float curlH = (Hy[p] - Hy[idx(feed_i - 1, feed_j, k)]) / DX -
                                (Hx[p] - Hx[idx(feed_i, feed_j - 1, k)]) / DY;
            const float A = dt * DZ / (2.f * Re * eps * DX * DY);
            Ez[p] = ((1.f - A) / (1.f + A)) * ez_old[s] +
                    (dt / eps) / (1.f + A) * curlH -
                    (dt / (Re * eps * DX * DY)) / (1.f + A) * Ve;
        }
    }

    void mur_abc(const std::vector<float>& Exo, const std::vector<float>& Eyo,
                 const std::vector<float>& Ezo) {
        // 1st-order Mur on domain faces. Interior already updated.
        for (int k = 1; k < NZ - 1; ++k) {
            for (int j = 1; j < NY - 1; ++j) {
                // x=0, x=nx-1 : Ey, Ez
                Ey[idx(0, j, k)] = Eyo[idx(1, j, k)] + mur_x * (Ey[idx(1, j, k)] - Eyo[idx(0, j, k)]);
                Ez[idx(0, j, k)] = Ezo[idx(1, j, k)] + mur_x * (Ez[idx(1, j, k)] - Ezo[idx(0, j, k)]);
                Ey[idx(NX - 1, j, k)] =
                    Eyo[idx(NX - 2, j, k)] + mur_x * (Ey[idx(NX - 2, j, k)] - Eyo[idx(NX - 1, j, k)]);
                Ez[idx(NX - 1, j, k)] =
                    Ezo[idx(NX - 2, j, k)] + mur_x * (Ez[idx(NX - 2, j, k)] - Ezo[idx(NX - 1, j, k)]);
            }
        }
        for (int k = 1; k < NZ - 1; ++k) {
            for (int i = 1; i < NX - 1; ++i) {
                Ex[idx(i, 0, k)] = Exo[idx(i, 1, k)] + mur_y * (Ex[idx(i, 1, k)] - Exo[idx(i, 0, k)]);
                Ez[idx(i, 0, k)] = Ezo[idx(i, 1, k)] + mur_y * (Ez[idx(i, 1, k)] - Ezo[idx(i, 0, k)]);
                Ex[idx(i, NY - 1, k)] =
                    Exo[idx(i, NY - 2, k)] + mur_y * (Ex[idx(i, NY - 2, k)] - Exo[idx(i, NY - 1, k)]);
                Ez[idx(i, NY - 1, k)] =
                    Ezo[idx(i, NY - 2, k)] + mur_y * (Ez[idx(i, NY - 2, k)] - Ezo[idx(i, NY - 1, k)]);
            }
        }
        for (int j = 1; j < NY - 1; ++j) {
            for (int i = 1; i < NX - 1; ++i) {
                Ex[idx(i, j, 0)] = Exo[idx(i, j, 1)] + mur_z * (Ex[idx(i, j, 1)] - Exo[idx(i, j, 0)]);
                Ey[idx(i, j, 0)] = Eyo[idx(i, j, 1)] + mur_z * (Ey[idx(i, j, 1)] - Eyo[idx(i, j, 0)]);
                Ex[idx(i, j, NZ - 1)] =
                    Exo[idx(i, j, NZ - 2)] + mur_z * (Ex[idx(i, j, NZ - 2)] - Exo[idx(i, j, NZ - 1)]);
                Ey[idx(i, j, NZ - 1)] =
                    Eyo[idx(i, j, NZ - 2)] + mur_z * (Ey[idx(i, j, NZ - 2)] - Eyo[idx(i, j, NZ - 1)]);
            }
        }
    }

    float energy() const {
        double e = 0;
#pragma omp parallel for reduction(+ : e) schedule(static)
        for (int p = 0; p < N; ++p) {
            e += double(Ex[p]) * Ex[p] + double(Ey[p]) * Ey[p] + double(Ez[p]) * Ez[p] +
                 double(Hx[p]) * Hx[p] + double(Hy[p]) * Hy[p] + double(Hz[p]) * Hz[p];
        }
        return float(e);
    }

    float sample_V() const {
        float v = 0;
        for (int k = K_GND; k < K_PATCH; ++k)
            v += Ez[idx(feed_i, feed_j, k)] * DZ;
        return v;
    }

    float sample_I() const {
        const int k = (K_GND + K_PATCH) / 2;
        const int i = feed_i, j = feed_j;
        return (Hy[idx(i, j, k)] - Hy[idx(i - 1, j, k)]) * DX -
               (Hx[idx(i, j, k)] - Hx[idx(i, j - 1, k)]) * DY;
    }

    static void accum_layer(const std::vector<float>& Ex, const std::vector<float>& Ey,
                            const std::vector<float>& Ez, int k, std::vector<float>& buf) {
        if (k < 0 || k >= NZ) return;
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                const int p = idx(i, j, k);
                buf[i + NX * j] += Ex[p] * Ex[p] + Ey[p] * Ey[p] + Ez[p] * Ez[p];
            }
    }

    void accum_air() {
        const int kAir = (K_AIR < NZ) ? K_AIR : (NZ - 2);
        accum_layer(Ex, Ey, Ez, kAir, airE);
        accum_layer(Ex, Ey, Ez, K_PATCH, patchE);
        accum_layer(Ex, Ey, Ez, K_MID, midE);
        accum_layer(Ex, Ey, Ez, K_TOP, topE);
        const int j0 = feed_j;
        for (int kcut = 0; kcut < NZ; ++kcut)
            for (int i = 0; i < NX; ++i) {
                const int p = idx(i, j0, kcut);
                cutE[i + NX * kcut] += Ex[p] * Ex[p] + Ey[p] * Ey[p] + Ez[p] * Ez[p];
            }
        for (int kcut = 0; kcut < NZ; ++kcut)
            for (int i = 0; i < NX; ++i) {
                double m = 0.0;
                for (int j = 0; j < NY; ++j) {
                    const int p = idx(i, j, kcut);
                    m = std::max(m, e2_at(Ex, Ey, Ez, i, j, kcut));
                }
                cutMaxE[i + NX * kcut] += float(m);
            }
        const int ia = 2, ib = NX - 3, ja = 2, jb = NY - 3, ka = 2, kb = NZ - 3;
        const double dax = double(DY) * double(DZ);
        const double day = double(DX) * double(DZ);
        const double daz = double(DX) * double(DY);
        auto add = [&](int fi, int i, int j, int k, float nx, float ny, float nz, double da) {
            const double w = e2_at(Ex, Ey, Ez, i, j, k) * da;
            faceE[fi] += w;
            if (in_cone(i, j, k, nx, ny, nz)) coneE[fi] += w;
        };
        for (int j = ja; j <= jb; ++j)
            for (int k = ka; k <= kb; ++k) {
                add(0, ib, j, k, 1.f, 0.f, 0.f, dax);
                add(1, ia, j, k, -1.f, 0.f, 0.f, dax);
            }
        for (int i = ia; i <= ib; ++i)
            for (int k = ka; k <= kb; ++k) {
                add(2, i, jb, k, 0.f, 1.f, 0.f, day);
                add(3, i, ja, k, 0.f, -1.f, 0.f, day);
            }
        for (int i = ia; i <= ib; ++i)
            for (int j = ja; j <= jb; ++j) {
                add(4, i, j, kb, 0.f, 0.f, 1.f, daz);
                add(5, i, j, ka, 0.f, 0.f, -1.f, daz);
            }
        ++air_n;
    }

    // tangential E(a,b), H(a,b) at cell center (face f, local l). Averaged from Yee positions.
    void nt_sample(int f, int l, float& ea, float& eb, float& ha, float& hb) const {
        if (f < 2) {
            const int I = (f == 0) ? NT_IB : NT_IA;
            const int j = NT_JA + l % NT_LY, k = NT_KA + l / NT_LY;
            ea = 0.5f * (Ey[idx(I, j, k)] + Ey[idx(I, j, k + 1)]);
            eb = 0.5f * (Ez[idx(I, j, k)] + Ez[idx(I, j + 1, k)]);
            ha = 0.25f * (Hy[idx(I - 1, j, k)] + Hy[idx(I, j, k)] + Hy[idx(I - 1, j + 1, k)] + Hy[idx(I, j + 1, k)]);
            hb = 0.25f * (Hz[idx(I - 1, j, k)] + Hz[idx(I, j, k)] + Hz[idx(I - 1, j, k + 1)] + Hz[idx(I, j, k + 1)]);
        } else if (f < 4) {
            const int J = (f == 2) ? NT_JB : NT_JA;
            const int i = NT_IA + l % NT_LX, k = NT_KA + l / NT_LX;
            ea = 0.5f * (Ex[idx(i, J, k)] + Ex[idx(i, J, k + 1)]);
            eb = 0.5f * (Ez[idx(i, J, k)] + Ez[idx(i + 1, J, k)]);
            ha = 0.25f * (Hx[idx(i, J - 1, k)] + Hx[idx(i, J, k)] + Hx[idx(i + 1, J - 1, k)] + Hx[idx(i + 1, J, k)]);
            hb = 0.25f * (Hz[idx(i, J - 1, k)] + Hz[idx(i, J, k)] + Hz[idx(i, J - 1, k + 1)] + Hz[idx(i, J, k + 1)]);
        } else {
            const int K = (f == 4) ? NT_KB : NT_KA;
            const int i = NT_IA + l % NT_LX, j = NT_JA + l / NT_LX;
            ea = 0.5f * (Ex[idx(i, j, K)] + Ex[idx(i, j + 1, K)]);
            eb = 0.5f * (Ey[idx(i, j, K)] + Ey[idx(i + 1, j, K)]);
            ha = 0.25f * (Hx[idx(i, j, K - 1)] + Hx[idx(i, j, K)] + Hx[idx(i + 1, j, K - 1)] + Hx[idx(i + 1, j, K)]);
            hb = 0.25f * (Hy[idx(i, j, K - 1)] + Hy[idx(i, j, K)] + Hy[idx(i, j + 1, K - 1)] + Hy[idx(i, j + 1, K)]);
        }
    }

    // end of iteration n: E is at (n+1)dt, H at (n+½)dt.
    void nt_accum(int n) {
        const int nf = nt_nf;
        float cE[NT_FMAX], sE[NT_FMAX], cH[NT_FMAX], sH[NT_FMAX];
        const double w_dt = double(dt) * NT_EVERY;
        for (int f = 0; f < nf; ++f) {
            const double w = 2.0 * M_PI * nt_freq[f];
            const double tE = double(n + 1) * dt, tH = (double(n) + 0.5) * dt;
            cE[f] = float(std::cos(w * tE) * w_dt);
            sE[f] = float(std::sin(w * tE) * w_dt);
            cH[f] = float(std::cos(w * tH) * w_dt);
            sH[f] = float(std::sin(w * tH) * w_dt);
        }
#pragma omp parallel for schedule(static)
        for (int c = 0; c < NT_CELLS; ++c) {
            int f6 = 5;
            for (int q = 0; q < 5; ++q)
                if (c < nt_face_off(q + 1)) { f6 = q; break; }
            const int l = c - nt_face_off(f6);
            float ea, eb, ha, hb;
            nt_sample(f6, l, ea, eb, ha, hb);
            float* b = nt_buf.data() + size_t(c) * nf * 8;
            for (int f = 0; f < nf; ++f) {
                float* q = b + f * 8;
                q[0] += ea * cE[f]; q[1] -= ea * sE[f];
                q[2] += eb * cE[f]; q[3] -= eb * sE[f];
                q[4] += ha * cH[f]; q[5] -= ha * sH[f];
                q[6] += hb * cH[f]; q[7] -= hb * sH[f];
            }
        }
    }

    // P_rad(f) = ½ Re ∮ (E×H*)·n dA,   P_inc(f) = |Vs(f)|² / (8 R).  Same DFT convention -> ratio = total efficiency.
    void nt_finish(const std::vector<float>& Vsrec, int nused) {
        const int nf = nt_nf;
        for (int f = 0; f < nf; ++f) {
            const double w = 2.0 * M_PI * nt_freq[f];
            double vr = 0, vi = 0;
            for (int n = 0; n < nused; ++n) {
                const double t = (double(n) + 0.5) * dt;
                vr += Vsrec[n] * std::cos(w * t) * dt;
                vi -= Vsrec[n] * std::sin(w * t) * dt;
            }
            nt_pinc[f] = (vr * vr + vi * vi) / (8.0 * double(RPORT));
            double p = 0.0;
            for (int f6 = 0; f6 < 6; ++f6) {
                const double sgn = (f6 % 2 == 0) ? 1.0 : -1.0;
                const double da = (f6 < 2) ? double(DY) * DZ : (f6 < 4) ? double(DX) * DZ : double(DX) * DY;
                const int nc = (f6 < 2) ? NT_CX : (f6 < 4) ? NT_CY : NT_CZ;
                double acc = 0.0;
                for (int l = 0; l < nc; ++l) {
                    const float* q = nt_buf.data() + (size_t(nt_face_off(f6) + l) * nf + f) * 8;
                    // (E×H*)·n̂ : component along n̂=+a×b = Ea Hb* − Eb Ha*.  x face a=y,b=z -> x, y face a=x,b=z -> −y, z face a=x,b=y -> z
                    const double re = (double(q[0]) * q[6] + double(q[1]) * q[7]) - (double(q[2]) * q[4] + double(q[3]) * q[5]);
                    acc += re;
                }
                const double orient = (f6 == 2 || f6 == 3) ? -1.0 : 1.0;
                p += 0.5 * sgn * orient * acc * da;
            }
            nt_prad[f] = p;
        }
    }

    void copy_air(float* out) const {
        if (airE.size() != size_t(NX * NY)) {
            std::fill(out, out + NX * NY, 0.f);
            return;
        }
        const float inv = (air_n > 0) ? 1.f / float(air_n) : 1.f;
        for (int n = 0; n < NX * NY; ++n) out[n] = std::sqrt(airE[n] * inv);
    }

    void copy_faces(float* out) const {
        const double ax = double((NY - 4) * (NZ - 4)) * double(DY) * double(DZ);
        const double ay = double((NX - 4) * (NZ - 4)) * double(DX) * double(DZ);
        const double az = double((NX - 4) * (NY - 4)) * double(DX) * double(DY);
        const double invt = (air_n > 0) ? 1.0 / double(air_n) : 1.0;
        const double area[6] = {ax, ax, ay, ay, az, az};
        double pall = 0;
        for (int i = 0; i < 6; ++i) pall += faceE[i];
        pall += 1e-30;
        for (int i = 0; i < 6; ++i) out[i] = float(faceE[i] * invt / area[i]);
        for (int i = 0; i < 6; ++i) out[6 + i] = float(coneE[i] / pall);
    }

    void copy_leave(float* out) const {
        if (out) *out = leave_ratio;
    }

    void copy_cut(float* out) const {
        if (cutE.size() != size_t(NX * NZ)) {
            std::fill(out, out + NX * NZ, 0.f);
            return;
        }
        const float inv = (air_n > 0) ? 1.f / float(air_n) : 1.f;
        for (int n = 0; n < NX * NZ; ++n) out[n] = std::sqrt(cutE[n] * inv);
    }

    void copy_layer(const std::vector<float>& src, float* out) const {
        const float inv = (air_n > 0) ? 1.f / float(air_n) : 1.f;
        for (int n = 0; n < NX * NY; ++n) out[n] = std::sqrt(src[n] * inv);
    }

    void copy_cut_max(float* out) const {
        const float inv = (air_n > 0) ? 1.f / float(air_n) : 1.f;
        for (int n = 0; n < NX * NZ; ++n) out[n] = std::sqrt(cutMaxE[n] * inv);
    }

    int run(int nfreq, const double* freqs, double* s11_db, double* min_db, double* min_hz) {
        std::fill(Ex.begin(), Ex.end(), 0);
        std::fill(Ey.begin(), Ey.end(), 0);
        std::fill(Ez.begin(), Ez.end(), 0);
        std::fill(Hx.begin(), Hx.end(), 0);
        std::fill(Hy.begin(), Hy.end(), 0);
        std::fill(Hz.begin(), Hz.end(), 0);
        apply_metal_bitmap();

        Vrec.assign(NSTEPS, 0);
        Irec.assign(NSTEPS, 0);
        airE.assign(NX * NY, 0);
        patchE.assign(NX * NY, 0);
        midE.assign(NX * NY, 0);
        topE.assign(NX * NY, 0);
        cutE.assign(NX * NZ, 0);
        cutMaxE.assign(NX * NZ, 0);
        for (int i = 0; i < 6; ++i) {
            faceE[i] = 0;
            coneE[i] = 0;
        }
        air_n = 0;
        nt_buf.assign(size_t(NT_CELLS) * nt_nf * 8, 0.f);
        std::vector<float> Vsrec(NSTEPS, 0);
        std::vector<float> Exo = Ex, Eyo = Ey, Ezo = Ez;

        float emax = 1e-30f;
        int nused = nplan();
        float vmax = 0.f, imax = 0.f;
        double corr = 0.0;
        double einc = 0.0;
        for (int n = 0; n < nused; ++n) {
            const float t = (n + 0.5f) * dt;
            update_H();
            Exo = Ex;
            Eyo = Ey;
            Ezo = Ez;
            update_E(source(t));
            mur_abc(Exo, Eyo, Ezo);
            apply_pec();
            Vrec[n] = sample_V();
            Irec[n] = sample_I();
            Vsrec[n] = source(t);
            einc += double(Vsrec[n]) * Vsrec[n] * double(dt) / (8.0 * double(RPORT));
            if ((n & 7) == 0 && n > 200) accum_air();
            if ((n % NT_EVERY) == 0) nt_accum(n);
            vmax = std::max(vmax, std::abs(Vrec[n]));
            imax = std::max(imax, std::abs(Irec[n]));
            corr += double(Vrec[n]) * double(Vsrec[n]);
            if ((n & 31) == 0) {
                const float e = energy();
                if (e > emax)
                    emax = e;
            }
        }
        nt_finish(Vsrec, nused);
        {
            // leave = radiated/incident power (NTFF, frequency-averaged) = total efficiency (includes mismatch and FR4 loss)
            double acc = 0.0;
            for (int f = 0; f < nt_nf; ++f) acc += nt_prad[f] / (nt_pinc[f] + 1e-300);
            leave_ratio = float(acc / std::max(nt_nf, 1));
        }

        std::fprintf(stderr,
                     "[voxel] nused=%d t=%.3gns cfl=%.2f vmax=%.4g leave=%.4g dt=%.4g\n",
                     nused, double(t_sec) * 1e9, cfl, vmax, leave_ratio, dt);

        double best = 1e9, bestf = freqs[0];
        for (int f = 0; f < nfreq; ++f) {
            const double w = 2.0 * M_PI * freqs[f];
            double vr = 0, vi = 0, srcr = 0, srci = 0;
            for (int n = 0; n < nused; ++n) {
                const double ph = w * n * dt;
                const double c = std::cos(ph);
                const double s = std::sin(ph);
                vr += Vrec[n] * c;
                vi -= Vrec[n] * s;
                srcr += Vsrec[n] * c;
                srci -= Vsrec[n] * s;
            }
            // complex reflection coefficient. The feed source enters as -Ve, so the port voltage is -V.
            // Gamma = 2(-V)/Vs - 1  (checked against the V/I impedance result, |Gamma|<=1)
            const double den = srcr * srcr + srci * srci + 1e-300;
            const double qr = (vr * srcr + vi * srci) / den;
            const double qi = (vi * srcr - vr * srci) / den;
            const double mag = std::hypot(-2.0 * qr - 1.0, -2.0 * qi);
            const double db = 20.0 * std::log10(mag + 1e-15);
            s11_db[f] = db;
            if (db < best) {
                best = db;
                bestf = freqs[f];
            }
        }
        *min_db = best;
        *min_hz = bestf;
        return nused;
    }
};

}  // namespace

API void* voxel_create() { return new Solver(); }

API void voxel_destroy(void* ctx) { delete static_cast<Solver*>(ctx); }

API void voxel_set_feed(void* ctx, int px, int py) {
    auto* s = static_cast<Solver*>(ctx);
    px = std::clamp(px, 0, PIX - 1);
    py = std::clamp(py, 0, PIX - 1);
    s->feed_i = PAD + px;
    s->feed_j = PAD + py;
}

API void voxel_set_lower_full_gnd(void* ctx, int on) {
    auto* s = static_cast<Solver*>(ctx);
    s->lower_full_gnd = (on != 0);
}

API void voxel_set_sim(void* ctx, double t_sec, double cfl, double f0_hz, double fc_hz) {
    auto* s = static_cast<Solver*>(ctx);
    s->t_sec = float(t_sec);
    s->cfl = float(cfl);
    s->f0_hz = float(std::max(f0_hz, 1e6));
    s->fc_hz = float(std::max(fc_hz, 1e6));
    s->apply_cfl();
}

API void voxel_leave(void* ctx, float* out) {
    auto* s = static_cast<Solver*>(ctx);
    if (out) s->copy_leave(out);
}

API void voxel_set_metal(void* ctx, const unsigned char* mask, int n) {
    auto* s = static_cast<Solver*>(ctx);
    const int m = PIX * PIX;
    const int cpy = n < m ? n : m;
    std::memcpy(s->metal.data(), mask, cpy);
}

API int voxel_run(void* ctx, int nfreq, const double* freqs, double* s11_db, double* min_db,
                  double* min_hz) {
    return static_cast<Solver*>(ctx)->run(nfreq, freqs, s11_db, min_db, min_hz);
}

API void voxel_grid(int* nx, int* ny, int* nz, int* pix) {
    if (nx) *nx = NX;
    if (ny) *ny = NY;
    if (nz) *nz = NZ;
    if (pix) *pix = PIX;
}

API void voxel_air_e(void* ctx, float* out, int* nx, int* ny) {
    auto* s = static_cast<Solver*>(ctx);
    if (nx) *nx = NX;
    if (ny) *ny = NY;
    if (out) s->copy_air(out);
}

API void voxel_cut_e(void* ctx, float* out, int* nx, int* nz) {
    auto* s = static_cast<Solver*>(ctx);
    if (nx) *nx = NX;
    if (nz) *nz = NZ;
    if (out) s->copy_cut(out);
}

API void voxel_extra_fields(void* ctx, float* patch, float* mid, float* top, float* cut_max, int* nx,
                            int* ny, int* nz) {
    auto* s = static_cast<Solver*>(ctx);
    if (nx) *nx = NX;
    if (ny) *ny = NY;
    if (nz) *nz = NZ;
    if (patch) s->copy_layer(s->patchE, patch);
    if (mid) s->copy_layer(s->midE, mid);
    if (top) s->copy_layer(s->topE, top);
    if (cut_max) s->copy_cut_max(cut_max);
}

API void voxel_faces(void* ctx, float* out) {
    auto* s = static_cast<Solver*>(ctx);
    if (out) s->copy_faces(out);
}

API void voxel_ntff_set(void* ctx, int nf, const double* freqs) {
    auto* s = static_cast<Solver*>(ctx);
    nf = std::clamp(nf, 1, NT_FMAX);
    s->nt_nf = nf;
    for (int f = 0; f < nf; ++f) s->nt_freq[f] = freqs[f];
}

// info[0..5] = IA IB JA JB KA KB, info[6] = nf, info[7] = ncells, info[8..10] = LX LY LZ,
// info[11] = K_PATCH, info[12] = AX relative to feed (=PAD+PIX/2)
API void voxel_ntff_info(void* ctx, int* info, double* d3) {
    auto* s = static_cast<Solver*>(ctx);
    const int v[13] = {NT_IA, NT_IB, NT_JA, NT_JB, NT_KA, NT_KB, s->nt_nf, NT_CELLS, NT_LX, NT_LY, NT_LZ,
                       K_PATCH, PAD + PIX / 2};
    if (info) std::memcpy(info, v, sizeof(v));
    if (d3) { d3[0] = DX; d3[1] = DY; d3[2] = DZ; }
}

// buf: ncells*nf*8 float, freq/pinc/prad: nf double
API void voxel_ntff_get(void* ctx, float* buf, double* freq, double* pinc, double* prad) {
    auto* s = static_cast<Solver*>(ctx);
    if (buf && !s->nt_buf.empty()) std::memcpy(buf, s->nt_buf.data(), s->nt_buf.size() * sizeof(float));
    for (int f = 0; f < s->nt_nf; ++f) {
        if (freq) freq[f] = s->nt_freq[f];
        if (pinc) pinc[f] = s->nt_pinc[f];
        if (prad) prad[f] = s->nt_prad[f];
    }
}
