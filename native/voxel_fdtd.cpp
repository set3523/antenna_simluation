// Voxel FDTD: fixed Yee mesh, patch metal is a 51x51 bitmap on one z-plane.
// Ground is a PEC sheet. Probe feed is a z-directed 50 ohm lumped port.
// OpenMP CPU. GPU 경로는 voxel_fdtd_cuda.cpp (NVRTC + Driver API).
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
constexpr int NZ_BELOW = 12;
constexpr int NZ_SUB = 4;
constexpr int NZ_ABOVE = 20;
constexpr int NZ = NZ_BELOW + NZ_SUB + NZ_ABOVE;
constexpr int N = NX * NY * NZ;

constexpr int K_GND = NZ_BELOW;
constexpr int K_PATCH = NZ_BELOW + NZ_SUB;
constexpr int K_AIR = K_PATCH + 4;  // ~1.5 mm above patch, in air

constexpr float DX = 1.0e-3f;
constexpr float DY = 1.0e-3f;
constexpr float DZ = 0.375e-3f;  // 1.5 mm / 4
constexpr float EPS_R = 4.4f;
constexpr float TAN_D = 0.02f;
constexpr float F0 = 7.25e9f;
constexpr float FC = 1.25e9f;
constexpr float RPORT = 50.0f;
constexpr int NSTEPS = 12000;
constexpr float END_CRIT = 1e-4f;

inline int idx(int i, int j, int k) { return i + NX * (j + NY * k); }

struct Solver {
    std::vector<float> Ex, Ey, Ez, Hx, Hy, Hz;
    std::vector<float> Caex, Cbex, Caey, Cbey, Caez, Cbez;
    std::vector<uint8_t> pecEx, pecEy, pecEz;
    std::vector<uint8_t> metal;  // PIX*PIX
    std::vector<float> Vrec, Irec;
    std::vector<float> airE;
    int air_n = 0;
    int feed_i = PAD + PIX / 2;
    int feed_j = PAD + PIX / 2;
    float dt = 0.f;
    float mur_x = 0.f, mur_y = 0.f, mur_z = 0.f;

    Solver()
        : Ex(N, 0), Ey(N, 0), Ez(N, 0), Hx(N, 0), Hy(N, 0), Hz(N, 0),
          Caex(N, 0), Cbex(N, 0), Caey(N, 0), Cbey(N, 0), Caez(N, 0), Cbez(N, 0),
          pecEx(N, 0), pecEy(N, 0), pecEz(N, 0),
          metal(PIX * PIX, 0) {
        const float inv = std::sqrt(1.f / (DX * DX) + 1.f / (DY * DY) + 1.f / (DZ * DZ));
        dt = 0.99f / (float(C0) * inv);
        mur_x = (float(C0) * dt - DX) / (float(C0) * dt + DX);
        mur_y = (float(C0) * dt - DY) / (float(C0) * dt + DY);
        mur_z = (float(C0) * dt - DZ) / (float(C0) * dt + DZ);
        build_materials();
        set_ground_pec();
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
            return TAN_D * 2.f * float(M_PI) * F0 * EPS_R * float(EPS0);
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

    void set_ground_pec() {
        std::fill(pecEx.begin(), pecEx.end(), 0);
        std::fill(pecEy.begin(), pecEy.end(), 0);
        std::fill(pecEz.begin(), pecEz.end(), 0);
        const int i0 = PAD, i1 = PAD + PIX;
        const int j0 = PAD, j1 = PAD + PIX;
        for (int j = j0; j <= j1; ++j)
            for (int i = i0; i < i1; ++i)
                pecEx[idx(i, j, K_GND)] = 1;
        for (int j = j0; j < j1; ++j)
            for (int i = i0; i <= i1; ++i)
                pecEy[idx(i, j, K_GND)] = 1;
    }

    void apply_metal_bitmap() {
        set_ground_pec();
        const int i0 = PAD, j0 = PAD;
        for (int q = 0; q < PIX; ++q) {
            for (int p = 0; p < PIX; ++p) {
                if (!metal[p + PIX * q])
                    continue;
                const int i = i0 + p;
                const int j = j0 + q;
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
        const float t0 = 3.0f / FC;
        const float tau = 0.55f / FC;
        const float x = (t - t0) / tau;
        return std::exp(-x * x) * std::sin(2.f * float(M_PI) * F0 * (t - t0));
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

    void accum_air() {
        const int k = (K_AIR < NZ) ? K_AIR : (NZ - 2);
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                const int p = idx(i, j, k);
                airE[i + NX * j] += Ex[p] * Ex[p] + Ey[p] * Ey[p] + Ez[p] * Ez[p];
            }
        ++air_n;
    }

    void copy_air(float* out) const {
        if (airE.size() != size_t(NX * NY)) {
            std::fill(out, out + NX * NY, 0.f);
            return;
        }
        const float inv = (air_n > 0) ? 1.f / float(air_n) : 1.f;
        for (int n = 0; n < NX * NY; ++n) out[n] = std::sqrt(airE[n] * inv);
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
        air_n = 0;
        std::vector<float> Vsrec(NSTEPS, 0);
        std::vector<float> Exo = Ex, Eyo = Ey, Ezo = Ez;

        float emax = 1e-30f;
        int nused = NSTEPS;
        float vmax = 0.f, imax = 0.f;
        double corr = 0.0;
        for (int n = 0; n < NSTEPS; ++n) {
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
            if ((n & 7) == 0 && n > 200) accum_air();
            vmax = std::max(vmax, std::abs(Vrec[n]));
            imax = std::max(imax, std::abs(Irec[n]));
            corr += double(Vrec[n]) * double(Vsrec[n]);
            if ((n & 31) == 0) {
                const float e = energy();
                if (e > emax)
                    emax = e;
                if (n > 4000 && e < END_CRIT * emax) {
                    nused = n + 1;
                    break;
                }
            }
        }

        std::fprintf(stderr, "[voxel] nused=%d vmax=%.4g imax=%.4g emax=%.4g dt=%.4g corr=%.4g\n",
                     nused, vmax, imax, emax, dt, corr);

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
            const double magV = std::hypot(vr, vi);
            const double magVs = std::hypot(srcr, srci) + 1e-30;
            // Thevenin 50 ohm: matched ⇒ |V|≈|Vs|/2. Phase-calibrated S11 is later work.
            const double mag = std::abs(2.0 * magV / magVs - 1.0);
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
