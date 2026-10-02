// Voxel FDTD CUDA backend. Same C API as voxel_fdtd.cpp.
// Host is MinGW C++; device code is NVRTC -> PTX, then the NVIDIA driver JITs it
// (CUDA 10.2 nvcc cannot emit sm_86, but compute_75 PTX runs on RTX 3060).
#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif

namespace {

constexpr double C0 = 299792458.0;
constexpr double MU0 = 4e-7 * M_PI;
constexpr double EPS0 = 1.0 / (MU0 * C0 * C0);

constexpr int PIX = 51;
constexpr int PAD = 12;
constexpr int NX = PIX + 2 * PAD;
constexpr int NY = NX;
constexpr int NZ_BELOW = 28;  // v4: larger air region for far field (NTFF) (same as fdtd_kernels.cu)
constexpr int NZ_SUB = 4;
constexpr int NZ_ABOVE = 40;
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
constexpr int K_MID = K_GND + (K_PATCH - K_GND) / 2;
constexpr int K_TOP = NZ - 3;

constexpr float DX = 1.0e-3f;
constexpr float DY = 1.0e-3f;
constexpr float DZ = 0.375e-3f;
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

// NTFF Huygens box, same as voxel_fdtd.cpp / fdtd_kernels.cu
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
    const int off[7] = {0, NT_CX, 2 * NT_CX, 2 * NT_CX + NT_CY, 2 * NT_CX + 2 * NT_CY,
                        2 * NT_CX + 2 * NT_CY + NT_CZ, NT_CELLS};
    return off[f];
}

#ifdef _WIN32
#define CUDAAPI __stdcall
#else
#define CUDAAPI
#endif

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef unsigned long long CUdeviceptr;

enum {
    CUDA_SUCCESS = 0,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
};

typedef CUresult(CUDAAPI* PFN_cuInit)(unsigned int);
typedef CUresult(CUDAAPI* PFN_cuDeviceGet)(CUdevice*, int);
typedef CUresult(CUDAAPI* PFN_cuDeviceGetCount)(int*);
typedef CUresult(CUDAAPI* PFN_cuDeviceGetAttribute)(int*, int, CUdevice);
typedef CUresult(CUDAAPI* PFN_cuCtxCreate)(CUcontext*, unsigned int, CUdevice);
typedef CUresult(CUDAAPI* PFN_cuCtxDestroy)(CUcontext);
typedef CUresult(CUDAAPI* PFN_cuCtxSetCurrent)(CUcontext);
typedef CUresult(CUDAAPI* PFN_cuModuleLoadData)(CUmodule*, const void*);
typedef CUresult(CUDAAPI* PFN_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
typedef CUresult(CUDAAPI* PFN_cuMemAlloc)(CUdeviceptr*, size_t);
typedef CUresult(CUDAAPI* PFN_cuMemFree)(CUdeviceptr);
typedef CUresult(CUDAAPI* PFN_cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
typedef CUresult(CUDAAPI* PFN_cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
typedef CUresult(CUDAAPI* PFN_cuMemcpyDtoD)(CUdeviceptr, CUdeviceptr, size_t);
typedef CUresult(CUDAAPI* PFN_cuMemsetD8)(CUdeviceptr, unsigned char, size_t);
typedef CUresult(CUDAAPI* PFN_cuMemsetD32)(CUdeviceptr, unsigned int, size_t);
typedef CUresult(CUDAAPI* PFN_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                                              unsigned int, unsigned int, unsigned int, unsigned int,
                                              void*, void**, void**);
typedef CUresult(CUDAAPI* PFN_cuCtxSynchronize)(void);
typedef CUresult(CUDAAPI* PFN_cuGetErrorName)(CUresult, const char**);

typedef int nvrtcResult;
typedef struct _nvrtcProgram* nvrtcProgram;
enum { NVRTC_SUCCESS = 0 };

typedef nvrtcResult (*PFN_nvrtcCreateProgram)(nvrtcProgram*, const char*, const char*, int,
                                              const char* const*, const char* const*);
typedef nvrtcResult (*PFN_nvrtcCompileProgram)(nvrtcProgram, int, const char* const*);
typedef nvrtcResult (*PFN_nvrtcGetPTXSize)(nvrtcProgram, size_t*);
typedef nvrtcResult (*PFN_nvrtcGetPTX)(nvrtcProgram, char*);
typedef nvrtcResult (*PFN_nvrtcGetProgramLogSize)(nvrtcProgram, size_t*);
typedef nvrtcResult (*PFN_nvrtcGetProgramLog)(nvrtcProgram, char*);
typedef nvrtcResult (*PFN_nvrtcDestroyProgram)(nvrtcProgram*);
typedef nvrtcResult (*PFN_nvrtcVersion)(int*, int*);

struct CudaApi {
    PFN_cuInit Init = nullptr;
    PFN_cuDeviceGet DeviceGet = nullptr;
    PFN_cuDeviceGetCount DeviceGetCount = nullptr;
    PFN_cuDeviceGetAttribute DeviceGetAttribute = nullptr;
    PFN_cuCtxCreate CtxCreate = nullptr;
    PFN_cuCtxDestroy CtxDestroy = nullptr;
    PFN_cuCtxSetCurrent CtxSetCurrent = nullptr;
    PFN_cuModuleLoadData ModuleLoadData = nullptr;
    PFN_cuModuleGetFunction ModuleGetFunction = nullptr;
    PFN_cuMemAlloc MemAlloc = nullptr;
    PFN_cuMemFree MemFree = nullptr;
    PFN_cuMemcpyHtoD MemcpyHtoD = nullptr;
    PFN_cuMemcpyDtoH MemcpyDtoH = nullptr;
    PFN_cuMemcpyDtoD MemcpyDtoD = nullptr;
    PFN_cuMemsetD8 MemsetD8 = nullptr;
    PFN_cuMemsetD32 MemsetD32 = nullptr;
    PFN_cuLaunchKernel LaunchKernel = nullptr;
    PFN_cuCtxSynchronize CtxSynchronize = nullptr;
    PFN_cuGetErrorName GetErrorName = nullptr;
};

struct NvrtcApi {
    PFN_nvrtcCreateProgram CreateProgram = nullptr;
    PFN_nvrtcCompileProgram CompileProgram = nullptr;
    PFN_nvrtcGetPTXSize GetPTXSize = nullptr;
    PFN_nvrtcGetPTX GetPTX = nullptr;
    PFN_nvrtcGetProgramLogSize GetProgramLogSize = nullptr;
    PFN_nvrtcGetProgramLog GetProgramLog = nullptr;
    PFN_nvrtcDestroyProgram DestroyProgram = nullptr;
    PFN_nvrtcVersion Version = nullptr;
};

#ifdef _WIN32
HMODULE g_nvcuda = nullptr;
HMODULE g_nvrtc_mod = nullptr;
#endif

CudaApi g_cu;
NvrtcApi g_nvrtc;
bool g_cuda_ready = false;

void die_cu(CUresult r, const char* what) {
    const char* name = "unknown";
    if (g_cu.GetErrorName) g_cu.GetErrorName(r, &name);
    std::fprintf(stderr, "[cuda] %s failed: %s (%d)\n", what, name, (int)r);
}

bool check_cu(CUresult r, const char* what) {
    if (r == CUDA_SUCCESS) return true;
    die_cu(r, what);
    return false;
}

#ifdef _WIN32
bool file_exists(const std::string& p) {
    DWORD a = GetFileAttributesA(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

std::string dirname_of(const std::string& p) {
    size_t s = p.find_last_of("\\/");
    if (s == std::string::npos) return ".";
    return p.substr(0, s);
}

std::vector<std::string> cuda_toolkit_bins() {
    std::vector<std::string> out;
    const char* env = std::getenv("CUDA_PATH");
    if (env) {
        out.push_back(std::string(env) + "\\bin");
    }
    std::string root = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA";
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((root + "\\v*").c_str(), &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                out.push_back(root + "\\" + fd.cFileName + "\\bin");
            }
        } while (FindNextFileA(h, &fd));
        FindClose(h);
    }
    return out;
}

HMODULE load_nvrtc_dll() {
    auto bins = cuda_toolkit_bins();
    std::vector<std::string> names = {"nvrtc64_130_0.dll", "nvrtc64_120_0.dll", "nvrtc64_118_0.dll",
                                      "nvrtc64_112_0.dll", "nvrtc64_111_0.dll", "nvrtc64_110_0.dll",
                                      "nvrtc64_102_0.dll"};
    for (const auto& bin : bins) {
        SetDllDirectoryA(bin.c_str());
        for (const auto& n : names) {
            std::string full = bin + "\\" + n;
            if (!file_exists(full)) continue;
            HMODULE h = LoadLibraryA(full.c_str());
            if (h) {
                std::fprintf(stderr, "[cuda] NVRTC %s\n", full.c_str());
                return h;
            }
        }
    }
    SetDllDirectoryA(nullptr);
    for (const auto& n : names) {
        HMODULE h = LoadLibraryA(n.c_str());
        if (h) return h;
    }
    return nullptr;
}
#endif

bool load_cuda_apis() {
    if (g_cuda_ready) return true;
#ifdef _WIN32
    g_nvcuda = LoadLibraryA("C:\\Windows\\System32\\nvcuda.dll");
    if (!g_nvcuda) g_nvcuda = LoadLibraryA("nvcuda.dll");
    if (!g_nvcuda) {
        std::fprintf(stderr, "[cuda] nvcuda.dll not found.\n");
        return false;
    }
#define GP(fn) reinterpret_cast<PFN_##fn>(GetProcAddress(g_nvcuda, "cu" #fn))
    g_cu.Init = GP(cuInit);
    // GetProcAddress names are the real symbol names.
#undef GP
    auto gp = [&](const char* n) { return GetProcAddress(g_nvcuda, n); };
    g_cu.Init = reinterpret_cast<PFN_cuInit>(gp("cuInit"));
    g_cu.DeviceGet = reinterpret_cast<PFN_cuDeviceGet>(gp("cuDeviceGet"));
    g_cu.DeviceGetCount = reinterpret_cast<PFN_cuDeviceGetCount>(gp("cuDeviceGetCount"));
    g_cu.DeviceGetAttribute = reinterpret_cast<PFN_cuDeviceGetAttribute>(gp("cuDeviceGetAttribute"));
    g_cu.CtxCreate = reinterpret_cast<PFN_cuCtxCreate>(gp("cuCtxCreate_v2"));
    if (!g_cu.CtxCreate) g_cu.CtxCreate = reinterpret_cast<PFN_cuCtxCreate>(gp("cuCtxCreate"));
    g_cu.CtxDestroy = reinterpret_cast<PFN_cuCtxDestroy>(gp("cuCtxDestroy_v2"));
    if (!g_cu.CtxDestroy) g_cu.CtxDestroy = reinterpret_cast<PFN_cuCtxDestroy>(gp("cuCtxDestroy"));
    g_cu.CtxSetCurrent = reinterpret_cast<PFN_cuCtxSetCurrent>(gp("cuCtxSetCurrent"));
    g_cu.ModuleLoadData = reinterpret_cast<PFN_cuModuleLoadData>(gp("cuModuleLoadData"));
    g_cu.ModuleGetFunction = reinterpret_cast<PFN_cuModuleGetFunction>(gp("cuModuleGetFunction"));
    g_cu.MemAlloc = reinterpret_cast<PFN_cuMemAlloc>(gp("cuMemAlloc_v2"));
    if (!g_cu.MemAlloc) g_cu.MemAlloc = reinterpret_cast<PFN_cuMemAlloc>(gp("cuMemAlloc"));
    g_cu.MemFree = reinterpret_cast<PFN_cuMemFree>(gp("cuMemFree_v2"));
    if (!g_cu.MemFree) g_cu.MemFree = reinterpret_cast<PFN_cuMemFree>(gp("cuMemFree"));
    g_cu.MemcpyHtoD = reinterpret_cast<PFN_cuMemcpyHtoD>(gp("cuMemcpyHtoD_v2"));
    if (!g_cu.MemcpyHtoD) g_cu.MemcpyHtoD = reinterpret_cast<PFN_cuMemcpyHtoD>(gp("cuMemcpyHtoD"));
    g_cu.MemcpyDtoH = reinterpret_cast<PFN_cuMemcpyDtoH>(gp("cuMemcpyDtoH_v2"));
    if (!g_cu.MemcpyDtoH) g_cu.MemcpyDtoH = reinterpret_cast<PFN_cuMemcpyDtoH>(gp("cuMemcpyDtoH"));
    g_cu.MemcpyDtoD = reinterpret_cast<PFN_cuMemcpyDtoD>(gp("cuMemcpyDtoD_v2"));
    if (!g_cu.MemcpyDtoD) g_cu.MemcpyDtoD = reinterpret_cast<PFN_cuMemcpyDtoD>(gp("cuMemcpyDtoD"));
    g_cu.MemsetD8 = reinterpret_cast<PFN_cuMemsetD8>(gp("cuMemsetD8"));
    g_cu.MemsetD32 = reinterpret_cast<PFN_cuMemsetD32>(gp("cuMemsetD32"));
    g_cu.LaunchKernel = reinterpret_cast<PFN_cuLaunchKernel>(gp("cuLaunchKernel"));
    g_cu.CtxSynchronize = reinterpret_cast<PFN_cuCtxSynchronize>(gp("cuCtxSynchronize"));
    g_cu.GetErrorName = reinterpret_cast<PFN_cuGetErrorName>(gp("cuGetErrorName"));
    if (!g_cu.Init || !g_cu.MemAlloc || !g_cu.LaunchKernel) {
        std::fprintf(stderr, "[cuda] missing CUDA Driver API symbols.\n");
        return false;
    }

    g_nvrtc_mod = load_nvrtc_dll();
    if (!g_nvrtc_mod) {
        std::fprintf(stderr, "[cuda] nvrtc DLL not found. CUDA Toolkit required.\n");
        return false;
    }
    auto np = [&](const char* n) { return GetProcAddress(g_nvrtc_mod, n); };
    g_nvrtc.CreateProgram = reinterpret_cast<PFN_nvrtcCreateProgram>(np("nvrtcCreateProgram"));
    g_nvrtc.CompileProgram = reinterpret_cast<PFN_nvrtcCompileProgram>(np("nvrtcCompileProgram"));
    g_nvrtc.GetPTXSize = reinterpret_cast<PFN_nvrtcGetPTXSize>(np("nvrtcGetPTXSize"));
    g_nvrtc.GetPTX = reinterpret_cast<PFN_nvrtcGetPTX>(np("nvrtcGetPTX"));
    g_nvrtc.GetProgramLogSize = reinterpret_cast<PFN_nvrtcGetProgramLogSize>(np("nvrtcGetProgramLogSize"));
    g_nvrtc.GetProgramLog = reinterpret_cast<PFN_nvrtcGetProgramLog>(np("nvrtcGetProgramLog"));
    g_nvrtc.DestroyProgram = reinterpret_cast<PFN_nvrtcDestroyProgram>(np("nvrtcDestroyProgram"));
    g_nvrtc.Version = reinterpret_cast<PFN_nvrtcVersion>(np("nvrtcVersion"));
    if (!g_nvrtc.CreateProgram || !g_nvrtc.CompileProgram) {
        std::fprintf(stderr, "[cuda] missing NVRTC symbols.\n");
        return false;
    }
#else
    std::fprintf(stderr, "[cuda] Windows only in this build.\n");
    return false;
#endif
    g_cuda_ready = true;
    return true;
}

std::string module_dir() {
#ifdef _WIN32
    HMODULE h = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCSTR>(&module_dir), &h);
    char buf[MAX_PATH];
    GetModuleFileNameA(h, buf, MAX_PATH);
    return dirname_of(buf);
#else
    return ".";
#endif
}

std::string read_kernels() {
    std::vector<std::string> cands;
    cands.push_back(module_dir() + "\\fdtd_kernels.cu");
    if (const char* env = std::getenv("ANTENNA_NATIVE")) {
        cands.push_back(std::string(env) + "\\fdtd_kernels.cu");
    }
    cands.push_back("native\\fdtd_kernels.cu");
    cands.push_back("fdtd_kernels.cu");
    for (const auto& p : cands) {
        std::ifstream in(p, std::ios::binary);
        if (!in) continue;
        std::ostringstream ss;
        ss << in.rdbuf();
        std::fprintf(stderr, "[cuda] kernels %s\n", p.c_str());
        return ss.str();
    }
    return {};
}

std::string compile_ptx(const std::string& src, int cap_major, int cap_minor) {
    int vm = 0, vn = 0;
    if (g_nvrtc.Version) g_nvrtc.Version(&vm, &vn);
    std::string arch = "--gpu-architecture=compute_75";
    if (vm > 11 || (vm == 11 && vn >= 1)) {
        if (cap_major >= 8 && cap_minor >= 6)
            arch = "--gpu-architecture=compute_86";
        else if (cap_major >= 8)
            arch = "--gpu-architecture=compute_80";
    } else if (vm == 11) {
        arch = "--gpu-architecture=compute_80";
    }
    std::fprintf(stderr, "[cuda] NVRTC %d.%d  %s  (gpu sm_%d%d)\n", vm, vn, arch.c_str(), cap_major,
                 cap_minor);

    nvrtcProgram prog = nullptr;
    nvrtcResult rr =
        g_nvrtc.CreateProgram(&prog, src.c_str(), "fdtd_kernels.cu", 0, nullptr, nullptr);
    if (rr != NVRTC_SUCCESS) {
        std::fprintf(stderr, "[cuda] nvrtcCreateProgram %d\n", (int)rr);
        return {};
    }
    const char* opts[] = {arch.c_str(), "--use_fast_math"};
    rr = g_nvrtc.CompileProgram(prog, 2, opts);
    size_t logsz = 0;
    g_nvrtc.GetProgramLogSize(prog, &logsz);
    if (logsz > 1) {
        std::string log(logsz, '\0');
        g_nvrtc.GetProgramLog(prog, log.data());
        std::fprintf(stderr, "[cuda] NVRTC log:\n%s\n", log.c_str());
    }
    if (rr != NVRTC_SUCCESS) {
        g_nvrtc.DestroyProgram(&prog);
        return {};
    }
    size_t ptxsz = 0;
    g_nvrtc.GetPTXSize(prog, &ptxsz);
    std::string ptx(ptxsz, '\0');
    g_nvrtc.GetPTX(prog, ptx.data());
    g_nvrtc.DestroyProgram(&prog);
    return ptx;
}

struct DeviceBuf {
    CudaApi* api = nullptr;
    CUdeviceptr p = 0;
    size_t bytes = 0;
    bool alloc(CudaApi* a, size_t n) {
        api = a;
        bytes = n;
        return a->MemAlloc(&p, n) == CUDA_SUCCESS;
    }
    void free() {
        if (api && p) api->MemFree(p);
        p = 0;
    }
};

struct Solver {
    CudaApi* cu = nullptr;
    CUcontext ctx = nullptr;
    CUmodule mod = nullptr;
    CUfunction fH{}, fE{}, fLump{}, fMurX{}, fMurY{}, fMurZ{}, fPec{}, fSamp{}, fEner{}, fAir{},
        fCut{}, fFace{}, fSlice{}, fCutMax{}, fZero{}, fNtff{};

    DeviceBuf Ex, Ey, Ez, Hx, Hy, Hz, Exo, Eyo, Ezo;
    DeviceBuf Caex, Cbex, Caey, Cbey, Caez, Cbez;
    DeviceBuf pecEx, pecEy, pecEz;
    DeviceBuf Vrec, Irec, Vsrec, energy, airE, patchE, midE, topE, cutE, cutMaxE, faceE;
    DeviceBuf ntBuf, ntTw;
    std::vector<float> hAir, hPatch, hMid, hTop, hCut, hCutMax, hFace;
    int air_n = 0;

    std::vector<float> hCaex, hCbex, hCaey, hCbey, hCaez, hCbez;
    std::vector<uint8_t> hPecEx, hPecEy, hPecEz, metal, metal_lo;
    bool lower_full_gnd = false;  // gate: full GND plane, same as openEMS
    int feed_i = PAD + PIX / 2;
    int feed_j = PAD + PIX / 2;
    float dt = 0.f;
    float mur_x = 0.f, mur_y = 0.f, mur_z = 0.f;
    float dt_mu = 0.f;
    float t_sec = 3.5e-9f;
    float cfl = 0.99f;
    float f0_hz = F0_DEFAULT;
    float fc_hz = FC_DEFAULT;
    float leave_ratio = 0.f;
    // NTFF
    int nt_nf = 7;
    double nt_freq[NT_FMAX] = {3e9, 4e9, 5e9, 6e9, 7e9, 8e9, 9e9};
    std::vector<float> nt_buf;
    double nt_pinc[NT_FMAX]{};
    double nt_prad[NT_FMAX]{};

    ~Solver() { close(); }

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
    static void cab(float eps, float sig, float dtv, float& Ca, float& Cb) {
        const float denom = 1.f + sig * dtv / (2.f * eps);
        Ca = (1.f - sig * dtv / (2.f * eps)) / denom;
        Cb = (dtv / eps) / denom;
    }

    void build_materials() {
        hCaex.assign(N, 0);
        hCbex.assign(N, 0);
        hCaey.assign(N, 0);
        hCbey.assign(N, 0);
        hCaez.assign(N, 0);
        hCbez.assign(N, 0);
        for (int k = 0; k < NZ; ++k)
            for (int j = 0; j < NY; ++j)
                for (int i = 0; i < NX; ++i) {
                    const int p = idx(i, j, k);
                    float e = eps_at(i, j, k);
                    float s = sig_at(i, j, k);
                    cab(e, s, dt, hCaex[p], hCbex[p]);
                    cab(e, s, dt, hCaey[p], hCbey[p]);
                    cab(e, s, dt, hCaez[p], hCbez[p]);
                }
    }

    void apply_metal_bitmap() {
        hPecEx.assign(N, 0);
        hPecEy.assign(N, 0);
        hPecEz.assign(N, 0);
        const int i0 = PAD, j0 = PAD;
        const int i1 = PAD + PIX, j1 = PAD + PIX;
        if (lower_full_gnd) {
            for (int j = j0; j <= j1; ++j)
                for (int i = i0; i < i1; ++i)
                    hPecEx[idx(i, j, K_GND)] = 1;
            for (int j = j0; j < j1; ++j)
                for (int i = i0; i <= i1; ++i)
                    hPecEy[idx(i, j, K_GND)] = 1;
        }
        for (int q = 0; q < PIX; ++q) {
            for (int p = 0; p < PIX; ++p) {
                const int i = i0 + p;
                const int j = j0 + q;
                if (!lower_full_gnd && metal_lo[p + PIX * q]) {
                    hPecEx[idx(i, j, K_GND)] = 1;
                    hPecEx[idx(i, j + 1, K_GND)] = 1;
                    hPecEy[idx(i, j, K_GND)] = 1;
                    hPecEy[idx(i + 1, j, K_GND)] = 1;
                }
                if (!metal[p + PIX * q]) continue;
                hPecEx[idx(i, j, K_PATCH)] = 1;
                hPecEx[idx(i, j + 1, K_PATCH)] = 1;
                hPecEy[idx(i, j, K_PATCH)] = 1;
                hPecEy[idx(i + 1, j, K_PATCH)] = 1;
            }
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

    bool launch(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                unsigned bz, void** args) {
        CUresult r = cu->LaunchKernel(f, gx, gy, gz, bx, by, bz, 0, nullptr, args, nullptr);
        return check_cu(r, "cuLaunchKernel");
    }

    bool init() {
        if (!load_cuda_apis()) return false;
        cu = &g_cu;
        CUresult r = cu->Init(0);
        if (!check_cu(r, "cuInit")) return false;
        int ndev = 0;
        if (!check_cu(cu->DeviceGetCount(&ndev), "cuDeviceGetCount") || ndev < 1) {
            std::fprintf(stderr, "[cuda] no GPU\n");
            return false;
        }
        CUdevice dev = 0;
        if (!check_cu(cu->DeviceGet(&dev, 0), "cuDeviceGet")) return false;
        int maj = 0, minv = 0;
        cu->DeviceGetAttribute(&maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
        cu->DeviceGetAttribute(&minv, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
        if (!check_cu(cu->CtxCreate(&ctx, 0, dev), "cuCtxCreate")) return false;

        std::string src = read_kernels();
        if (src.empty()) {
            std::fprintf(stderr, "[cuda] fdtd_kernels.cu not found.\n");
            return false;
        }
        std::string ptx = compile_ptx(src, maj, minv);
        if (ptx.empty()) return false;
        if (!check_cu(cu->ModuleLoadData(&mod, ptx.c_str()), "cuModuleLoadData")) return false;

        auto gf = [&](CUfunction* f, const char* n) {
            return check_cu(cu->ModuleGetFunction(f, mod, n), n);
        };
        if (!gf(&fH, "k_update_H") || !gf(&fE, "k_update_E") || !gf(&fLump, "k_lumped") ||
            !gf(&fMurX, "k_mur_x") || !gf(&fMurY, "k_mur_y") || !gf(&fMurZ, "k_mur_z") ||
            !gf(&fPec, "k_pec") || !gf(&fSamp, "k_sample") || !gf(&fEner, "k_energy") ||
            !gf(&fAir, "k_accum_air") || !gf(&fCut, "k_accum_cut") ||
            !gf(&fFace, "k_accum_faces") || !gf(&fSlice, "k_accum_slice") ||
            !gf(&fCutMax, "k_accum_cut_max") || !gf(&fZero, "k_zero_f32") ||
            !gf(&fNtff, "k_ntff_accum"))
            return false;

        auto al = [&](DeviceBuf& b, size_t n) {
            if (!b.alloc(cu, n)) {
                std::fprintf(stderr, "[cuda] cuMemAlloc %zu failed (out of VRAM?)\n", n);
                return false;
            }
            return true;
        };
        const size_t nf = sizeof(float) * N;
        if (!al(Ex, nf) || !al(Ey, nf) || !al(Ez, nf) || !al(Hx, nf) || !al(Hy, nf) || !al(Hz, nf) ||
            !al(Exo, nf) || !al(Eyo, nf) || !al(Ezo, nf) || !al(Caex, nf) || !al(Cbex, nf) ||
            !al(Caey, nf) || !al(Cbey, nf) || !al(Caez, nf) || !al(Cbez, nf) || !al(pecEx, N) ||
            !al(pecEy, N) || !al(pecEz, N) || !al(Vrec, sizeof(float) * NSTEPS) ||
            !al(Irec, sizeof(float) * NSTEPS) || !al(Vsrec, sizeof(float) * NSTEPS) ||
            !al(energy, sizeof(float)) || !al(airE, sizeof(float) * NX * NY) ||
            !al(patchE, sizeof(float) * NX * NY) || !al(midE, sizeof(float) * NX * NY) ||
            !al(topE, sizeof(float) * NX * NY) || !al(cutE, sizeof(float) * NX * NZ) ||
            !al(cutMaxE, sizeof(float) * NX * NZ) || !al(faceE, sizeof(float) * 12) ||
            !al(ntBuf, sizeof(float) * size_t(NT_CELLS) * NT_FMAX * 8) || !al(ntTw, sizeof(float) * 4 * NT_FMAX))
            return false;

        metal.assign(PIX * PIX, 0);
        metal_lo.assign(PIX * PIX, 0);
        fill_rect_mask(metal_lo, PATCH_LX_MM, PATCH_LY_MM);
        apply_cfl();
        std::fprintf(stderr, "[cuda] ready  grid=%dx%dx%d  dt=%.4g\n", NX, NY, NZ, dt);
        return true;
    }

    void apply_cfl() {
        cfl = std::clamp(cfl, 0.10f, 0.99f);
        t_sec = std::max(t_sec, 1e-12f);
        const float inv = std::sqrt(1.f / (DX * DX) + 1.f / (DY * DY) + 1.f / (DZ * DZ));
        dt = cfl / (float(C0) * inv);
        mur_x = (float(C0) * dt - DX) / (float(C0) * dt + DX);
        mur_y = (float(C0) * dt - DY) / (float(C0) * dt + DY);
        mur_z = (float(C0) * dt - DZ) / (float(C0) * dt + DZ);
        dt_mu = dt / float(MU0);
        build_materials();
        if (cu && ctx && Caex.p) {
            cu->CtxSetCurrent(ctx);
            const size_t nf = sizeof(float) * N;
            cu->MemcpyHtoD(Caex.p, hCaex.data(), nf);
            cu->MemcpyHtoD(Cbex.p, hCbex.data(), nf);
            cu->MemcpyHtoD(Caey.p, hCaey.data(), nf);
            cu->MemcpyHtoD(Cbey.p, hCbey.data(), nf);
            cu->MemcpyHtoD(Caez.p, hCaez.data(), nf);
            cu->MemcpyHtoD(Cbez.p, hCbez.data(), nf);
        }
    }

    int nplan() const {
        int n = (int)std::ceil(double(t_sec) / double(dt));
        return std::clamp(n, 1, NSTEPS);
    }

    void close() {
        if (!cu) return;
        if (ctx) cu->CtxSetCurrent(ctx);
        Ex.free();
        Ey.free();
        Ez.free();
        Hx.free();
        Hy.free();
        Hz.free();
        Exo.free();
        Eyo.free();
        Ezo.free();
        Caex.free();
        Cbex.free();
        Caey.free();
        Cbey.free();
        Caez.free();
        Cbez.free();
        pecEx.free();
        pecEy.free();
        pecEz.free();
        Vrec.free();
        Irec.free();
        Vsrec.free();
        energy.free();
        airE.free();
        patchE.free();
        midE.free();
        topE.free();
        cutE.free();
        cutMaxE.free();
        faceE.free();
        ntBuf.free();
        ntTw.free();
        if (ctx) {
            cu->CtxDestroy(ctx);
            ctx = nullptr;
        }
        cu = nullptr;
    }

    void zero_f32(DeviceBuf& b) {
        if (!b.p || !b.bytes) return;
        const int n = int(b.bytes / sizeof(float));
        if (n <= 0) return;
        CUdeviceptr dp = b.p;
        int nv = n;
        unsigned blocks = (unsigned(n) + 255u) / 256u;
        void* args[] = {&dp, &nv};
        launch(fZero, blocks, 1, 1, 256, 1, 1, args);
    }

    int run(int nfreq, const double* freqs, double* s11_db, double* min_db, double* min_hz) {
        cu->CtxSetCurrent(ctx);
        hAir.clear();
        hPatch.clear();
        hMid.clear();
        hTop.clear();
        hCut.clear();
        hCutMax.clear();
        hFace.clear();
        apply_metal_bitmap();
        zero_f32(Ex);
        zero_f32(Ey);
        zero_f32(Ez);
        zero_f32(Hx);
        zero_f32(Hy);
        zero_f32(Hz);
        zero_f32(Exo);
        zero_f32(Eyo);
        zero_f32(Ezo);
        cu->MemcpyHtoD(pecEx.p, hPecEx.data(), N);
        cu->MemcpyHtoD(pecEy.p, hPecEy.data(), N);
        cu->MemcpyHtoD(pecEz.p, hPecEz.data(), N);
        zero_f32(airE);
        zero_f32(patchE);
        zero_f32(midE);
        zero_f32(topE);
        zero_f32(cutE);
        zero_f32(cutMaxE);
        zero_f32(faceE);
        zero_f32(Vrec);
        zero_f32(Irec);
        zero_f32(Vsrec);
        zero_f32(energy);
        zero_f32(ntBuf);
        air_n = 0;
        /* k_accum_* use +=, so start timestepping only after init and PEC upload are done */
        cu->CtxSynchronize();

        unsigned bx = 8, by = 8, bz = 4;
        unsigned gx = (NX + bx - 1) / bx, gy = (NY + by - 1) / by, gz = (NZ + bz - 1) / bz;
        unsigned pec_threads = 256;
        unsigned pec_blocks = (N + pec_threads - 1) / pec_threads;
        unsigned mx = 16, my = 16;
        unsigned gmurj = (NY + mx - 1) / mx, gmurk = (NZ + my - 1) / my;
        unsigned gmuri = (NX + mx - 1) / mx;

        int nused = nplan();
        float emax = 1e-30f;
        CUdeviceptr dEx = Ex.p, dEy = Ey.p, dEz = Ez.p, dHx = Hx.p, dHy = Hy.p, dHz = Hz.p;
        CUdeviceptr dExo = Exo.p, dEyo = Eyo.p, dEzo = Ezo.p;
        CUdeviceptr dCaex = Caex.p, dCbex = Cbex.p, dCaey = Caey.p, dCbey = Cbey.p, dCaez = Caez.p,
                    dCbez = Cbez.p;
        CUdeviceptr dPex = pecEx.p, dPey = pecEy.p, dPez = pecEz.p;
        CUdeviceptr dV = Vrec.p, dI = Irec.p, dVs = Vsrec.p, dEne = energy.p;
        float dtmu = dt_mu;
        int fi = feed_i, fj = feed_j;
        float mxv = mur_x, myv = mur_y, mzv = mur_z;
        float dtv = dt;

        for (int n = 0; n < nused; ++n) {
            const float t = (n + 0.5f) * dt;
            const float vs = source(t);

            void* aH[] = {&dHx, &dHy, &dHz, &dEx, &dEy, &dEz, &dtmu};
            launch(fH, gx, gy, gz, bx, by, bz, aH);

            cu->MemcpyDtoD(dExo, dEx, Ex.bytes);
            cu->MemcpyDtoD(dEyo, dEy, Ey.bytes);
            cu->MemcpyDtoD(dEzo, dEz, Ez.bytes);

            void* aE[] = {&dEx,  &dEy,   &dEz,   &dHx,   &dHy,   &dHz, &dCaex, &dCbex,
                          &dCaey, &dCbey, &dCaez, &dCbez, &fi,    &fj};
            launch(fE, gx, gy, gz, bx, by, bz, aE);

            float vsv = vs;
            void* aL[] = {&dEz, &dHx, &dHy, &fi, &fj, &dtv, &vsv};
            launch(fLump, 1, 1, 1, 8, 1, 1, aL);

            void* aMx[] = {&dEy, &dEz, &dEyo, &dEzo, &mxv};
            launch(fMurX, gmurj, gmurk, 1, mx, my, 1, aMx);
            void* aMy[] = {&dEx, &dEz, &dExo, &dEzo, &myv};
            launch(fMurY, gmuri, gmurk, 1, mx, my, 1, aMy);
            void* aMz[] = {&dEx, &dEy, &dExo, &dEyo, &mzv};
            launch(fMurZ, gmuri, gmurj, 1, mx, my, 1, aMz);

            void* aP[] = {&dEx, &dEy, &dEz, &dPex, &dPey, &dPez};
            launch(fPec, pec_blocks, 1, 1, pec_threads, 1, 1, aP);

            int nn = n;
            void* aS[] = {&dEz, &dHx, &dHy, &dV, &dI, &dVs, &fi, &fj, &nn, &vsv};
            launch(fSamp, 1, 1, 1, 1, 1, 1, aS);

            if ((n % NT_EVERY) == 0) {
                float tw[4 * NT_FMAX] = {0};
                const double w_dt = double(dt) * NT_EVERY;
                const double tE = double(n + 1) * dt, tH = (double(n) + 0.5) * dt;
                for (int f = 0; f < nt_nf; ++f) {
                    const double w = 2.0 * M_PI * nt_freq[f];
                    tw[f] = float(std::cos(w * tE) * w_dt);
                    tw[NT_FMAX + f] = float(std::sin(w * tE) * w_dt);
                    tw[2 * NT_FMAX + f] = float(std::cos(w * tH) * w_dt);
                    tw[3 * NT_FMAX + f] = float(std::sin(w * tH) * w_dt);
                }
                cu->MemcpyHtoD(ntTw.p, tw, sizeof(tw));
                CUdeviceptr dNb = ntBuf.p, dTw = ntTw.p;
                int nfv = nt_nf;
                void* aN[] = {&dEx, &dEy, &dEz, &dHx, &dHy, &dHz, &dNb, &dTw, &nfv};
                launch(fNtff, (NT_CELLS + 255) / 256, 1, 1, 256, 1, 1, aN);
            }

            if ((n & 7) == 0 && n > 200) {
                CUdeviceptr dAir = airE.p;
                unsigned agx = (NX + 15) / 16, agy = (NY + 15) / 16;
                void* aA[] = {&dEx, &dEy, &dEz, &dAir};
                launch(fAir, agx, agy, 1, 16, 16, 1, aA);
                CUdeviceptr dCut = cutE.p;
                unsigned cgx = (NX + 15) / 16, cgz = (NZ + 15) / 16;
                void* aC[] = {&dEx, &dEy, &dEz, &dCut, &fj};
                launch(fCut, cgx, cgz, 1, 16, 16, 1, aC);
                CUdeviceptr dCutMax = cutMaxE.p;
                void* aCm[] = {&dEx, &dEy, &dEz, &dCutMax};
                launch(fCutMax, cgx, cgz, 1, 16, 16, 1, aCm);
                auto slice = [&](CUdeviceptr dst, int kL) {
                    int kv = kL;
                    void* aS[] = {&dEx, &dEy, &dEz, &dst, &kv};
                    launch(fSlice, agx, agy, 1, 16, 16, 1, aS);
                };
                slice(patchE.p, K_PATCH);
                slice(midE.p, K_MID);
                slice(topE.p, K_TOP);
                CUdeviceptr dFace = faceE.p;
                void* aF[] = {&dEx, &dEy, &dEz, &dFace};
                launch(fFace, 32, 1, 1, 256, 1, 1, aF);
                ++air_n;
            }

            if ((n & 255) == 0) {
                float zero = 0.f;
                cu->MemcpyHtoD(dEne, &zero, sizeof(float));
                int eblocks = 64;
                void* aEn[] = {&dEx, &dEy, &dEz, &dHx, &dHy, &dHz, &dEne};
                launch(fEner, eblocks, 1, 1, 256, 1, 1, aEn);
                float e = 0.f;
                cu->MemcpyDtoH(&e, dEne, sizeof(float));
                if (e > emax) emax = e;
            }
        }
        cu->CtxSynchronize();

        std::vector<float> hV(nused), hI(nused), hS(nused);
        cu->MemcpyDtoH(hV.data(), Vrec.p, sizeof(float) * nused);
        cu->MemcpyDtoH(hI.data(), Irec.p, sizeof(float) * nused);
        cu->MemcpyDtoH(hS.data(), Vsrec.p, sizeof(float) * nused);

        float vmax = 0.f, imax = 0.f;
        double corr = 0.0;
        for (int n = 0; n < nused; ++n) {
            vmax = std::max(vmax, std::abs(hV[n]));
            imax = std::max(imax, std::abs(hI[n]));
            corr += double(hV[n]) * double(hS[n]);
        }
        double einc = 0.0;
        for (int n = 0; n < nused; ++n)
            einc += double(hS[n]) * hS[n] * double(dt) / (8.0 * double(RPORT));
        (void)einc;
        nt_buf.assign(size_t(NT_CELLS) * nt_nf * 8, 0.f);
        cu->MemcpyDtoH(nt_buf.data(), ntBuf.p, nt_buf.size() * sizeof(float));
        nt_finish(hS, nused);
        {
            // leave = radiated/incident power (NTFF, frequency-averaged) = total efficiency (includes mismatch and FR4 loss)
            double acc = 0.0;
            for (int f = 0; f < nt_nf; ++f) acc += nt_prad[f] / (nt_pinc[f] + 1e-300);
            leave_ratio = float(acc / std::max(nt_nf, 1));
        }
        std::fprintf(stderr,
                     "[voxel-cuda] nused=%d t=%.3gns cfl=%.2f vmax=%.4g leave=%.4g dt=%.4g\n",
                     nused, double(t_sec) * 1e9, cfl, vmax, leave_ratio, dt);

        double best = 1e9, bestf = freqs[0];
        for (int f = 0; f < nfreq; ++f) {
            const double w = 2.0 * M_PI * freqs[f];
            double vr = 0, vi = 0, srcr = 0, srci = 0;
            for (int n = 0; n < nused; ++n) {
                const double ph = w * n * dt;
                const double c = std::cos(ph);
                const double s = std::sin(ph);
                vr += hV[n] * c;
                vi -= hV[n] * s;
                srcr += hS[n] * c;
                srci -= hS[n] * s;
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
        hAir.assign(NX * NY, 0);
        hPatch.assign(NX * NY, 0);
        hMid.assign(NX * NY, 0);
        hTop.assign(NX * NY, 0);
        hCut.assign(NX * NZ, 0);
        hCutMax.assign(NX * NZ, 0);
        cu->MemcpyDtoH(hAir.data(), airE.p, sizeof(float) * NX * NY);
        cu->MemcpyDtoH(hPatch.data(), patchE.p, sizeof(float) * NX * NY);
        cu->MemcpyDtoH(hMid.data(), midE.p, sizeof(float) * NX * NY);
        cu->MemcpyDtoH(hTop.data(), topE.p, sizeof(float) * NX * NY);
        cu->MemcpyDtoH(hCut.data(), cutE.p, sizeof(float) * NX * NZ);
        cu->MemcpyDtoH(hCutMax.data(), cutMaxE.p, sizeof(float) * NX * NZ);
        const float inv = (air_n > 0) ? 1.f / float(air_n) : 1.f;
        auto sqrt_layer = [&](std::vector<float>& h) {
            for (int n = 0; n < int(h.size()); ++n) h[n] = std::sqrt(h[n] * inv);
        };
        sqrt_layer(hAir);
        sqrt_layer(hPatch);
        sqrt_layer(hMid);
        sqrt_layer(hTop);
        sqrt_layer(hCut);
        sqrt_layer(hCutMax);
        std::vector<float> raw(12, 0.f);
        cu->MemcpyDtoH(raw.data(), faceE.p, sizeof(float) * 12);
        const float ax = float((NY - 4) * (NZ - 4)) * DY * DZ;
        const float ay = float((NX - 4) * (NZ - 4)) * DX * DZ;
        const float az = float((NX - 4) * (NY - 4)) * DX * DY;
        const float area[6] = {ax, ax, ay, ay, az, az};
        float pall = 1e-30f;
        for (int i = 0; i < 6; ++i) pall += raw[i];
        hFace.assign(12, 0.f);
        for (int i = 0; i < 6; ++i) hFace[i] = raw[i] * inv / area[i];
        for (int i = 0; i < 6; ++i) hFace[6 + i] = raw[6 + i] / pall;
        return nused;
    }

    void copy_air(float* out) const {
        if (hAir.empty()) {
            std::fill(out, out + NX * NY, 0.f);
            return;
        }
        std::memcpy(out, hAir.data(), sizeof(float) * NX * NY);
    }

    void copy_cut(float* out) const {
        if (hCut.empty()) {
            std::fill(out, out + NX * NZ, 0.f);
            return;
        }
        std::memcpy(out, hCut.data(), sizeof(float) * NX * NZ);
    }

    void copy_layer(const std::vector<float>& h, float* out) const {
        if (h.empty()) {
            std::fill(out, out + NX * NY, 0.f);
            return;
        }
        std::memcpy(out, h.data(), sizeof(float) * NX * NY);
    }

    void copy_cut_max(float* out) const {
        if (hCutMax.empty()) {
            std::fill(out, out + NX * NZ, 0.f);
            return;
        }
        std::memcpy(out, hCutMax.data(), sizeof(float) * NX * NZ);
    }

    // P_rad(f) = ½ Re ∮ (E×H*)·n dA,   P_inc(f) = |Vs(f)|² / (8 R)   (same as voxel_fdtd.cpp)
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
                const int nc = nt_face_off(f6 + 1) - nt_face_off(f6);
                double acc = 0.0;
                for (int l = 0; l < nc; ++l) {
                    const float* q = nt_buf.data() + (size_t(nt_face_off(f6) + l) * nf + f) * 8;
                    acc += (double(q[0]) * q[6] + double(q[1]) * q[7]) - (double(q[2]) * q[4] + double(q[3]) * q[5]);
                }
                const double orient = (f6 == 2 || f6 == 3) ? -1.0 : 1.0;
                p += 0.5 * sgn * orient * acc * da;
            }
            nt_prad[f] = p;
        }
    }

    void copy_faces(float* out) const {
        if (hFace.size() != 12) {
            std::fill(out, out + 12, 0.f);
            return;
        }
        std::memcpy(out, hFace.data(), sizeof(float) * 12);
    }

    void copy_leave(float* out) const {
        if (out) *out = leave_ratio;
    }
};

}  // namespace

API void* voxel_create() {
    auto* s = new Solver();
    if (!s->init()) {
        delete s;
        return nullptr;
    }
    return s;
}

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
    std::fill(s->metal.begin(), s->metal.end(), 0);
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

API void voxel_faces(void* ctx, float* out) {
    auto* s = static_cast<Solver*>(ctx);
    if (out) s->copy_faces(out);
}

API void voxel_extra_fields(void* ctx, float* patch, float* mid, float* top, float* cut_max, int* nx,
                            int* ny, int* nz) {
    auto* s = static_cast<Solver*>(ctx);
    if (nx) *nx = NX;
    if (ny) *ny = NY;
    if (nz) *nz = NZ;
    if (patch) s->copy_layer(s->hPatch, patch);
    if (mid) s->copy_layer(s->hMid, mid);
    if (top) s->copy_layer(s->hTop, top);
    if (cut_max) s->copy_cut_max(cut_max);
}

API void voxel_ntff_set(void* ctx, int nf, const double* freqs) {
    auto* s = static_cast<Solver*>(ctx);
    nf = std::clamp(nf, 1, NT_FMAX);
    s->nt_nf = nf;
    for (int f = 0; f < nf; ++f) s->nt_freq[f] = freqs[f];
}

API void voxel_ntff_info(void* ctx, int* info, double* d3) {
    auto* s = static_cast<Solver*>(ctx);
    const int v[13] = {NT_IA, NT_IB, NT_JA, NT_JB, NT_KA, NT_KB, s->nt_nf, NT_CELLS, NT_LX, NT_LY, NT_LZ,
                       K_PATCH, PAD + PIX / 2};
    if (info) std::memcpy(info, v, sizeof(v));
    if (d3) { d3[0] = DX; d3[1] = DY; d3[2] = DZ; }
}

API void voxel_ntff_get(void* ctx, float* buf, double* freq, double* pinc, double* prad) {
    auto* s = static_cast<Solver*>(ctx);
    if (buf && !s->nt_buf.empty()) std::memcpy(buf, s->nt_buf.data(), s->nt_buf.size() * sizeof(float));
    for (int f = 0; f < s->nt_nf; ++f) {
        if (freq) freq[f] = s->nt_freq[f];
        if (pinc) pinc[f] = s->nt_pinc[f];
        if (prad) prad[f] = s->nt_prad[f];
    }
}
