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
constexpr int NZ_BELOW = 12;
constexpr int NZ_SUB = 4;
constexpr int NZ_ABOVE = 20;
constexpr int NZ = NZ_BELOW + NZ_SUB + NZ_ABOVE;
constexpr int N = NX * NY * NZ;

constexpr int K_GND = NZ_BELOW;
constexpr int K_PATCH = NZ_BELOW + NZ_SUB;

constexpr float DX = 1.0e-3f;
constexpr float DY = 1.0e-3f;
constexpr float DZ = 0.375e-3f;
constexpr float EPS_R = 4.4f;
constexpr float TAN_D = 0.02f;
constexpr float F0 = 7.25e9f;
constexpr float FC = 1.25e9f;
constexpr float RPORT = 50.0f;
constexpr int NSTEPS = 12000;
constexpr float END_CRIT = 1e-4f;

inline int idx(int i, int j, int k) { return i + NX * (j + NY * k); }

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
        std::fprintf(stderr, "[cuda] nvcuda.dll 을 찾지 못했습니다.\n");
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
    g_cu.LaunchKernel = reinterpret_cast<PFN_cuLaunchKernel>(gp("cuLaunchKernel"));
    g_cu.CtxSynchronize = reinterpret_cast<PFN_cuCtxSynchronize>(gp("cuCtxSynchronize"));
    g_cu.GetErrorName = reinterpret_cast<PFN_cuGetErrorName>(gp("cuGetErrorName"));
    if (!g_cu.Init || !g_cu.MemAlloc || !g_cu.LaunchKernel) {
        std::fprintf(stderr, "[cuda] CUDA Driver API 심볼이 부족합니다.\n");
        return false;
    }

    g_nvrtc_mod = load_nvrtc_dll();
    if (!g_nvrtc_mod) {
        std::fprintf(stderr, "[cuda] nvrtc DLL 을 찾지 못했습니다. CUDA Toolkit 이 필요합니다.\n");
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
        std::fprintf(stderr, "[cuda] NVRTC 심볼이 부족합니다.\n");
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
    CUfunction fH{}, fE{}, fLump{}, fMurX{}, fMurY{}, fMurZ{}, fPec{}, fSamp{}, fEner{}, fAir{};

    DeviceBuf Ex, Ey, Ez, Hx, Hy, Hz, Exo, Eyo, Ezo;
    DeviceBuf Caex, Cbex, Caey, Cbey, Caez, Cbez;
    DeviceBuf pecEx, pecEy, pecEz;
    DeviceBuf Vrec, Irec, Vsrec, energy, airE;
    std::vector<float> hAir;
    int air_n = 0;

    std::vector<float> hCaex, hCbex, hCaey, hCbey, hCaez, hCbez;
    std::vector<uint8_t> hPecEx, hPecEy, hPecEz, metal;
    int feed_i = PAD + PIX / 2;
    int feed_j = PAD + PIX / 2;
    float dt = 0.f;
    float mur_x = 0.f, mur_y = 0.f, mur_z = 0.f;
    float dt_mu = 0.f;

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
            return TAN_D * 2.f * float(M_PI) * F0 * EPS_R * float(EPS0);
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

    void set_ground_pec() {
        hPecEx.assign(N, 0);
        hPecEy.assign(N, 0);
        hPecEz.assign(N, 0);
        const int i0 = PAD, i1 = PAD + PIX;
        const int j0 = PAD, j1 = PAD + PIX;
        for (int j = j0; j <= j1; ++j)
            for (int i = i0; i < i1; ++i)
                hPecEx[idx(i, j, K_GND)] = 1;
        for (int j = j0; j < j1; ++j)
            for (int i = i0; i <= i1; ++i)
                hPecEy[idx(i, j, K_GND)] = 1;
    }

    void apply_metal_bitmap() {
        set_ground_pec();
        const int i0 = PAD, j0 = PAD;
        for (int q = 0; q < PIX; ++q) {
            for (int p = 0; p < PIX; ++p) {
                if (!metal[p + PIX * q]) continue;
                const int i = i0 + p;
                const int j = j0 + q;
                hPecEx[idx(i, j, K_PATCH)] = 1;
                hPecEx[idx(i, j + 1, K_PATCH)] = 1;
                hPecEy[idx(i, j, K_PATCH)] = 1;
                hPecEy[idx(i + 1, j, K_PATCH)] = 1;
            }
        }
    }

    float source(float t) const {
        const float t0 = 3.0f / FC;
        const float tau = 0.55f / FC;
        const float x = (t - t0) / tau;
        return std::exp(-x * x) * std::sin(2.f * float(M_PI) * F0 * (t - t0));
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
            std::fprintf(stderr, "[cuda] GPU 없음\n");
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
            std::fprintf(stderr, "[cuda] fdtd_kernels.cu 를 찾지 못했습니다.\n");
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
            !gf(&fAir, "k_accum_air"))
            return false;

        auto al = [&](DeviceBuf& b, size_t n) {
            if (!b.alloc(cu, n)) {
                std::fprintf(stderr, "[cuda] cuMemAlloc %zu 실패 (VRAM 부족?)\n", n);
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
            !al(energy, sizeof(float)) || !al(airE, sizeof(float) * NX * NY))
            return false;

        const float inv = std::sqrt(1.f / (DX * DX) + 1.f / (DY * DY) + 1.f / (DZ * DZ));
        dt = 0.99f / (float(C0) * inv);
        mur_x = (float(C0) * dt - DX) / (float(C0) * dt + DX);
        mur_y = (float(C0) * dt - DY) / (float(C0) * dt + DY);
        mur_z = (float(C0) * dt - DZ) / (float(C0) * dt + DZ);
        dt_mu = dt / float(MU0);
        metal.assign(PIX * PIX, 0);
        build_materials();
        cu->MemcpyHtoD(Caex.p, hCaex.data(), nf);
        cu->MemcpyHtoD(Cbex.p, hCbex.data(), nf);
        cu->MemcpyHtoD(Caey.p, hCaey.data(), nf);
        cu->MemcpyHtoD(Cbey.p, hCbey.data(), nf);
        cu->MemcpyHtoD(Caez.p, hCaez.data(), nf);
        cu->MemcpyHtoD(Cbez.p, hCbez.data(), nf);
        std::fprintf(stderr, "[cuda] ready  grid=%dx%dx%d  dt=%.4g\n", NX, NY, NZ, dt);
        return true;
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
        if (ctx) {
            cu->CtxDestroy(ctx);
            ctx = nullptr;
        }
        cu = nullptr;
    }

    int run(int nfreq, const double* freqs, double* s11_db, double* min_db, double* min_hz) {
        cu->CtxSetCurrent(ctx);
        apply_metal_bitmap();
        cu->MemsetD8(Ex.p, 0, Ex.bytes);
        cu->MemsetD8(Ey.p, 0, Ey.bytes);
        cu->MemsetD8(Ez.p, 0, Ez.bytes);
        cu->MemsetD8(Hx.p, 0, Hx.bytes);
        cu->MemsetD8(Hy.p, 0, Hy.bytes);
        cu->MemsetD8(Hz.p, 0, Hz.bytes);
        cu->MemcpyHtoD(pecEx.p, hPecEx.data(), N);
        cu->MemcpyHtoD(pecEy.p, hPecEy.data(), N);
        cu->MemcpyHtoD(pecEz.p, hPecEz.data(), N);
        cu->MemsetD8(airE.p, 0, airE.bytes);
        air_n = 0;

        unsigned bx = 8, by = 8, bz = 4;
        unsigned gx = (NX + bx - 1) / bx, gy = (NY + by - 1) / by, gz = (NZ + bz - 1) / bz;
        unsigned pec_threads = 256;
        unsigned pec_blocks = (N + pec_threads - 1) / pec_threads;
        unsigned mx = 16, my = 16;
        unsigned gmurj = (NY + mx - 1) / mx, gmurk = (NZ + my - 1) / my;
        unsigned gmuri = (NX + mx - 1) / mx;

        int nused = NSTEPS;
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

        for (int n = 0; n < NSTEPS; ++n) {
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

            if ((n & 7) == 0 && n > 200) {
                CUdeviceptr dAir = airE.p;
                unsigned agx = (NX + 15) / 16, agy = (NY + 15) / 16;
                void* aA[] = {&dEx, &dEy, &dEz, &dAir};
                launch(fAir, agx, agy, 1, 16, 16, 1, aA);
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
                if (n > 4000 && e < END_CRIT * emax) {
                    nused = n + 1;
                    break;
                }
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
        std::fprintf(stderr, "[voxel-cuda] nused=%d vmax=%.4g imax=%.4g emax=%.4g dt=%.4g corr=%.4g\n",
                     nused, vmax, imax, emax, dt, corr);

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
            const double magV = std::hypot(vr, vi);
            const double magVs = std::hypot(srcr, srci) + 1e-30;
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
        hAir.assign(NX * NY, 0);
        cu->MemcpyDtoH(hAir.data(), airE.p, sizeof(float) * NX * NY);
        const float inv = (air_n > 0) ? 1.f / float(air_n) : 1.f;
        for (int n = 0; n < NX * NY; ++n) hAir[n] = std::sqrt(hAir[n] * inv);
        return nused;
    }

    void copy_air(float* out) const {
        if (hAir.empty()) {
            std::fill(out, out + NX * NY, 0.f);
            return;
        }
        std::memcpy(out, hAir.data(), sizeof(float) * NX * NY);
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
