import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import tempfile
import time
import traceback

# ==========================================
# 0. 경로 설정 (사용자 환경에 맞게 유지)
# ==========================================
openems_dir = r'C:\Users\set\Downloads\jung\openEMS_v0.0.36\openEMS'
bin_dir = os.path.join(openems_dir, 'bin')

# Windows DLL 경로 및 환경 변수 등록
os.add_dll_directory(openems_dir)
if os.path.exists(bin_dir):
    os.add_dll_directory(bin_dir)

if bin_dir not in os.environ['PATH']:
    os.environ['PATH'] = bin_dir + os.pathsep + os.environ['PATH']
if openems_dir not in os.environ['PATH']:
    os.environ['PATH'] = openems_dir + os.pathsep + os.environ['PATH']

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import *

import contextlib
import sys

@contextlib.contextmanager
def redirect_c_stdout_stderr(log_file):
    """Python은 물론 C++ 엔진(openEMS)의 터미널 출력까지 모두 지정된 파일로 강제 리다이렉트합니다."""
    try:
        original_stdout_fd = sys.stdout.fileno()
        original_stderr_fd = sys.stderr.fileno()
    except Exception:
        yield
        return

    saved_stdout_fd = os.dup(original_stdout_fd)
    saved_stderr_fd = os.dup(original_stderr_fd)

    sys.stdout.flush()
    sys.stderr.flush()

    # log_file의 OS 파일 디스크립터를 터미널 출력 경로에 덮어씌움
    os.dup2(log_file.fileno(), original_stdout_fd)
    os.dup2(log_file.fileno(), original_stderr_fd)

    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        # 원래 터미널 출력으로 원상복구
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.dup2(saved_stderr_fd, original_stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)

@contextlib.contextmanager
def suppress_stdout_stderr():
    """openEMS C++ 엔진이 뱉어내는 저수준 터미널 출력을 완벽히 차단합니다."""
    try:
        original_stdout_fd = sys.stdout.fileno()
        original_stderr_fd = sys.stderr.fileno()
    except Exception:
        yield
        return

    save_stdout = os.dup(original_stdout_fd)
    save_stderr = os.dup(original_stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)

    sys.stdout.flush()
    sys.stderr.flush()

    os.dup2(devnull, original_stdout_fd)
    os.dup2(devnull, original_stderr_fd)
    os.close(devnull)

    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(save_stdout, original_stdout_fd)
        os.dup2(save_stderr, original_stderr_fd)
        os.close(save_stdout)
        os.close(save_stderr)

# ==========================================
# 4. 파라뷰(ParaView) 디버깅 및 시각화 전용 함수
# ==========================================
def simulate_and_dump_paraview(genome, output_dir="paraview_debug_output"):
    """
    고유한 폴더명을 사용하므로 삭제 없이 안전하게 디렉토리를 생성하고 로그를 기록합니다.
    """
    original_cwd = os.getcwd()
    output_dir = os.path.realpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    sim_data_dir = os.path.join(output_dir, "sim_data")
    log_path = os.path.join(output_dir, "log.txt")

    with open(log_path, "w", encoding="utf-8") as log_file:
        try:
            log_file.write("=== ParaView 디버깅 시뮬레이션 시작 ===\n")
            f0 = 7.25e9
            fc = 3.5e9

            FDTD = openEMS(NrTS=100000, EndCriteria=1e-3)
            CSX = ContinuousStructure()
            FDTD.SetCSX(CSX)

            mesh = CSX.GetGrid()
            mesh.SetDeltaUnit(1e-3)

            # 1. 고정되어야 하는 핵심 좌표들(Key lines)만 먼저 입력
            mesh_xy = np.arange(-25.0, 26.0, 1.0)
            mesh.AddLine('x', mesh_xy)
            mesh.AddLine('y', mesh_xy)

            # NF2FF 박스와 PML 영역을 위한 기준선 (X, Y)
            mesh.AddLine('x', [-50.0, -35.0, 35.0, 50.0])
            mesh.AddLine('y', [-50.0, -35.0, 35.0, 50.0])

            # Z축 기준선 (FR4 내부를 0.5mm 간격으로 분할)
            metal_thickness = 0.035
            mesh.AddLine('z', np.arange(0, 1.51, 0.5))
            mesh.AddLine('z', [-metal_thickness, 1.5 + metal_thickness])

            # NF2FF 박스와 PML 영역을 위한 기준선 (Z)
            mesh.AddLine('z', [-40.0, -5.0, 15.0, 40.0])

            # 2. 💡 [핵심] SmoothMeshLines를 통한 부드러운 메쉬 생성
            # 최대 메쉬 크기(max_res)를 지정하면, 촘촘한 곳에서부터 1.3배씩 서서히 커지도록 빈 공간을 채워줍니다.
            max_res = 10.0  # 공기 중 최대 메쉬 크기를 10mm로 제한 (파장 대비 충분히 작은 값)
            mesh.SmoothMeshLines('x', max_res, ratio=1.3)
            mesh.SmoothMeshLines('y', max_res, ratio=1.3)
            mesh.SmoothMeshLines('z', max_res, ratio=1.3)

            FR4 = CSX.AddMaterial('FR4', epsilon=4.4)
            FR4.AddBox([-25.0, -25.0, 0], [25.0, 25.0, 1.5],priority = 0)

            metal = CSX.AddMetal('Copper')
            metal.AddBox([-25.0, -25.0, -metal_thickness], [25.0, 25.0, 0], priority=10)
            patch_count = 1

            for i in range(51):
                for j in range(51):
                    if genome[i, j] == 1:
                        x1 = round(-25.0 + i, 3)
                        x2 = round(x1 + 1.0, 3)
                        y1 = round(-25.0 + j, 3)
                        y2 = round(y1 + 1.0, 3)
                        metal.AddBox([x1, y1, 1.5], [x2, y2, 1.5 + metal_thickness], priority=10)
                        patch_count += 1


            log_file.write(f"메탈 픽셀 개수: {patch_count}개\n")

            FDTD.AddEdges2Grid(dirs='xy', properties=metal)
            port = FDTD.AddLumpedPort(1, 50, [-0.5, -10.5, 0.0], [0.5, -9.5, 1.5], 'z', 1.0, priority=5, edges2grid='xy')
            e_dump = CSX.AddDump('Et', dump_type=0, file_type=0, sub_sampling=[2, 2, 2])
            e_dump.AddBox([-50.0, -50.0, -5.0], [50.0, 50.0, 20.0])

            FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])
            FDTD.SetGaussExcite(f0, fc)

            log_file.write("FDTD.Run 실행 시작...\n")
            log_file.flush()

            with redirect_c_stdout_stderr(log_file):
                FDTD.Run(sim_data_dir, cleanup=False, verbose=4)

            log_file.write("\n시뮬레이션 정상 완료.\n")
            print(f"  📁 [로그 저장 완료] {output_dir}/log.txt")

        except AssertionError:
            # 💡 [핵심 추가] 엔진 폭주로 인한 AssertionError는 정상적인(?) 불량 반응이므로 부드럽게 넘깁니다.
            warn_msg = "\n⚠️ [시뮬레이션 발산 경고] 안테나 구조가 불안정하여 전자기 에너지가 폭주했습니다. 엔진이 강제 종료되었습니다.\n"
            log_file.write(warn_msg)
            log_file.flush()
            print(f"  ⚠️ [발산 경고] {os.path.basename(output_dir)} 구조 불안정으로 연산 중단 (ParaView용 XML은 정상 저장됨)")


        except Exception as e:

            # 💡 [핵심] 텅 빈 에러 메시지 대신, 에러의 종류와 발생 위치를 싹 다 긁어옵니다.

            err_type = type(e).__name__

            full_traceback = traceback.format_exc()

            err_msg = f"❌ [오류 발생]: {err_type} - {str(e)}\n"

            err_msg += "=== 📜 상세 파이썬 트레이스백 ===\n"

            err_msg += full_traceback

            log_file.write(err_msg)

            log_file.flush()

            print(f"  ❌ [덤프 에러 발생] ({output_dir}): {err_type} (자세한 건 log.txt 확인)")

        finally:
            # 💡 [방어 코드 2] openEMS가 경로를 바꿨더라도 무조건 원래 위치로 강제 복귀!
            os.chdir(original_cwd)

# ==========================================
# 1. 단일 개체 시뮬레이션 함수 (독립 프로세스 구동)
# ==========================================
def simulate_individual(args):
    """
    args: (genome, sim_id) 튜플 형태로 받도록 수정 (Pool 연동용)
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    genome, sim_id = args
    base_cwd = os.getcwd()
    local_temp_base = os.path.realpath(f"C:\\temp_openEMS_GA_worker_{sim_id}")
    os.makedirs(local_temp_base, exist_ok=True)

    for filename in os.listdir(local_temp_base):
        file_path = os.path.join(local_temp_base, filename)
        try:
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.unlink(file_path)
        except Exception:
            pass

    try:
        f0 = 7.25e9
        fc = 3.5e9

        # 타임스텝 한계를 15000으로 조여서 불량 개체 빠른 차단
        FDTD = openEMS(NrTS=100000, EndCriteria=1e-3)
        CSX = ContinuousStructure()
        FDTD.SetCSX(CSX)

        mesh = CSX.GetGrid()
        mesh.SetDeltaUnit(1e-3)

        # 1. 고정되어야 하는 핵심 좌표들(Key lines)만 먼저 입력
        mesh_xy = np.arange(-25.0, 26.0, 1.0)
        mesh.AddLine('x', mesh_xy)
        mesh.AddLine('y', mesh_xy)

        # NF2FF 박스와 PML 영역을 위한 기준선 (X, Y)
        mesh.AddLine('x', [-50.0, -35.0, 35.0, 50.0])
        mesh.AddLine('y', [-50.0, -35.0, 35.0, 50.0])

        # Z축 기준선 (FR4 내부를 0.5mm 간격으로 분할)
        metal_thickness = 0.035
        mesh.AddLine('z', np.arange(0, 1.51, 0.5))
        mesh.AddLine('z', [-metal_thickness, 1.5 + metal_thickness])

        # NF2FF 박스와 PML 영역을 위한 기준선 (Z)
        mesh.AddLine('z', [-40.0, -5.0, 15.0, 40.0])

        # 2. 💡 [핵심] SmoothMeshLines를 통한 부드러운 메쉬 생성
        # 최대 메쉬 크기(max_res)를 지정하면, 촘촘한 곳에서부터 1.3배씩 서서히 커지도록 빈 공간을 채워줍니다.
        max_res = 10.0  # 공기 중 최대 메쉬 크기를 10mm로 제한 (파장 대비 충분히 작은 값)
        mesh.SmoothMeshLines('x', max_res, ratio=1.3)
        mesh.SmoothMeshLines('y', max_res, ratio=1.3)
        mesh.SmoothMeshLines('z', max_res, ratio=1.3)

        FR4 = CSX.AddMaterial('FR4', epsilon=4.4)
        FR4.AddBox([-25.0, -25.0, 0], [25.0, 25.0, 1.5], priority=0)

        metal = CSX.AddMetal('Copper')
        metal.AddBox([-25.0, -25.0, -metal_thickness], [25.0, 25.0, 0], priority=10)

        for i in range(51):
            for j in range(51):
                if genome[i, j] == 1:
                    x1 = round(-25.0 + i, 3)
                    x2 = round(x1 + 1.0, 3)
                    y1 = round(-25.0 + j, 3)
                    y2 = round(y1 + 1.0, 3)
                    metal.AddBox([x1, y1, 1.5], [x2, y2, 1.5 + metal_thickness], priority=10)

        FDTD.AddEdges2Grid(dirs='xy', properties=metal)
        port = FDTD.AddLumpedPort(1, 50, [-0.5, -10.5, 0.0], [0.5, -9.5, 1.5], 'z', 1.0, priority=5, edges2grid='xy')

        FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])
        FDTD.SetGaussExcite(f0, fc)

        calc_nf2ff = FDTD.CreateNF2FFBox(start=[-35.0, -35.0, -5.0], stop=[35.0, 35.0, 15.0])

        with suppress_stdout_stderr():
            FDTD.Run(local_temp_base, cleanup=False, verbose=0)

        freqs = np.linspace(6.5e9, 8.0e9, 101)
        port.CalcPort(local_temp_base, freqs)
        S11 = port.uf_ref / port.uf_inc
        S11_dB = 20 * np.log10(np.abs(S11) + 1e-12)
        max_S11_in_band = np.max(S11_dB)

        s11_variance = np.var(S11_dB)

        theta = np.arange(-180.0, 180.0, 10.0)
        phi = np.arange(0.0, 360.0, 10.0)
        nf2ff_res = calc_nf2ff.CalcNF2FF(local_temp_base, freqs, theta, phi)

        peak_directivity = np.max(nf2ff_res.Dmax)

        if max_S11_in_band > -1.0:
            # -1.0dB 이상: 에너지가 아예 나가지 않는 구조 (-0.06dB 같은 녀석들) -> 즉시 도태
            score = -999.0
        elif max_S11_in_band > -10.0:
            # 에너지가 나가긴 하지만 매칭이 덜 된 상태 -> 기본 페널티 + 잔물결 페널티 복합 부여
            base_penalty = (max_S11_in_band - (-10.0)) * 15.0
            ringing_penalty = s11_variance * 5.0
            score = peak_directivity - base_penalty - ringing_penalty
        else:
            # -10dB 이하 합격선: 합격했더라도 잔물결(Variance)이 심하면 점수를 깎아 최적화 유도
            ringing_penalty = s11_variance * 2.0
            score = peak_directivity - ringing_penalty

        return (score, max_S11_in_band, peak_directivity, S11_dB, freqs)

    except Exception as e:
        print(f"  [오류 발생] {str(e)}")
        return (-999, 0, 0, None, None)
    finally:
        shutil.rmtree(local_temp_base, ignore_errors=True)


# ==========================================
# 2. JPG 시각화 저장 함수
# ==========================================
def save_generation_image(genome, S11, freqs, gen, score, peak_dir, max_s11):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 왼쪽: 안테나 형상 (픽셀)
    ax1.imshow(genome.T, cmap='Greys', interpolation='nearest', origin='lower', extent=[-50, 50, -50, 50])
    # 포트 위치 표시를 (0,0)으로 지정
    ax1.set_title(f"Gen {gen} Best Topology\n(Black = Copper, White = Air)", fontsize=14)
    # 그리드 추가
    ax1.set_xticks(np.arange(-50.5, 51.5, 1), minor=True)
    ax1.set_yticks(np.arange(-50.5, 51.5, 1), minor=True)
    ax1.grid(True, which='minor', color='black', linestyle='-', linewidth=0.5)
    # 급전점 표시 (빨간 X)
    ax1.plot(0, -10, 'rx', markersize=12, markeredgewidth=3, label='Feed Point (0,-10)')
    ax1.legend(loc='upper right')

    # 오른쪽: S11 그래프
    ax2.plot(freqs / 1e9, S11, 'b-', linewidth=2.5)
    ax2.axhline(-10, color='r', linestyle='--', linewidth=2, label='-10 dB Threshold')
    ax2.set_title(f"Score: {score:.2f} | Peak Dir: {peak_dir:.2f} dBi | Max S11: {max_s11:.2f} dB", fontsize=14)
    ax2.set_xlabel("Frequency (GHz)", fontsize=12)
    ax2.set_ylabel("S11 (dB)", fontsize=12)
    ax2.set_ylim([-30, 0])
    ax2.grid(True, linestyle=':')
    ax2.legend()

    plt.tight_layout()
    # JPG로 저장
    filename = f"Generation_{gen:03d}_Best.jpg"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  📸 저장 완료: {filename}")


# ==========================================
# 3. 유전 알고리즘 (GA) 메인 엔진
# ==========================================

#군론 응용
def D4_group_theory(size=51):
    """
    군론(Group Theory)의 이면군(Dihedral Group, D4) 및 부분군 대칭성을
    적용하여 안테나 패치(Genome)를 생성하는 함수입니다.
    - size: 정중앙 대칭을 위해 홀수 크기 권장 (기본값: 51)
    """
    assert size % 2 == 1, "정중앙 대칭과 포트 일치를 위해 size는 홀수여야 합니다."

    center = size // 2
    half = center + 1

    # 1/4 크기의 기본 코어 조각 무작위 생성
    core = np.random.randint(0, 2, size=(half, half))

    # 정중앙(급전점 위치)은 항상 구리(1)로 고정
    core[0, 0] = 1

    ind = np.zeros((size, size), dtype=int)

    # D4 군(Dihedral Group)에 속하는 대칭 모드 중 하나를 무작위 선택
    # 0: 완벽한 D4 대칭 (모든 반사축 + 90도 회전 성분)
    # 1: 직교 반사 대칭 (상하 / 좌우 대칭)
    # 2: 대각선 반사 대칭
    symmetry_mode = np.random.choice([0, 1, 2])

    for i in range(half):
        for j in range(half):
            val = core[i, j]

            if symmetry_mode == 0:
                # D4 전체 대칭 (4방향 반사 및 대각선 대칭 반영)
                ind[center + i, center + j] = val
                ind[center - i, center + j] = val
                ind[center + i, center - j] = val
                ind[center - i, center - j] = val

                ind[center + j, center + i] = val
                ind[center - j, center + i] = val
                ind[center + j, center - i] = val
                ind[center - j, center - i] = val

            elif symmetry_mode == 1:
                # 상하 / 좌우 반사 대칭 (Klein 4-group 성질)
                ind[center + i, center + j] = val
                ind[center - i, center + j] = val
                ind[center + i, center - j] = val
                ind[center - i, center - j] = val

            else:
                # 대각선 반사 대칭
                ind[center + i, center + j] = val
                ind[center - i, center - j] = val
                ind[center + j, center + i] = val
                ind[center - j, center - i] = val

    # 정중앙(급전점 픽셀)은 예외 없이 확실하게 1로 보장
    ind[center-1:center+1, center-1:center+1] = 1
    ind[center - 1:center + 1, 14:center] = 1

    return ind


def create_individual():
    # 101x101 크기로 변경하고, 정중앙 인덱스 [50, 50]을 구리로 고정
    return D4_group_theory(size=51)


"""
def create_individual():
    # 101x101 크기로 변경하고, 정중앙 인덱스 [50, 50]을 구리로 고정
    ind = np.random.randint(0, 2, size=(51, 51))
    ind[25, 25] = 1
    return ind
"""
"""
def create_individual():
    ind = np.zeros((51, 51), dtype=int)

    # 예시 패치 크기 조절 (중앙 25,25를 기준으로 10x10 크기 패치 생성)
    start_x = 25 - 5
    end_x = 25 + 5
    start_y = 25 - 5
    end_y = 25 + 5

    ind[start_x:end_x, start_y:end_y] = 1
    ind[24:26, 24:26] = 1  # 정중앙 급전점 고정
    return ind
"""

def run_ga():
    POP_SIZE = 3
    GENERATIONS = 30
    MUTATION_RATE = 0.05

    print(f"🚀 GA 최적화 시작! (Pop Size: {POP_SIZE}, Gens: {GENERATIONS})")
    BASE_DIR = r"C:\Temp\antenna_sim_output"

    population = [create_individual() for _ in range(POP_SIZE)]

    # 사용할 CPU 코어 수 설정
    num_workers = min(POP_SIZE, mp.cpu_count())
    print(f"⚡ 멀티스레딩 풀(Pool) 가동: {num_workers}개 코어 동시 병렬 연산")

    best_history = []
    # 💡 [권장] Pool을 세대 루프 밖에서 단 한번만 열고 30세대 동안 쭉 재사용합니다.
    with mp.Pool(processes=num_workers) as pool:
        for gen in range(1, GENERATIONS + 1):
            print(f"\n[{gen}/{GENERATIONS} 세대] 시뮬레이션 평가 중...")

            fitness_scores = []
            best_genome = None
            best_s11_array = None
            best_freqs = None
            best_metrics = (-999, 0, 0)

            tasks = [(population[i], i) for i in range(POP_SIZE)]
            results = pool.map(simulate_individual, tasks)

            # 결과 수집 및 필터링
            for i, (score, s11, p_dir, s11_arr, frq_arr) in enumerate(results):
                genome = population[i]

                # 방어선: 함정/고장 개체 차단 (세대와 인덱스로 고유 폴더명 지정)
                if score == -999 or s11 > -1.0 or p_dir > 20.0:
                    score = -999
                    # 💡 BASE_DIR을 합쳐서 무조건 최상위 폴더 아래에 만들어지도록 강제
                    folder_name = os.path.join(BASE_DIR, f"faulty_gen{gen:02d}_ind{i + 1:02d}")
                    print(f"  🚨 함정 개체 차단! (개체 {i + 1} | S11: {s11:.2f}dB, Dir: {p_dir:.2f}dBi) -> 덤프: {folder_name}")
                    #simulate_and_dump_paraview(genome, output_dir=folder_name)

                fitness_scores.append(score)

                if score > best_metrics[0]:
                    best_metrics = (score, s11, p_dir)
                    best_genome = genome
                    best_s11_array = s11_arr
                    best_freqs = frq_arr

                print(f"  👉 개체 {i + 1} 완료 | Score: {score:.2f} (S11: {s11:.2f}dB, Dir: {p_dir:.2f}dBi)")

            # 세대 최고 개체 이미지 저장
            if best_genome is not None and best_s11_array is not None:
                # 💡 이미지 저장도 CWD 꼬임 방지를 위해 강제 복귀 한 번 더 수행
                os.chdir(BASE_DIR)

                save_generation_image(
                    best_genome, best_s11_array, best_freqs,
                    gen, best_metrics[0], best_metrics[2], best_metrics[1]
                )

                score_str = f"{best_metrics[0]:.1f}".replace('.', '_')
                # 💡 우수 개체 덤프 역시 BASE_DIR 기준 절대 경로로!
                folder_name = os.path.join(BASE_DIR, f"best_gen{gen:02d}_score_{score_str}")
                print(f"  ⭐ 우수 개체 갱신! 파라뷰 덤프: {folder_name}")
                simulate_and_dump_paraview(best_genome, output_dir=folder_name)

            # 다음 세대 생성 로직 (토너먼트 선택, 교차, 돌연변이)
            new_population = []
            for _ in range(POP_SIZE):
                idx1, idx2, idx3 = np.random.choice(POP_SIZE, 3, replace=False)
                parent1_idx = max([(fitness_scores[i], i) for i in [idx1, idx2, idx3]])[1]

                idx1, idx2, idx3 = np.random.choice(POP_SIZE, 3, replace=False)
                parent2_idx = max([(fitness_scores[i], i) for i in [idx1, idx2, idx3]])[1]

                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]

                mask = np.random.randint(0, 2, size=(51, 51))
                child = np.where(mask == 1, parent1, parent2)

                mutation_mask = np.random.rand(51, 51) < MUTATION_RATE
                child = np.where(mutation_mask, 1 - child, child)

                child[25, 25] = 1

                new_population.append(child)

            population = new_population

    print("\n🎉 모든 진화 과정이 완료되었습니다!")


if __name__ == '__main__':
    # Windows에서 multiprocessing을 사용할 때 필수 방어 코드
    mp.freeze_support()
    run_ga()