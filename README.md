# MVP-gsplat Optimization (CUDA Enhanced)

이 저장소는 연세대학교 V-Lab에서 진행된 연구의 일환으로, [gsplat](https://github.com/gsplat-org/gsplat) 라이브러리의 CUDA 커널을 수정하여 렌더링 성능과 메모리 효율을 극대화한 최적화 버전입니다.

## 🚀 주요 최적화 특징 (Key Optimizations)

기존 MVP 모델이 PyTorch 상에서 Opacity의 SH(Spherical Harmonics)를 별도로 계산하던 방식을 CUDA 커널 내부로 통합(Kernel Fusion)하여 병목 현상을 제거했습니다.

### 1. CUDA Kernel Fusion
* **내용**: `fully_fused_projection` 커널 내에서 Opacity SH 연산을 직접 수행하도록 수정했습니다.
* **이점**: Opacity가 임계값(`ALPHA_THRESHOLD`) 이하인 가우시안을 사전에 필터링하여 불필요한 연산을 줄이고 조기 종료(Early Exit)를 유도합니다.

### 2. GPU 메모리 최적화 (Memory Efficiency)
* **내용**: PyTorch Tensor 할당 없이 커널 내에서 Opacity를 직접 처리하여 불필요한 메모리 점유를 방지했습니다.
* **결과**: GPU 메모리 사용량을 **64%에서 40%로 약 24%p 감소**시켰습니다.

### 3. 청크 렌더링 (Chunk Rendering)
* **내용**: View별로 반복되던 렌더링 루프를 Chunk 단위로 묶어 GPU 자원 활용도를 높였습니다. 
* **결과**: Chunk Size 32 적용 시, 기존 대비 학습 속도가 **최대 15% 향상**되었습니다.

---

## 📊 성능 분석 (Benchmark Results)

가용 서버 자원 제약 하에 단일 씬(Single Scene) 학습을 통해 측정한 결과입니다.

| 지표 (Metrics) | 원본 (Official) | 최적화 (Optimal) | 개선율 (Improvement) |
| :--- | :---: | :---: | :---: |
| **GPU Memory 할당량** | 64% | **40%** | **24%p 감소** |
| **Iter 당 평균 시간** | 1.88s | **1.62s** | **약 13% 단축** |
| **총 학습 시간 (Train)** | 02:44:10 | **02:19:13** | **약 15% 빨라짐** |
| **추론 시간 (Inference)** | 1.79min | **1.74min** | **약 2% 빨라짐** |

> **Verification**: Opacity Loss가 꾸준히 감소하는 그래프를 통해 순전파(Forward) 및 역전파(Backward) 과정이 수학적으로 정확하게 구현되었음을 검증했습니다.

---

## 🛠 수정 및 추가된 주요 파일
* `gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu`: 전방/후방 호출 커널 최적화
* `gsplat/cuda/include/Utils.cuh`: `spherical_harmonics_opacity` 연산 및 `vjp` 함수 구현
* `gsplat/rendering.py`: 최적화된 커널 인터페이스 적용
* `MVP/model.py`: 커널 최적화에 따른 모델 구조 수정

---

## 🔗 원본 출처 및 저작권 (Citations)

* **Original Library**: [gsplat-org/gsplat](https://github.com/gsplat-org/gsplat)
* **Optimization Author**: 조형빈 (Hyeongbhin Cho)
* **Affiliation**: V-Lab, Yonsei University
* **Repository**: [Hyeongbhin-Cho/mvp_optimization](https://github.com/Hyeongbhin-Cho/mvp_optimization.git)

본 프로젝트는 Apache-2.0 라이선스를 따르는 gsplat을 기반으로 수정되었습니다.

---
# MVP-gsplat Optimization (CUDA Enhanced)

As part of the research conducted at Yonsei University's V-Lab, this repository provides an optimized version of the [gsplat](https://github.com/gsplat-org/gsplat) library by modifying CUDA kernels to maximize rendering performance and memory efficiency.

## 🚀 Key Optimizations

Bottlenecks were eliminated by integrating the Opacity Spherical Harmonics (SH) calculation—previously performed separately in PyTorch within the original MVP model—directly into the CUDA kernel (Kernel Fusion).

### 1. CUDA Kernel Fusion
* **Details**: Modified the `fully_fused_projection` kernel to perform Opacity SH operations directly.
* **Benefits**: Pre-filters Gaussians with opacity below the threshold (`ALPHA_THRESHOLD`) to reduce unnecessary computations and induce early exits.

### 2. GPU Memory Efficiency
* **Details**: Optimized memory usage by processing opacity within the kernel instead of allocating PyTorch Tensors.
* **Results**: Reduced GPU memory allocation from **64% to 40% (a 24%p decrease)**.

### 3. Chunk Rendering
* **Details**: Enhanced GPU resource utilization by grouping rendering loops, which previously processed views individually, into chunks.
* **Results**: Achieved up to a **15% increase** in training speed by applying a Chunk Size of 32.

---

## 📊 Benchmark Results

Results measured through single-scene training under server resource constraints.

| Metrics | Official (Original) | Optimal (Optimized) | Improvement |
| :--- | :---: | :---: | :---: |
| **GPU Memory Allocation** | 64% | **40%** | **24%p Decrease** |
| **Avg. Time per Iteration** | 1.88s | **1.62s** | **~13% Reduction** |
| **Total Training Time** | 02:44:10 | **02:19:13** | **~15% Faster** |
| **Inference Time** | 1.79min | **1.74min** | **~2% Faster** |

> **Verification**: The steady decrease in Opacity Loss confirms that the Forward and Backward passes are mathematically implemented correctly.

---

## 🛠 Modified & Added Key Files
* `gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu`: Optimized forward/backward call kernels.
* `gsplat/cuda/include/Utils.cuh`: Implemented `spherical_harmonics_opacity` operations and `vjp` functions.
* `gsplat/rendering.py`: Applied optimized kernel interfaces.
* `MVP/model.py`: Modified model structure to accommodate kernel optimizations.

---

## 🔗 Citations & Copyright

* **Original Library**: [gsplat-org/gsplat](https://github.com/gsplat-org/gsplat)
* **Optimization Author**: Hyeongbhin Cho
* **Affiliation**: V-Lab, Yonsei University
* **Repository**: [Hyeongbhin-Cho/mvp_optimization](https://github.com/Hyeongbhin-Cho/mvp_optimization.git)

This project was modified based on gsplat under the Apache-2.0 License.