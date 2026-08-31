# 🎨 PixelArtSmith (픽셀아트 스미스)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Platform: Cross-Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey.svg)]()

> **PixelArtSmith** transforms AI-generated character sprite sheets (Stable Diffusion, Midjourney, ComfyUI, etc.) into **authentic, grid-perfect 1px retro pixel art** with zero mixels, semantic chroma-weighted palette snapping, non-leaking background matting, 2D Motion Matrix preservation ($4\times 4$ / $4\times 5$ walk cycles), standardized cell packaging ($32\times 32$, $48\times 48$, $64\times 64$), and automated quality auditing.

---

## 🌟 핵심 기능 및 아키텍처 (Key Features & Architecture)

```
[Raw SD Image (1024x1024)]
       │
       ▼ [1. Center-Subblock Downsampling] (Pitch: 8px -> 128x128 Core Grid, Zero-Bleed)
[Core Pixel Grid]
       │
       ▼ [2. Non-Leaking Background Matting] (4-Connected Floodfill from Perimeter)
[Segmented Transparent Grid]
       │
       ▼ [3. Chroma-Weighted Palette Quantization] (Snapper-16 / Snapper-13 / Retro Palettes)
[Clean Pixel Art Grid]
       │
       ▼ [4. 2D Matrix Energy Variance Balancer] (Auto-Detect 4x4 (16F) or 4x5 (20F) Motion Layout)
[Isolated Frames Matrix]
       │
       ▼ [5. Standardized Grid Packing] (Fixed 32x32 / Auto-Fit / Bottom-Center Grounding Anchor)
[Output Pixel Sprite Sheet + JSON Metadata + result.md Audit Report]
```

1. **True-Grid Center-Subblock 다운샘플링 (100% 믹셀 방지 & 원본 색상 보존)**:
   - **오토 피치 감지 (Auto-Pitch Detection)**: $8\text{px}$, $10\text{px}$, $12\text{px}$, $16\text{px}$ 등 AI가 생성한 가상 픽셀 블록 크기를 자동 감지.
   - **중앙 서브블록 샘플링 (Center $4\times 4$ Subblock)**: 블록 외곽의 안티앨리어싱 블러를 배제하고 중앙 순수 픽셀만 샘플링하여, 선명한 $1\text{px}$ 외곽선과 원본 RGB 채도를 $100\%$ 보존합니다.
2. **침범 없는 공간적 배경 제거 (Non-Leaking Spatial Matting)**:
   - 4방향 연결성(4-connected) 플러드필을 테두리에서 수행하여 캐릭터 내부의 흰색 의상/피부/머리카락을 파먹지 않고 외곽 배경만 완벽하게 투명화합니다.
3. **채도 가중 시맨틱 팔레트 양자화 (Chroma-Weighted Semantic Quantization)**:
   - 배경을 제외한 **순수 캐릭터 전경(Foreground)에 16색을 온전히 할당**.
   - 인간 시각이 민감한 피부색, 머리카락, 포인트 장식의 생동감을 보존하도록 채도 가중치($w_{\text{chroma}} = 2.0$) CIELAB 양자화를 적용합니다.
4. **2D 매트릭스 에너지 분산 밸런서 (2D Motion Matrix Isolator)**:
   - **$4\times 4$ (16프레임)** 및 **$4\times 5$ (20프레임)** 4방향 보행 시트를 자동 판별.
   - 프레임 간격이 맞닿아 있는 몬스터(오거, 좀비, 고블린 등)도 서브픽셀 부동소수점 반올림(`round(c * W / N)`)으로 오차 누적이나 잘림 없이 완벽 분리합니다.
5. **규격화된 그리드 패킹 및 바닥 접지 정렬 (Standardized Grid & Grounding)**:
   - **`fixed-32` ($32\times 32\text{px}$)**, **`fixed-48` ($48\times 48\text{px}$)**, **`fixed-64` ($64\times 64\text{px}$)** 규격 캔버스 패킹 지원.
   - **바닥 중앙 접지(Bottom-Center Ground Anchor)** 정렬로 게임 엔진 적용 시 상하 흔들림(Jitter)을 방지합니다.
6. **결정론적 품질 감사 리포트 (Deterministic Quality Audit)**:
   - 매 변환마다 코어 색상 보존율($\ge 98\%$), 배경 누수율($0.0\%$), 프레임 수($16\text{F}$ / $20\text{F}$), 매트릭스 밸런스 지수를 계산하여 `result.md` 마크다운 보고서를 자동 생성합니다.

---

## 🚀 빠른 시작 (Quick Start)

### 요구사항 (Prerequisites)
- Python 3.10, 3.11, 또는 3.12

### 1. 인터랙티브 GUI 스튜디오 실행
```bash
# macOS / Linux
./run.sh

# Windows (PowerShell)
.\run.ps1
```

### 2. 고속 배치 CLI 실행 (Headless CLI)
```bash
# 단일 이미지: 32x32 규격, Snapper-16 팔레트, 4배율 출력
./run.sh /path/to/character.png -c 32x32 -p snapper-16 -s 4 -o ./output

# 디렉토리 일괄 변환 (Batch Processing)
./run.sh /path/to/raw_images_dir/ -c 32x32 -p snapper-16 -s 4 -o ./output

# 프레임 낱장 분할 PNG 함께 추출 (--export-frames)
./run.sh /path/to/sprite.png -c 32x32 --export-frames -o ./output
```

---

## 📖 CLI 실행 인자 및 옵션 상세 (CLI Arguments & Options)

PixelArtSmith CLI는 단일 이미지 파일 및 디렉토리 일괄 처리를 모두 지원합니다.

```
사용법: run.sh [INPUT] [OPTIONS]
```

### 📌 필수 및 기본 인자 (Positional Arguments)

| 인자명 | 타입 | 설명 |
| :--- | :---: | :--- |
| **`input`** | `str` | 변환할 원본 이미지 파일 경로(`character.png`) 또는 이미지가 담긴 디렉토리 경로. |

---

### ⚙️ 옵션 플래그 상세 (Optional Arguments)

| 옵션 플래그 | 타입 | 기본값 | 설명 |
| :--- | :---: | :---: | :--- |
| **`-o`, `--output-dir`** | `str` | `.` (현재 폴더) | 결과물 스프라이트 시트 및 메타데이터가 저장될 출력 디렉토리 경로. |
| **`-c`, `--cell-size`** | `str` | `None` (auto) | **개별 스프라이트 프레임의 규격 크기**.<br>예: `32x32`, `32`, `48x48`, `64x64`.<br>지정 시 모든 프레임을 해당 캔버스 크기(바닥 중앙 정렬)로 패킹합니다. |
| **`-g`, `--grid-mode`** | `str` | `fixed-32` | **그리드 패킹 모드**:<br>• `fixed-32`: $32\times 32\text{px}$ 표준 규격 패킹<br>• `auto-fit`: 캐릭터 실루엣에 맞춘 가변 최적 크기<br>• `fixed-48`: $48\times 48\text{px}$ 패킹<br>• `fixed-64`: $64\times 64\text{px}$ 패킹<br>• `canvas`: 모션 분할 없는 1:1 통짜 일러스트 모드 |
| **`-P`, `--pitch`** | `int` | `8` (auto) | 가상 픽셀 블록의 피치 크기(raw 픽셀 단위). 생략 시 자동 감지. |
| **`-p`, `--palette`** | `str` | `snapper-16` | **색상 팔레트 프리셋** (아래 팔레트 목록 참조). |
| **`-k`, `--max-colors`** | `int` | `16` | 캐릭터 전경(Foreground)에 할당할 최대 색상 수 (기본: 16). |
| **`-s`, `--scale`** | `int` | `4` | **최종 출력 디스플레이 확대 배율** (최근방 이웃 보간):<br>• `1`: 네이티브 논리 픽셀 크기 ($128\times 128$ 등)<br>• `2`: 2배율 확대 ($256\times 256$ 등)<br>• `4`: 4배율 표준 확대 ($512\times 512$ 등)<br>• `8`: 8배율 고해상도 확대 ($1024\times 1024$ 등) |
| **`--export-1x`** | `bool` | `True` | **게임 엔진용 1배(1x) 원본 규격 스프라이트 시트 및 메타데이터를 `1x/` 하위 폴더에 동시 생성** (`--no-export-1x`로 비활성화 가능). |
| **`--export-gifs`** | `bool` | `True` | **방향/동작별 개별 애니메이션 GIF 및 전방향 통합 프리뷰 GIF 동시 생성** (`--no-export-gifs`로 비활성화 가능). |
| **`--gif-duration`** | `int` | `150` | 애니메이션 GIF 프레임당 노출 시간 (ms 단위, 기본: 150ms). |
| **`--export-frames`** | `bool` | `True` | **동작(모션)별 및 프레임별 1배(1x) 원본 규격 낱장 PNG 파일들을 하위 폴더(`_frames/`)에 동시 분할 저장** (`--no-export-frames`로 비활성화 가능). |
| **`--clean-orphans`** | `flag` | `False` | $1\text{px}$ 크기의 고립된 단일 노이즈 픽셀 자동 제거. |
| **`--no-bg-remove`** | `flag` | `False` | 배경 투명화 제거를 건너뛰고 원본 배경색을 그대로 유지. |
| **`--report-name`** | `str` | `result.md` | 품질 감사 마크다운 리포트 파일명. |

---

## 🎨 지원 색상 팔레트 프리셋 (Color Palettes)

| 팔레트 키값 | 색상 수 | 특징 및 용도 |
| :--- | :---: | :--- |
| **`snapper-16`** | 16 | **[추천 기본값]** 고채도 전경 적응형 16색 (상용 스내퍼 도구 패리티) |
| **`snapper-13`** | 13 | 레트로 콘솔 스타일의 콤팩트 13색 |
| **`endesga-32`** | 32 | EdG32 명품 판타지 RPG 마스터 팔레트 |
| **`endesga-64`** | 64 | 풍부한 음영의 고디테일 64색 팔레트 |
| **`nes-54`** | 54 | 닌텐도 패미컴(NES) 하드웨어 마스터 팔레트 |
| **`snes-classic`**| 32 | 슈퍼 패미컴(SNES) 16비트 클래식 램프 |
| **`pico-8`** | 16 | 피코-8 판타지 콘솔 전용 16색 |
| **`gameboy-classic`**| 4 | 오리지널 게임보이(DMG-01) 4단계 올리브 그린 |
| **`gameboy-color`** | 32 | 게임보이 컬러(GBC) 마스터 32색 |
| **`c64-commodore`** | 16 | 코모도어 64 레트로 팔레트 |
| **`none`** | 원본 | 팔레트 양자화 없이 논리 그리드 다운샘플링만 수행 |

---

## 📦 생성 결과물 구조 (Output Artifacts)

변환 완료 시 지정된 출력 디렉토리에 중복 없이 체계화되어 생성됩니다:

```
output_dir/
├── 1x/                         # 🎮 게임 엔진 직접 임포트용 1배(1x) 원본 규격 폴더
│   ├── hero_pixel_sheet.png    # 32x32 1배(1x) 원본 규격 시트 (128x128 / 136x136)
│   ├── hero_metadata.json      # 1배(1x) 기준 메타데이터
│   ├── hero_frames/            # 1배(1x) 낱장 프레임 (모션별 서브폴더 + 플랫)
│   │   ├── motion_00_down/
│   │   │   ├── frame_00.png
│   │   │   └── ...
│   │   └── motion_00_down_frame_00.png
│   └── hero_gifs/              # 1배(1x) 애니메이션 GIF
│       ├── hero_all_motions.gif    # 1배(1x) 통합 프리뷰 GIF
│       ├── hero_motion_00_down.gif # 1배(1x) 개별 동작 GIF
│       └── ...
├── 4x/                         # 🖼️ 고해상도 4배율(4x) 디스플레이 전용 폴더
│   ├── hero_pixel_sheet.png    # 4배율(4x) 고화질 시트 (512x512 / 544x544)
│   ├── hero_metadata.json      # 4배율(4x) 기준 메타데이터
│   ├── hero_frames/            # 4배율(4x) 낱장 프레임 (모션별 서브폴더 + 플랫)
│   └── hero_gifs/              # 4배율(4x) 애니메이션 GIF
│       ├── hero_all_motions.gif    # 4배율(4x) 통합 프리뷰 GIF
│       ├── hero_motion_00_down.gif # 4배율(4x) 개별 동작 GIF
│       └── ...
└── result.md                   # 📊 100% 코어 보존율 및 품질 감사 종합 보고서
```

---

## 🏗️ 독립 실행형 바이너리 빌드 (PyInstaller Standalone Build)

모노레포 루트의 `build_all.sh` 또는 자체 빌드 스크립트를 사용하여 파이썬 설치 없이 구동 가능한 독립 실행형 바이너리를 빌드할 수 있습니다:

```bash
# macOS / Linux (build/pixel-art-smith/ 생성)
./build.sh

# Windows (PowerShell)
.\build.ps1
```

---

## 📄 라이선스 (License)

이 프로젝트는 **GNU General Public License v3.0 (GPL-3.0)** 하에 배포됩니다. 자세한 내용은 [`LICENSE`](./LICENSE) 파일을 참조하십시오.
