# DeepSeek-V4-Flash-0731 部署总结（2× RTX PRO 6000 SM120）

> **English summary**: Deploying **DeepSeek-V4-Flash-0731** (304B, FP8+NVFP4) on **2× NVIDIA RTX PRO 6000 Blackwell (SM120)** with **vLLM PR #41834 (jasl fork) + DSpark speculative decoding**, in a **no-overseas-network environment**. Final throughput: **~200-227 tok/s single stream** (vs 17-45 tok/s without speculation). This repo documents the complete process including all mirror sources used to work around restricted network access.

> 部署完成日期：2026-08-02/03
> 最终状态：**vLLM 0.26.0.post1+cu132 (SM120 fork) + DSpark 运行中，~200-227 tok/s**

## 1. 硬件与环境

| 项 | 值 |
|---|---|
| GPU | 2× NVIDIA RTX PRO 6000 Blackwell Workstation (SM120, compute capability 12.0)，各 96 GiB，无 NVLink |
| CPU/内存 | 32 核 / 251 GiB |
| 磁盘 | 6.6 TB 可用（NVMe）|
| OS | Ubuntu 22.04.2 LTS |
| 驱动 | 595.58.03 |
| CUDA | 13.2（/usr/local/cuda）|
| 网络 | **无海外网络**（github.com / Docker Hub / pytorch.org 等均不可达）|

## 2. 最终运行配置

```bash
# vLLM DSpark 服务（port 8000）
cd ~ && VIRTUAL_ENV=~/vllm-venv NCCL_P2P_DISABLE=1 CUDA_HOME=/usr/local/cuda \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  nohup ~/vllm-venv/bin/vllm serve ~/models/DeepSeek-V4-Flash-0731 \
  --tensor-parallel-size 2 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code \
  --port 8000 > ~/vllm_server.log 2>&1 &
```

**关键环境变量**：
- `NCCL_P2P_DISABLE=1`：Blackwell PCIe P2P allreduce 死锁规避
- `FLASHINFER_DISABLE_VERSION_CHECK=1`：flashinfer-python 0.6.16 vs cubin 版本号不匹配规避
- `CUDA_HOME=/usr/local/cuda`：TileLang/MHC-prenorm JIT 需要 nvcc

## 3. 测速结果（DSpark K5, temp=0）

| 场景 | tok/s |
|---|---|
| 单流散文 200 tok | 198-227（均值 ~215）|
| 代码生成 200 tok | 202-211 |
| warmup 后峰值 | 303 |

**DSpark 实测指标**（服务日志）：Mean acceptance length 2.87，接受率 37.3%（per-position 0.667/0.533/0.400/0.200/0.067）。

**对比纯 target（SGLang 无投机）**：17-45 tok/s → **提升约 5-10 倍**。

## 4. 镜像源速查表（核心：无海外网络下的解决方案）

> 以下全部为实测结论。按"用途 → 源 → 效果"组织，含备选。

### 4.1 模型权重（HF 模型）

| 源 | 用法 | 实测 | 备注 |
|---|---|---|---|
| **ModelScope（首选）** | `pip install modelscope; modelscope download --model <id> --local_dir <dir>` | ✅ **~8-13 MB/s，稳定** | 国内原生，大文件可靠；DeepSeek 官方模型有镜像 |
| hf-mirror.com | `HF_ENDPOINT=https://hf-mirror.com huggingface-cli download ...` | ⚠️ 小文件 OK；**大文件超时**（read timed out 反复重试）| 307 跟随可用，但 170GB 权重实测不行 |
| HF 直连 | — | ❌ 不可达 | 服务器无海外网 |

> **关键坑**：`huggingface_hub` 新版默认走 **xet 加速**（`cas-server.xethub.hf.co`），该域被墙。必须 `HF_HUB_DISABLE_XET=1` 或直接用 ModelScope。

### 4.2 Python 包（pip）

| 源 | 用法 | 实测 | 备注 |
|---|---|---|---|
| **清华 PyPI** | `-i https://pypi.tuna.tsinghua.edu.cn/simple` | ✅ 200，全量 | sglang / vllm / torch / flashinfer-python 全有 |
| 阿里 PyPI | `-i https://mirrors.aliyun.com/pypi/simple` | ✅ 200 | 备用 |

**覆盖情况（关键版本实测存在）**：
- `sglang==0.5.16` ✅、`vllm==0.26.0` ✅
- `torch==2.13.0` ✅（**PyPI 标准版即 cu130**，依赖 `nvidia-*-cu13` 全套）
- `flashinfer-python==0.6.16` ✅、`flashinfer-cubin==0.6.16` ✅
- `sgl-kernel` / `sglang-kernel` / `triton-kernels` ✅

> **坑**：包名是 `flashinfer-python` 不是 `flashinfer`（后者 404）。`flashinfer-jit-cache` 清华无，但 flashinfer.ai 根可达可补。

### 4.3 GitHub 源码/发布（无海外网核心难题）

| 方式 | 用法 | 实测 | 备注 |
|---|---|---|---|
| **ghfast.top（首选）** | `https://ghfast.top/https://github.com/<owner>/<repo>/archive/<ref>.tar.gz` | ✅ **~2-3 MB/s**，稳定 | 代理 GitHub 下载；tag/commit 均可 |
| ghproxy.net | 同上格式 | ✅ 200 | 备用 |
| git 直连 GitHub | `git clone https://github.com/...` | ❌ 卡死/超时 | 即使配代理也极慢（节点带宽差）|
| 官方 tarball | `github.com/<repo>/archive/...` | ❌ 不可达 | — |

> **用法要点**：
> - 下载源码 tarball：`archive/refs/heads/<branch>.tar.gz` 或 `archive/<commit>.tar.gz`
> - 编译期 GitHub 依赖（CUTLASS 等）：先 ghfast 下 tarball，再通过 cmake 的 `*_SRC_DIR` 环境变量指向本地（见 §6.4）
> - 查仓库 submodule pin：`gh api repos/<repo>/git/trees/<commit>?recursive=1`

### 4.4 编译产物 / 特殊 index

| 源 | 用法 | 实测 |
|---|---|---|
| flashinfer.ai | `--extra-index-url https://flashinfer.ai/whl/cu130` | ✅ 根可达（200）|
| 阿里 pytorch-wheels | `mirrors.aliyun.com/pytorch-wheels/cu130/` | ✅ 200（torch 已从 PyPI 拿到则不用）|
| 清华 pytorch-wheels | `mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/` | ❌ 404（目录结构不对）|

### 4.5 Docker 镜像（结论：放弃）

| 源 | 实测 | 结论 |
|---|---|---|
| Docker Hub 直连 | ❌ 被墙 | 不可用 |
| docker.xuanyuan.me | 429 限流 | 小镜像可，大镜像不行 |
| docker.m.daocloud.io | 401 但 pull 卡死 | 不可用 |
| docker.1panel.live | manifest 200 但 blob 极慢 | 不可用 |
| docker.1ms.run 等 | hello-world 要 4-5 分钟 | 太慢 |

> **结论**：国内免费 Docker 加速源对 18GB 级大镜像全部不可用，**放弃 docker 路线，改 pip/源码**。

### 4.6 综合推荐（无海外网络）

1. **权重** → ModelScope（不用 hf-mirror）
2. **pip** → 清华 PyPI（全依赖一个源搞定）
3. **GitHub 源码** → ghfast.top 逐个下 tarball
4. **一切 GitHub 编译期依赖** → ghfast + `*_SRC_DIR` 指向本地解压目录
5. **不做 Docker**（镜像源全废）

## 5. 引擎选型结论

| 引擎 | SM120 DSpark | 结论 |
|---|---|---|
| SGLang 0.5.16（pip）| ❌ 内核限制 `num_tokens > 64`（flashinfer SM120 sparse-MLA decode 要求 batch≥64，DSpark 提交 5）| 纯 target 可用（~17-45 tok/s），无投机 |
| vLLM stock 0.26 | ❌ 同样限制（hermia 确认）| 纯 target，256K 上限 |
| **vLLM PR #41834 (jasl fork)** | ✅ **DSpark 是主投机路径** | **最终采用** |

> 0731 checkpoint 已移除 MTP 头、内置 DSpark draft，所以 `method:"dspark"` 是唯一投机路径。

## 6. vLLM fork 源码编译（无外网实操）

### 6.1 获取源码

```bash
# 分支 tarball（ghfast 代理），37MB
curl -L -o vllm-fork.tar.gz "https://ghfast.top/https://github.com/jasl/vllm/archive/refs/heads/codex/ds4-sm120-min-enable.tar.gz"
tar xzf vllm-fork.tar.gz   # → ~/vllm-codex-ds4-sm120-min-enable
```

### 6.2 环境准备

```bash
python3 -m venv ~/vllm-venv
~/vllm-venv/bin/pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
~/vllm-venv/bin/pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple torch==2.13.0 flashinfer-python==0.6.16
~/vllm-venv/bin/pip install setuptools-scm setuptools-rust wheel ninja cmake -i https://pypi.tuna.tsinghua.edu.cn/simple
sudo apt-get install -y ninja-build cmake python3.10-venv
```

### 6.3 编译踩坑与修复

| # | 错误 | 根因 | 修复 |
|---|---|---|---|
| 1 | `ModuleNotFoundError: setuptools_scm` | tarball 无 .git，setuptools-scm 无法取版本 | 手动创建 `vllm/_version.py`：`__version__="0.26.0.post1+sm120"` |
| 2 | `ModuleNotFoundError: setuptools_rust` | --no-build-isolation 模式缺构建依赖 | 装 setuptools-rust/setuptools-scm/wheel |
| 3 | `CMake 3.26 required, have 3.22` | 系统 cmake 太老 | `pip install cmake`（4.4.0），PATH 前置 venv/bin |
| 4 | git clone GitHub 依赖卡死 | FetchContent 拉 CUTLASS/DeepGEMM 等 | ghfast 逐个下载 tarball + SRC_DIR 指向本地 |
| 5 | triton_kernels 需 MLIR | SRC_DIR 指向了 triton 根目录 | 指向 `python/triton_kernels/triton_kernels`（python 包目录）|
| 6 | `FlashKDA: cutlass/bfloat16.h not found` | CUTLASS 头文件没传给 FlashKDA | `CPLUS_INCLUDE_PATH=~/vllm-deps/cutlass/include` |
| 7 | FA3 `sm90_get_smem_store_op_for_accumulator` 不匹配 | **FA 的 CUTLASS submodule 是 v3.9**，vLLM 主是 v4.4.2 | 下载 CUTLASS v3.9 软链到 `flash-attention/csrc/cutlass` |
| 8 | DeepGEMM install: `third-party/cutlass/include not found` | DeepGEMM 也有 CUTLASS submodule（v4.2.1）| 下载 v4.2.1 软链到 `DeepGEMM/third-party/cutlass` |
| 9 | MSA install: `python/fmha_sm100/cutlass/include not found` | MSA 的 CUTLASS submodule（v4.3，commit eb61c911）| 下载软链到 `MSA/python/fmha_sm100/cutlass` |
| 10 | `flashinfer fd_exchange.py: 'type' object is not subscriptable` | flashinfer 0.6.16 用了 `array.array[int]`，Python 3.10 不支持 | sed 改 `array.array[int]` → `array.array` |
| 11 | `No module named 'flash_attn'` | vLLM 需要顶层 flash_attn 包（cute/）| 建最小包：复制 FA `flash_attn/cute/*.py` + 空 `__init__.py` |

### 6.4 依赖 SRC_DIR 清单（编译命令）

```bash
DEEPGEMM_SRC_DIR=~/vllm-deps/DeepGEMM \
FLASH_MLA_SRC_DIR=~/vllm-deps/FlashMLA \
FMHA_SM100_SRC_DIR=~/vllm-deps/MSA \
FLASH_KDA_SRC_DIR=~/vllm-deps/FlashKDA \
QUTLASS_SRC_DIR=~/vllm-deps/qutlass \
TML_FA4_SRC_DIR=~/vllm-deps/tml-fa4 \
VLLM_FLASH_ATTN_SRC_DIR=~/vllm-deps/flash-attention \
TRITON_KERNELS_SRC_DIR=~/vllm-deps/triton_kernels/python/triton_kernels/triton_kernels \
VLLM_CUTLASS_SRC_DIR=~/vllm-deps/cutlass \
~/vllm-venv/bin/pip install -e . --no-build-isolation --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 6.5 各依赖 CUTLASS submodule 版本（关键！）

| 依赖 | CUTLASS 版本 | 获取方式 |
|---|---|---|
| vLLM 主 | v4.4.2 | ghfast tag tarball |
| flash-attention | v3.9 (commit 62750a2b) | ghfast tag v3.9.0 |
| DeepGEMM | v4.2.1 (commit f3fde583) | ghfast tag v4.2.1 |
| MSA (fmha_sm100) | v4.3 (commit eb61c911) | ghfast commit tarball |

> 查询 submodule pin：`gh api repos/<repo>/git/trees/<commit>?recursive=1` 找 gitlink sha。

## 7. 服务启动踩坑

| 问题 | 修复 |
|---|---|
| Engine core 初始化失败（flash_attn 缺失）| 见 6.3 #11 |
| `6.97 GiB KV needed > 1.12 GiB available` | `--gpu-memory-utilization 0.92→0.95` + `--max-model-len 32768→8192` |

## 8. 当前状态与后续优化

**运行中**：port 8000，`curl http://127.0.0.1:8000/v1/chat/completions` 可用。

**后续可调**：
1. **加大 context**：`--max-model-len` 8192 是显存妥协（KV 10,364 tokens @ 0.95）。128K 需评估显存，或 `--gpu-memory-utilization` 提到 0.97-0.98
2. **并发吞吐**：多请求测试 c2/c4 聚合吞吐（当前单流 ~215 tok/s）
3. **K 值**：`num_speculative_tokens` 5（DSpark block size），可试 7 对比
4. **prefix caching**：`--enable-prefix-caching` 对共享 system prompt 场景有提升
5. SGLang 纯 target（port 30000）可保留作对照/降级方案

## 9. 关键文件位置

| 文件 | 路径 |
|---|---|
| vLLM 源码 | ~/vllm-codex-ds4-sm120-min-enable |
| vLLM venv | ~/vllm-venv |
| 依赖源码 | ~/vllm-deps/ |
| 模型权重 | ~/models/DeepSeek-V4-Flash-0731 (156G) |
| vLLM 日志 | ~/vllm_server.log |
| SGLang venv | ~/sglang-venv（备用）|
