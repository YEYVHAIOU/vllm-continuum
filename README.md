# vLLM + Continuum 研究分支

本仓库保存本项目最终使用的 vLLM / Continuum 修改源码。它以 Continuum 调度代码为基础，进一步加入 Dynamic TTL、Prefill / Reload 成本建模、运行时 KV 压力与时序提示，以及与 UCM KVConnector 协作所需的生命周期和性能修正。

主项目、部署脚本和正式实验结果位于：

<https://github.com/YEYVHAIOU/vllm-prefix-experiment>

> 本仓库是研究修改分支，不是 vLLM 或 Continuum 的官方发行版。

## 最终版本

```text
branch = joint-offload-v1
commit = 6e7d571b831e6e4b82f1b2f8228cc84e9a0261a2
tag    = vllm-continuum-final-20260820
```

正式 V/U/C/F 实验均以该源码树作为 vLLM 侧统一代码基座，通过调度策略和 UCM 开关构造不同配置。

## 主要修改

### Dynamic TTL

Continuum 基础代码使用固定工具调用时间阈值控制 KV 驻留。本项目在此基础上加入动态 TTL 估计，使保留时间能够结合工具调用历史、上下文规模和 Prefill / Reload 成本变化。

主要实现包括：

```text
vllm/v1/core/dynamic_ttl_estimator.py
vllm/v1/core/estimate_with_func.py
```

正式配置支持：

```text
CONTINUUM_TTL_CDF_IMPL
CONTINUUM_HISTORY_THRESHOLD
CONTINUUM_DEFAULT_TTL_SECONDS
CONTINUUM_PREFILL_PROFILE_SCALE
CONTINUUM_TTL_TIMING_INTERVAL
```

经验 CDF 路径同时进行了性能优化，以减少调度热路径中的重复扫描开销。

### 运行时 KV 上下文

为了让 UCM 能够使用 Continuum 的时序信息，vLLM 侧向 KVConnector 传递运行时上下文，包括：

```text
kv_pressure
continuum_hints
context_tokens
is_terminal
finish_probability
expected_tool_duration
prefill_reload_cost
```

这些信息用于联合判断未来复用价值、GPU KV 压力和外部迁移成本。

### KVConnector 生命周期

项目针对当前 vLLM V1 与 UCM 的组合处理了 load/save 活跃状态、元数据生命周期、warmup 和空传输路径。

当一次前向过程没有实际 KV load/save 时，vLLM 可以通过 active load/save 状态跳过不必要的 KVConnector 数据路径，从而降低额外同步和 Python 调用开销。

### CUDA Graph

早期 UCM 集成曾使用 `--enforce-eager` 保证兼容。随着 KVConnector 生命周期和空传输路径完善，最终正式版本恢复非 eager 执行，允许正常 CUDA Graph 路径。

## 与 UCM 的关系

UCM 修改源码位于独立仓库：

<https://github.com/YEYVHAIOU/unified-cache-management-continuum>

最终联合系统使用：

```text
Dynamic TTL
    +
cost_full WHEN
    +
Frontier-Tail WHAT (K=4)
    +
TieredStore WHERE
```

vLLM 侧负责调度、TTL、运行时上下文和 KVConnector 调用；UCM 侧负责外部 KV 的 WHEN / WHAT / WHERE 决策与存储实现。

## 快速使用

本仓库不单独提供完整实验入口。推荐按照主项目的三个同级仓库结构部署：

```text
workspace/
├── vllm-prefix-experiment/
├── vllm-continuum/
└── unified-cache-management-continuum/
```

切换本仓库到最终标签：

```bash
git checkout vllm-continuum-final-20260820
```

随后按照主项目的 [`docs/DEPLOYMENT.md`](https://github.com/YEYVHAIOU/vllm-prefix-experiment/blob/main/docs/DEPLOYMENT.md) 配置 Python 环境、模型和 UCM 源码路径。

完整系统启动入口：

```bash
bash deployment/scripts/start_full.sh
```

正式基准测试方法与结果：

<https://github.com/YEYVHAIOU/vllm-prefix-experiment/blob/main/docs/BENCHMARK.md>

## 运行环境

最终验证环境为 RTX 4090、Python 3.12.3、PyTorch 2.8.0+cu129 和 CUDA Toolkit 12.8。运行时报告的 vLLM 包版本为：

```text
0.1.dev10+g05f00f8a8
```

该字符串来自构建阶段生成的 `_version.py`；源码身份以本页记录的 Git commit/tag 为准。

## 当前边界

Dynamic TTL 使用的基础 Prefill 曲线历史上来自 RTX 4060，项目在 RTX 4090 上完成过缩放敏感性分析，但没有重新完整拟合 4090 专用曲线。跨 GPU 使用时建议重新校准。

UCM 上游显式支持的 vLLM 版本与本项目源码身份不同，因此联合运行依赖本项目额外完成的 KVConnector 生命周期与接口适配。详细修改见主项目：

<https://github.com/YEYVHAIOU/vllm-prefix-experiment/blob/main/docs/COMPATIBILITY.md>

## 上游与许可证

本仓库建立在 vLLM 与 Continuum 相关开源代码基础上。原有许可证、版权声明和第三方 notice 应继续保留，并按照对应上游条款进行使用和再分发。

本项目对上游代码的修改历史可通过 Git commit 和 tag 追溯。
