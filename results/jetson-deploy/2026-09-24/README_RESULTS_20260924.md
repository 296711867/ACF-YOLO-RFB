# Jetson 部署实验结果包（2026-09-24）

板卡：NVIDIA Orin NX Developer Kit 16GB ｜ JetPack 5.1.3 (L4T R35.5.0) ｜ CUDA 11.4 ｜ cuDNN 8.6.0.166 ｜ TensorRT 8.5.2.2 ｜ MAXN + jetson_clocks 锁定
（台账脚本 dpkg 查询用的是 JetPack6 包名会显示 no packages found，以上为真实版本，论文表格以此为准）

## 本包内容

- `results/` —— 全部实验证据 JSON/图/视频（机器证据，论文数字从这里摘录）
- `scripts/` —— 实际运行的脚本（含 4 处本机兼容修复，见下；比 jetson_patch_20260924.zip 更全）

## 核心结果

| 指标 | L0 FP16 | K1 FP16 |
|---|---|---|
| 纯模型延迟 P50 / P95（E） | 10.211 / 10.253 ms | 10.221 / 10.275 ms |
| trtexec GPU 计时 mean（交叉验证） | 9.571 ms | 9.599 ms |
| 图像端到端 FPS（F） | 29.0 | 29.9 |
| 视频稳态 FPS（G，0.55× 实时） | 16.65 | 16.59 |
| GPU 功耗均/峰（H，延迟轮） | 8.43 / 12.95 W | 8.41 / 12.95 W |
| 整机功耗均/峰（VDD_IN） | 16.42 / 22.30 W | 16.41 / 22.30 W |
| 稳定温度（末10样本） | 58.6 °C | 59.8 °C |
| RAM 峰值 | 7263 MB | 7238 MB |
| 冻结子集 AP all/small（I，1000 图） | 0.272698 / 0.111138 | 0.276181 / 0.107471 |

**功耗口径注意**：本板无独立 GPU 电轨，`gpu_power_mw` 来自 `VDD_CPU_GPU_CV`（CPU+GPU+CV 合并轨，
是 GPU 功耗上界）；整机 `VDD_IN` 无歧义。论文表格须如实标注（各 power_*.json 的
gpu_power_source / field_note 字段已注明）。

## 待办（不在本机）

1. ~~RTX 参考侧对跑~~ **已完成（2026-09-24 23:06，见下方"RTX 参考侧对跑结果"）**
2. ~~服务器 FP16 绝对延迟复核（§0-1 建议项）~~ **已完成（同上，fp16 误标事项关闭）**
3. 导师追认清单：G2/G3 判据校准（见 results/NUMERIC_CONSISTENCY_G_DECOMPOSITION.md）；
   D 步骤 FAIL 定性（5/580 失配全部为 0.25 阈值 ±0.01 边界翻转，匹配对 IoU 0.986，引擎健全）

## RTX 参考侧对跑结果（2026-09-24 补录，RTX 3090 / TRT 11.3.0.99）

证据文件（与 Jetson 侧同目录）：`subset_size_val_{l0,k1}_rtx_fp16.json`（+predictions.jsonl）、
`latency_b1_{l0,k1}_rtx_20260924_*.json`、`engine_build_{l0,k1}_ep200_fp16_fp16_b1.json`。

**精度一致性（预注册 |ΔAP|≤0.5%，四项全 PASS）**：

| | Jetson FP16 | RTX FP16 | Δ | 判定 |
|---|---:|---:|---:|---|
| L0 all / small | 0.272698 / 0.111138 | 0.273209 / 0.111044 | +0.187% / -0.085% | PASS |
| K1 all / small | 0.276181 / 0.107471 | 0.276087 / 0.107722 | -0.034% / +0.234% | PASS |

两侧预测清单 SHA256 一致（`e5eb190f...`，同一份冻结子集）。

**RTX FP16 b1 延迟（新绝对口径，关闭 §11.3 fp16 误标事项）**：
per-forward P50 = L0 1.6947 / K1 1.6988 ms（K1 +0.24%，与 Jetson 侧同脚本同口径
"pure engine forward, H2D inside, CUDA events"；run 级均值口径为 2.3063/2.3014 ms，
论文引用以 per-forward P50 为准，与 Jetson 侧 10.21/10.22 的引用口径一致）。

**口径注记（论文必须写明）**：TRT 11.3 移除了 BuilderFlag.FP16，RTX 侧引擎为
**同一份 .pt 权重经 Ultralytics export(half=True) 导出的 fp16-native ONNX + strongly-typed**
（语义等价旧 FP16 旗标）；Jetson 侧（TRT 8.5）为 fp32 ONNX + FP16 旗标。两侧构建路线不同
但精度一致性由上表受控验证。详见 `../rtx_reference_bundle/SERVER_FIXES_20260924.md`。

## 本机脚本修复（相对 jetson_patch_20260924.zip 的增量）

1. `build_engine.py`：TRT 8.5 需显式 EXPLICIT_BATCH 建网（探测式兼容 TRT 10）
2. `trt_runner.py`：旧 pycuda 无 Device.retain_context → 回退 pycuda.autoinit（否则退出 abort）
3. `bench_power_temp.py`：sudo 跑 tegrastats（非 root 无电轨）；root 下 --logfile 失效改重定向 stdout；
   用 `sudo pkill -P` 停止 root 采样器；新增 VDD_CPU_GPU_CV 合并轨解析与来源标注
4. 运行一律用 `/usr/bin/python3`（conda base 自动激活会顶掉系统 python3）
