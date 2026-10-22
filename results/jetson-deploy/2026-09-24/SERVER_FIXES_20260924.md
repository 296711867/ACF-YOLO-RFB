# 服务器侧执行修复记录（2026-09-24，RTX 3090 / TRT 11.3.0.99）

RTX 侧跑通 `run_rtx_reference.sh` 所需的三处环境适配，Windows 侧的 bundle 副本建议同步吸收：

## 1. build_engine.py：TRT 11 移除了 BuilderFlag.FP16

TRT 11.3 的 BuilderFlag 已无 FP16/INT8（强类型网络是唯一路线），原 `--mode fp16` 的
`config.set_flag(trt.BuilderFlag.FP16)` 直接 AttributeError。修复：探测到无该旗标时改走
**fp16-native ONNX + strongly-typed** 路线（语义等价旧 FP16 旗标）。

曾尝试 onnxconverter-common 转换 fp32→fp16 两条路（keep_io_types / Concat 加 block list），
分别卡在混合 dtype Concat 和检测头 cast 链 bookkeeping，均放弃；最终方案是用
Ultralytics `export(half=True)` 从同一份 .pt 权重直接导出 fp16 ONNX：
`models/{l0,k1}_ep200_fp16.onnx`（各自 5.08MB；fp32 版 SHA 不变，前置核验不受影响）。
`trt_runner.py` 本身 dtype 自适应（fp32 宿主输入自动转 fp16），无需改动。

## 2. data/bdd10k_classes.json（缺失）

`eval_frozen_subset.py` 等脚本需要，bundle 未携带。已从仓库
`configs/data/bdd100k.yaml` 的 names 生成（10 类，与训练配置一致）。

## 3. data/bench_images_100.txt（缺失）

`bench_latency.py` 的图片清单默认路径，bundle 未携带。已复制仓库冻结清单
`results/acfr-stage20/bench_images_100.txt`（与 frozen_subset/bench_100/ 的 100 张
逐一核对存在）。

## 产物（results/，只增不改）

- `subset_size_val_{l0,k1}_rtx_fp16.json`（+ predictions.jsonl）
- `latency_b1_{l0,k1}_rtx_20260924_*.json`
- `engine_build_{l0,k1}_ep200_fp16_fp16_b1.json`（构建记录，含引擎 SHA）

## 对减结论（预注册判据 |ΔAP|≤0.5%）

l0: all +0.187% / small -0.085%；k1: all -0.034% / small +0.234% —— 全 PASS。
延迟 FP16 b1：L0 2.306ms / K1 2.301ms。
