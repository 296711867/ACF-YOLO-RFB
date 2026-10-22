# Jetson 重测结果包(2026-10-17)— AGX Orin Super 首轮

板卡: NVIDIA Jetson AGX Orin Developer Kit Super 32GB | L4T R39.2 (2026-06) | TensorRT 10.16.2.10+cuda13.2 (apt)
| MAXN_SUPER + jetson_clocks | Python 3.12 venv(~/venv_trt, cuda-python 12.9 充当 pycuda 兼容层,见 scripts/cuda_shim.py)
| 输入 640×640 b1 | 口径与 2026-09-24 Orin NX 轮完全一致(48 预热+52×10、CUDA 事件含 H2D、5 轮取中位)

## 旧板(Orin NX 16GB, JetPack 5.1.3/TRT 8.5)全部数字作废; 本章以本包为准

## 核心结果(FP16, b1)

| 指标 | L0 (ACF-YOLO) | K1 (KD) | RFB (最终部署) |
|---|---|---|---|
| 纯推理延迟 ms_per_image_mean (E) | 4.802 | 4.811 | 5.164 |
| trtexec GPU 时间 mean (参照) | 4.537 | 4.515 | 4.794 |
| 图像端到端 FPS (F) | 61.3 | 61.1 | 60.1 |
| 拼接视频稳态 FPS (G, 450f) | 34.4 (1.15×实时) | 34.3 | 33.9 (1.13×) |
| 冻结子集 AP all/small (I, 1000图) | 0.272786 / 0.111188 | 0.276293 / 0.107447 | 0.275611 / 0.114115 |
| car.mp4 端到端 + 标注视频 | 见 end2end_video_*_car.json / car_*_annotated.mp4 | 同 | 同 |
| 数值一致性 D (fp16 vs fp32) | G4 IoU 0.986 PASS(raw 差仅 top-k 槽位漂移,协议既定签名) | 同 | 同 |
| 功耗/温度 (H) | power_latency_*.json / power_video_*.json / power_car_*.json ×3 模型 | | |

## 服务器对照(同批,RTX 3090 TRT 11.3.0.99,同子集)

- l0 0.272725/0.111064,k1 0.276180/0.107694,rfb(fp16io) 0.275631/0.114215
- Jetson↔服务器 all 差 ≤0.04%,small 差 ≤0.24%(rfb 0.09%),平台一致性与旧章结论一致
- 与 09-24 旧服务器值(l0 0.272698/0.111138,k1 0.276181/0.107471)复现差 <0.03% → 流水线可信

## 重要口径说明(与旧轮的差异)

1. **rfb 引擎配方**:首测用框架导出的半精度图(fp16 权重+fp16 输入端口),流水线 CPU 转 fp16 成瓶颈,
   延迟 9.38ms 失真;改用 keep_io_types 转换(fp16 权重+**fp32 输入端口**,与 l0/k1 同配方)重建后 5.164ms,
   trtexec 交叉验证 4.79ms 佐证。**论文表 6-2 用 fp32-IO 配方**;halfio 档案(引擎+全套数据)保留于
   *_halfio.* 供对照。服务器侧同步改配方重评(差 <1e-4)。
2. **D 步 status=FAIL 为协议既定签名**:raw 张量差来自 top-300 槽位重排(诊断字段),G1-G4 判据全过
   (配对 IoU≈0.986,检测一致性 ≥0.96);与 09-24 轮定性相同,归导师追认清单。
3. l0 两引擎的 engine_build_*.json 因链重启被跳过未生成(引擎本身经 D/E/F/G/I 实载验证);
   k1/rfb 构建记录完整。
4. pycuda 在 R39 无法安装(Ubuntu 打包依赖损坏),测量经 cuda_shim.py(cuda-python 驱动绑定)等价执行,
   计时语义不变(CUDA 事件、同流、同序)。
5. 素材:car_{l0,k1,rfb}_annotated.mp4(连续行车视频标注)、{l0,k1,rfb}_annotated.mp4(拼接视频标注)、
   scene_rfb/(GT|L0|RFB)与 scene_l0k1/(GT|L0|K1)五类场景三联图(day/dense/distant/night/occlusion)。

## 服务器侧证据镜像

<user>: results/jetson-deploy/rtx-server-20261017/(fp16 转换、引擎、子集 AP 与 predictions)
本地论文工作区: <project>\jetson_results_20261017\(本包镜像)
