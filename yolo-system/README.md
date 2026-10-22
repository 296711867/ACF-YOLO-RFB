# 道路小目标检测工作台

已部署：[http://192.168.x.x:8078](http://192.168.x.x:8078)

- 本地源码：`<project>\yolo-system`
- Jetson 源码：`/home/<user>/projects/yolo-system`
- 推理依赖：`/home/<user>/projects/yolo_jetson/jetson_deploy`，复用其 `scripts/trt_runner.py`、CUDA 兼容层、三份 FP16 引擎和演示素材。
- Python：`/home/<user>/venv_trt/bin/python`。新增后端仅用 Python 标准库；OpenCV、NumPy、TensorRT、CUDA 绑定均复用既有环境。
- 前端：原生 HTML/CSS/JavaScript，响应式中文页面，无需 Node 构建。

## 使用

选择模型与输入文件，点击“开始检测”。“示例图片”使用冻结子集中的真实道路图片，“行车视频演示”使用已有 car.mp4。三模型对比仅用于图片，依次执行三份引擎，在同图、同阈值下显示结果。

视频运行时刷新标注预览、当前帧检测表和累计统计；可停止并保存已处理部分。视频累计框数包含同一目标在不同帧重复出现，不表示独立车辆数量。摄像头入口使用 Jetson `/dev/video0`；验收时设备未连接，尚未进行真机摄像头测试。

每个任务保存到 `runtime/<任务ID>/`：输入文件（上传时）、`result.json`、标注图，或标注 AVI 与逐帧 `frames.jsonl`。SQLite 保存最近任务索引，网页展示最近 100 条，可导出 CSV。图片对比的 CSV 汇总以 RFB 为主，全部模型结果在 JSON 中。

引擎 SHA-256、输入 SHA-256、输入文件名、模型、阈值、时间、处理口径均随结果留档。失败任务保留错误；服务重启时把未完成任务标记为 interrupted。

## 运维

```bash
ssh user@192.168.x.x
systemctl --user status yolo-workbench
systemctl --user restart yolo-workbench
journalctl --user -u yolo-workbench -n 80 --no-pager
/home/<user>/venv_trt/bin/python /home/<user>/projects/yolo-system/smoke_test.py
```

用户级 systemd 服务已启用，`loginctl show-user jetson -p Linger` 已验证为 `yes`，支持开机启动、退出 SSH 后运行和异常退出自动重启。没有重启 Jetson 做整机开机复验。修改本地源码后用 scp 同步到上述远端目录并重启服务；不覆盖原有部署证据。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 引擎可用性、设备、温度、内存、磁盘、摄像头、活动任务 |
| POST | `/api/jobs?kind=image&model=rfb&confidence=0.25&name=x.jpg` | 请求体为图片原始字节，202 返回任务 ID |
| POST | `/api/jobs?kind=video&model=rfb` | 请求体为可被 OpenCV 解码的视频原始字节 |
| POST | `/api/jobs?kind=demo-image&model=compare` | 真实示例图片的三模型对比 |
| POST | `/api/jobs?kind=demo-video&model=rfb` | car.mp4 视频演示 |
| POST | `/api/jobs?kind=camera&model=rfb` | Jetson USB 摄像头 |
| POST | `/api/stop` | 停止并保存当前任务 |
| GET | `/api/jobs` | 最近 100 个任务与结果 |
| GET | `/api/latest.jpg` | 当前任务最近标注帧 |
| GET | `/api/history.csv` | 历史汇总 |
| GET | `/artifacts/<任务ID>/<文件>` | 图片、JSON、JSONL、AVI 结果 |

模型 ID 为 `rfb`、`l0`、`k1`，图片支持 `compare`。阈值范围为 `(0,1]`，网页提供 0.05～0.95。上传限制 200 MB；视频和摄像头单任务最多 9 000 帧，达到上限记录 truncated。

## 测量与使用边界

- 单个专用工作线程持有 CUDA 上下文，串行处理推理；同一时刻只运行一个任务，第二个返回 409。状态接口不受推理线程阻塞。
- 每个引擎首次加载预热 10 次，不计模型加载与预热时间。图片的 pipeline_ms 包含预处理、GPU、回传、坐标恢复和绘制，不含图片解码、文件落盘及网络。
- GPU 时间为 CUDA 事件计时，含 H2D，不含 D2H。视频 P50/P95 包含读帧、预处理、GPU、D2H、解码、绘制、AVI 编码、JPEG 编码、逐帧记录；不含每 20 帧的任务索引提交、HTTP 传输与浏览器显示。FPS 为服务器累计处理帧数 / 处理总时间，包含索引提交和录像结束处理。
- 小目标按原图预测框面积 < 1 024 像素判定，是展示统计，不等于基于 GT 的 APs 或小目标召回。
- 网页参数只用于应用演示；不会更新训练或冻结评测配置。无 GT 输入不报告准确率、AP 或召回率。
- 当前为可信局域网研究演示，未配置账号权限与 HTTPS，不适合直接对公网开放。结果按任务留存，视频默认使用 mp4v AVI（浏览器下载后本地播放），每 4 500 帧分段，全部分段可下载；没有自动清理磁盘的策略。

## 2026-10-17 完整流程优化

默认流程逐帧检测、绘制、录制原分辨率标注视频及保存 JSONL，预览宽度不超过 960 像素、最高 10 Hz。`pipeline=baseline` 可复现原分辨率逐帧 JPEG + MJPG AVI 以便配对比较；默认 `pipeline=optimized` 改用 mp4v。压缩格式与质量条件发生变化，不宣称相同画质。

10 Hz 是后端 JPEG 生成上限。当前页面每 1.5 s 轮询任务并拉取最近预览，属于间歇查看，不是 30 FPS 流畅播放；浏览器实际显示 FPS 尚未单独测量。

视频 FPS 的明确边界：模型预热后、打开视频前开始，关闭录像后结束；逐帧日志和周期索引提交计入，开始前的输入哈希及结束后的最终预览文件/结果元数据保存不计入。不是从上传到下载完成的交互墙钟吞吐。

各阶段均值保存在 `stages_mean_ms`，CUDA 事件时间单独报告。`kind=demo-video&duration_s=1800` 可循环固定素材连续测试，最长 30 分钟；这一受限入口不适用于上传文件或摄像头。实验协议与驱动见 `experiments/20261017_pipeline/`。本次论文完整结果和逐章正文见 `../01_大论文/02_正文_v2_完整版_20261017/README_交付说明.md`。下列 22 FPS 原记录保留为优化前的历史证据。

## 验收与论文材料

- [自动验收结果](ACCEPTANCE.json)：11 类检查通过，包含三模型真实推理、图片/视频上传、视频结果导出、停止保存、冲突与非法输入检查。
- [界面截图](docs/workbench.png)：实际浏览器检测画面。
- [完整行车视频运行记录](docs/full_video_result.json)：452 帧全部完成，含预览与录制为 22.00 FPS；与冻结基准的计时口径不同。
- [项目盘点](docs/项目盘点_20261017.md)：服务器和本地缺口。
- [第六章新增小节草稿](docs/第六章_系统实现增补稿.md)：独立稿，尚未合并到正在更新的论文正文。
- `docs/evidence/`：本次只读复制的 P31、VisDrone、KITTI、RTTS 指标 JSON 与结果说明，未复制大体积 predictions 或训练权重；不等于完整备份。
