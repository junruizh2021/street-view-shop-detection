# 街景商铺识别 Demo

本项目使用 MiniCPM-o 4.5 OpenVINO 模型分析车载前置摄像头拍摄的街景视频，识别道路两侧的包子铺、馒头店、早餐店、面点店及奶茶饮品店，并输出店铺首次出现的时间、名称或类型以及车辆左/右侧方位。

Demo 以连续三帧作为一个分析窗口。当前推荐配置在 Intel 358H 远端机器的 GPU 上运行，视频采样率为 6 FPS，使用 `omni` 后端并关闭并行 OCR。

## Demo 效果演示

![MiniCPM-o 包子铺 Demo 效果](assets/Minicpm-o%20-%20包子铺Demo%20-%202026年7月14日%2016.52.40.gif)

## 启动 Demo

在项目目录 `~/junrui/baozipu` 中执行：

```bash
python3 dashcam_event_web.py \
  --host 0.0.0.0 \
  --port 7861 \
  --vlm-backend omni \
  --vlm-every 1 \
  --sample-fps 6 \
  --model-path /home/auto/junrui/MiniCPM-o-4_5-OV \
  --video 街景视频.mp4 \
  --device GPU \
  --max-slice-nums 1 \
  --max-new-tokens 96 \
  --disable-ocr
```

Web UI 监听地址为 `http://0.0.0.0:7861`。远程访问时，在浏览器中使用 `http://<服务器 IP>:7861`，点击 **开始体验** 播放视频并查看检测事件。

## Docker 部署

Docker 镜像包含 Demo 源码、街景视频、MiniCPM-o OpenVINO 模型、Python 依赖和 Intel GPU 用户态驱动。默认从项目相邻目录 `../MiniCPM-o-4_5-OV` 读取模型，并使用 `http://proxy.cd.intel.com:911` 下载构建依赖。

一键构建：

```bash
./docker/build.sh
```

使用其他模型目录或镜像标签：

```bash
MODEL_DIR=/path/to/MiniCPM-o-4_5-OV \
IMAGE_TAG=baozipu-demo:custom \
./docker/build.sh
```

不使用代理或使用其他代理：

```bash
PROXY_URL="" ./docker/build.sh
PROXY_URL=http://proxy.example.com:8080 ./docker/build.sh
```

构建完成后启动 GPU 容器：

```bash
./docker/run.sh
```

脚本会自动挂载 `/dev/dri` 并添加设备对应的用户组。Web UI 监听地址为 `http://0.0.0.0:7861`；远程访问时使用 `http://<服务器 IP>:7861`。查看启动日志可执行：

```bash
docker logs -f baozipu-demo
```

## Benchmark

`minicpm_o_benchmark.py` 使用连续三帧街景窗口测试首 Token 延迟（TTFT）、端到端耗时（E2E）、解码速度和内存占用。

### 三帧拼接测试

`grid` 模式将连续三帧拼接成一张网格图，再作为单张视觉输入送入模型：

```bash
python minicpm_o_benchmark.py \
  --model-path /home/auto/junrui/MiniCPM-o-4_5-OV \
  --video 街景视频.mp4 \
  --device GPU \
  --input-mode grid \
  --sample-fps 6 \
  --max-slice-nums 1 \
  --max-new-tokens 96 \
  --warmup-windows 1 \
  --output minicpm-o-grid.json \
  2>&1 | tee minicpm-o-grid.log
```

### 三帧独立输入测试

`independent` 模式保留三帧为三张独立图片，并在同一次多模态请求中送入模型：

```bash
python minicpm_o_benchmark.py \
  --model-path /home/auto/junrui/MiniCPM-o-4_5-OV \
  --video 街景视频.mp4 \
  --device GPU \
  --input-mode independent \
  --sample-fps 6 \
  --max-slice-nums 1 \
  --max-new-tokens 96 \
  --warmup-windows 1 \
  --output minicpm-o-independent.json \
  2>&1 | tee minicpm-o-independent.log
```

Benchmark 生成的 `.json` 结果和 `.log` 日志仅保存在本机，不会提交到 Git 仓库。
