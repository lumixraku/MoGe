# MoGe Gradio API 正确使用指南

本文档介绍了如何正确使用 curl 调用 MoGe 的 Gradio API 来获取模型推断结果，包括深度图和法线图。

## 启动 MoGe Gradio 应用

首先，您需要启动 MoGe 的 Gradio 应用：

```bash
# 使用 CPU 运行（适用于没有 CUDA 支持的环境）
uv run python moge/scripts/app_cpu.py --device cpu
```

应用启动后，默认会在 `http://127.0.0.1:7861` 运行。

## 正确的 API 调用方法

### 1. 提交任务到队列

MoGe 使用队列系统处理请求，需要先提交任务到 `/gradio_api/queue/join` 端点：

```bash
curl 'http://127.0.0.1:7861/gradio_api/queue/join?' \
  -H 'Accept: */*' \
  -H 'Content-Type: application/json' \
  -H 'Origin: http://127.0.0.1:7861' \
  -H 'Referer: http://127.0.0.1:7861/' \
  --data-raw '{
    "data": [
      {
        "path": "/path/to/your/image.jpg",
        "url": "http://127.0.0.1:7861/gradio_api/file=/path/to/your/image.jpg",
        "orig_name": "image.jpg",
        "size": 12345,
        "mime_type": "image/jpeg",
        "meta": {"_type": "gradio.FileData"}
      },
      800,
      "High",
      true,
      true
    ],
    "event_data": null,
    "fn_index": 2,
    "trigger_id": 13,
    "session_hash": "unique_session_hash"
  }'
```

### 2. 获取处理结果

提交任务后，使用返回的 session_hash 获取处理结果：

```bash
curl 'http://127.0.0.1:7861/gradio_api/queue/data?session_hash=unique_session_hash' \
  -H 'Accept: text/event-stream' \
  -H 'Origin: http://127.0.0.1:7861' \
  -H 'Referer: http://127.0.0.1:7861/'
```

## 参数说明

1. 图像文件信息：
   - `path`: 本地文件路径
   - `url`: 文件的 URL（如果是本地文件，可以是本地服务器 URL）
   - `orig_name`: 原始文件名
   - `size`: 文件大小（字节）
   - `mime_type`: MIME 类型
   - `meta`: 元数据

2. 其他参数：
   - `fn_index`: 函数索引（在 MoGe 中通常是 2）
   - `trigger_id`: 触发器 ID（在 MoGe 中通常是 13）
   - `session_hash`: 会话哈希（需要生成唯一的字符串）

3. 图像处理参数：
   - 最大图像尺寸（默认：800）
   - 推断分辨率级别（可选值："Low", "Medium", "High", "Ultra"，默认："High"）
   - 是否应用遮罩（默认：true）
   - 是否移除边缘（默认：true）

## 响应格式

成功处理后，API 将返回包含以下信息的响应：

```json
{
  "msg": "process_completed",
  "output": {
    "data": [
      null,
      {
        "path": "/path/to/depth.exr",
        "url": "http://127.0.0.1:7861/gradio_api/file=/path/to/depth.exr",
        "orig_name": "depth.exr"
      },
      {
        "path": "/path/to/normal.exr",
        "url": "http://127.0.0.1:7861/gradio_api/file=/path/to/normal.exr",
        "orig_name": "normal.exr"
      },
      // 其他输出文件...
    ]
  }
}
```

## 注意事项

1. MoGe 应用默认运行在端口 7861 上
2. 必须包含正确的 HTTP 头部信息
3. 需要使用队列接口而不是直接的预测接口
4. 需要生成唯一的 session_hash
5. 处理可能需要一些时间，特别是对于高分辨率图像
6. 返回的文件路径是临时文件，处理完后会被删除

## 完整示例

以下是一个完整的调用示例：

```bash
# 1. 提交任务
curl 'http://127.0.0.1:7861/gradio_api/queue/join?' \
  -H 'Accept: */*' \
  -H 'Content-Type: application/json' \
  -H 'Origin: http://127.0.0.1:7861' \
  -H 'Referer: http://127.0.0.1:7861/' \
  --data-raw '{
    "data": [
      {
        "path": "/private/var/folders/6w/l0zkgvvs0_7fkm3x8rwdfpvh0000gn/T/gradio/34e1e8ee6d35a563752df4ade37f8549664c2f652ef587ede8bf2085510e417b/blahaj.jpg",
        "url": "http://127.0.0.1:7861/gradio_api/file=/private/var/folders/6w/l0zkgvvs0_7fkm3x8rwdfpvh0000gn/T/gradio/34e1e8ee6d35a563752df4ade37f8549664c2f652ef587ede8bf2085510e417b/blahaj.jpg",
        "orig_name": "blahaj.jpg",
        "size": 39207,
        "mime_type": "image/jpeg",
        "meta": {"_type": "gradio.FileData"}
      },
      800,
      "High",
      true,
      true
    ],
    "event_data": null,
    "fn_index": 2,
    "trigger_id": 13,
    "session_hash": "pgouo101mk"
  }'

# 2. 获取结果（使用上面返回的 session_hash）
curl 'http://127.0.0.1:7861/gradio_api/queue/data?session_hash=pgouo101mk' \
  -H 'Accept: text/event-stream' \
  -H 'Origin: http://127.0.0.1:7861' \
  -H 'Referer: http://127.0.0.1:7861/'
```

通过这种方式，您可以成功获取到深度图、法线图和其他 3D 几何数据。