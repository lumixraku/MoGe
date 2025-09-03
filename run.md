## command
uv run moge infer --device cpu -i example_images/blahaj.jpg -o output/ --maps

## start web service
uv run python moge/scripts/app_cpu.py --device cpu

## start API service
uv run simple_api.py

## request
# 获取信息
curl -X POST http://localhost:8000/predict -F "file=@example_images/blahaj.jpg"

# 下载深度图
curl http://localhost:8000/download/blahaj_depth.png -o blahaj_depth.png



##
curl 'http://127.0.0.1:7861/gradio_api/queue/join?' \
  -H 'Accept: */*' \
  -H 'Accept-Language: en,zh-CN;q=0.9,zh;q=0.8,en-US;q=0.7,ja;q=0.6' \
  -H 'Cache-Control: no-cache' \
  -H 'Connection: keep-alive' \
  -H 'Origin: http://127.0.0.1:7861' \
  -H 'Pragma: no-cache' \
  -H 'Referer: http://127.0.0.1:7861/' \
  -H 'Sec-Fetch-Dest: empty' \
  -H 'Sec-Fetch-Mode: cors' \
  -H 'Sec-Fetch-Site: same-origin' \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36' \
  -H 'content-type: application/json' \
  -H 'sec-ch-ua: "Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"' \
  --data-raw '{"data":[{"path":"/private/var/folders/6w/l0zkgvvs0_7fkm3x8rwdfpvh0000gn/T/gradio/34e1e8ee6d35a563752df4ade37f8549664c2f652ef587ede8bf2085510e417b/blahaj.jpg","url":"http://127.0.0.1:7861/gradio_api/file=/private/var/folders/6w/l0zkgvvs0_7fkm3x8rwdfpvh0000gn/T/gradio/34e1e8ee6d35a563752df4ade37f8549664c2f652ef587ede8bf2085510e417b/blahaj.jpg","orig_name":"blahaj.jpg","size":39207,"mime_type":"image/jpeg","meta":{"_type":"gradio.FileData"}},800,"High",true,true],"event_data":null,"fn_index":2,"trigger_id":13,"session_hash":"pgouo101mk"}'