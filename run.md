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


## build docker
docker build -t slim-moge-image
docker-compose -f docker-compose.cpu.yml up
