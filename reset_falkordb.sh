#!/bin/bash
# 一键清空并重启 falkordb-dev 容器
set -e

CONTAINER_NAME="falkordb-dev"

echo "=== 停止容器 ${CONTAINER_NAME} ==="
docker stop ${CONTAINER_NAME}

echo "=== 删除容器 ${CONTAINER_NAME} ==="
docker rm ${CONTAINER_NAME}

echo "=== 重新启动 ${CONTAINER_NAME} ==="
docker run -d \
  --name ${CONTAINER_NAME} \
  -p 3000:3000 \
  -p 6379:6379 \
  falkordb/falkordb

echo "=== 等待 FalkorDB 就绪 ==="
for i in $(seq 1 30); do
  if docker exec ${CONTAINER_NAME} redis-cli ping 2>/dev/null | grep -q PONG; then
    echo "FalkorDB 已就绪！"
    exit 0
  fi
  sleep 1
done

echo "警告：等待超时，请手动检查容器状态"
docker ps | grep ${CONTAINER_NAME}
