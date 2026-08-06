#!/usr/bin/env bash
# 修复 Ray 启动失败：opentelemetry 版本冲突
# 在每台机器上执行一次：bash /root/fix_otel.sh
set -e

echo "== 升级 opentelemetry 全家到匹配版本（Ray dashboard 依赖）=="
/root/miniforge3/bin/pip install --no-deps -q \
  "opentelemetry-api==1.44.0" \
  "opentelemetry-sdk==1.44.0" \
  "opentelemetry-semantic-conventions==0.58b0" \
  "opentelemetry-proto==1.44.0" \
  "opentelemetry-exporter-prometheus==0.65b0" \
  "opentelemetry-exporter-otlp==1.44.0" \
  "opentelemetry-exporter-otlp-proto-common==1.44.0" \
  "opentelemetry-exporter-otlp-proto-grpc==1.44.0" \
  "opentelemetry-exporter-otlp-proto-http==1.44.0"

echo "== 验证 =="
/root/miniforge3/bin/python -c "
from opentelemetry.semconv._incubating.attributes.otel_attributes import OtelComponentTypeValues
assert hasattr(OtelComponentTypeValues, 'PROMETHEUS_HTTP_TEXT_METRIC_EXPORTER')
print('otel 修复 OK')
"
