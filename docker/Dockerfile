FROM python:3.10-slim

WORKDIR /app

# 复制PyInstaller打包生成的单个可执行文件
COPY dist/chatglm_rag_app ./chatglm_rag_app

# 复制配置文件（如果你的程序运行时需要它）
COPY config.yaml ./config.yaml

# 如果你的程序依赖模型文件，建议挂载数据卷或复制必要模型文件
# COPY models ./models  # 如果需要

EXPOSE 8000

CMD ["./chatglm_rag_app"]
