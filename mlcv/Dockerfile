# drawcv: модуль CV (OpenCV) + тесты, работает без интернета (модели детектора текста скачаны при сборке).
# Сборка (из корня репозитория):  docker build -t drawcv mlcv
# Тесты:                          docker run --rm drawcv
# Лист -> cv.json:                docker run --rm -v "D:/путь/к/папке:/data" drawcv drawcv /data/лист.pdf --out /data/out
# Оценка на датасете:             docker run --rm -v "D:/pp/ProjectPractice44Labs:/repo" drawcv python /repo/mlcv/eval/evaluate.py --out /repo/mlcv/eval/out
FROM python:3.14-slim
ENV PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt
COPY pyproject.toml ./
COPY drawcv ./drawcv
COPY tests ./tests
RUN pip install --no-deps . \
 && python -c "import numpy as np; from drawcv.textmask import detect_rapidocr; detect_rapidocr(np.full((64, 64), 255, np.uint8))"
CMD ["python", "-m", "pytest", "-q", "tests"]
