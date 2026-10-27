FROM python:3.12-slim
WORKDIR /srv
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import hashlib,pathlib,urllib.request; p=pathlib.Path('/srv/models/eslav_PP-OCRv5_rec_mobile.onnx'); p.parent.mkdir(parents=True,exist_ok=True); urllib.request.urlretrieve('https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec/eslav_PP-OCRv5_rec_mobile.onnx',p); assert hashlib.sha256(p.read_bytes()).hexdigest()=='08705d6721849b1347d26187f15a5e362c431963a2a62bfff4feac578c489aab'"
COPY backend/app ./app
COPY backend/migrations ./migrations
COPY backend/alembic.ini .
COPY runs/klapan_png2dxf/pipeline_b ./runs/klapan_png2dxf/pipeline_b
COPY tools/eskd_dxf.py ./tools/eskd_dxf.py
ENV PYTHONPATH=/srv
ENV OCR_MODEL_PATH=/srv/models/eslav_PP-OCRv5_rec_mobile.onnx
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
