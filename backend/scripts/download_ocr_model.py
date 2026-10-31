import hashlib
import os
import time
import urllib.request
from pathlib import Path


MODEL_URL = (
    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/"
    "onnx/PP-OCRv5/rec/eslav_PP-OCRv5_rec_mobile.onnx"
)
MODEL_SHA256 = "08705d6721849b1347d26187f15a5e362c431963a2a62bfff4feac578c489aab"
MAX_DOWNLOAD_SECONDS = 180


def main() -> None:
    path = Path(os.environ["OCR_MODEL_PATH"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    deadline = time.monotonic() + MAX_DOWNLOAD_SECONDS
    digest = hashlib.sha256()
    print(f"Downloading OCR model to {path}", flush=True)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=20) as response, temporary.open("wb") as output:
            while chunk := response.read(64 * 1024):
                if time.monotonic() > deadline:
                    raise TimeoutError(f"OCR model download exceeded {MAX_DOWNLOAD_SECONDS} seconds")
                output.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != MODEL_SHA256:
            raise ValueError(f"OCR model checksum mismatch: {digest.hexdigest()}")
        os.replace(temporary, path)
        print("OCR model ready", flush=True)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
