# FrontEnd

## Состав раздела:
**dev/** - файлы для запуска проекта в связке с BackEnd\
├── index.html - файл веб-страницы\
├── styles.css - настройки стилей страницы\
├── app.js - логика страницы


**demo/** - демонстрационные файлы с заглушками и макеты\
├── app.js - логика страницы\
├── index.html - файл веб-страницы\
├── mockups-light.html - макеты (светлая тема)\
├── mockups.html - макеты (тёмная тема)\
├── styles.css - настройки стилей страницы

## Настройка Backend для Python
```from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Разрешаем запросы с локального фронтенда
origins = [
    "http://localhost:3000",  # или другой порт, на котором будет фронт
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], //ТОЛЬКО ДЛЯ ТЕСТА, лучше указать реальный порт, откуда будут идти запросы
    allow_headers=["*"], //ТОЛЬКО ДЛЯ ТЕСТА, лучше указать реальный порт, откуда будут идти запросы
)
```
## Инструкция по запуску в VS Code:
1. Запустить backend.
2. Скачать расширение Live Server.
2. Открыть терминал в папке с проектом и выполнить ```python3 -m http.server 3000``` или ```python -m http.server 3000``` (для Windows).
2. Нажать правой кнопкой мыши на ```index.html``` -> «Open with Live Server».
3. Загрузить файл в формате 'pdf', 'jpeg', 'jpg', 'png', 'tif', 'tiff', 'bmp'.
4. Дождаться обработки.
5. Скачать файл dxf.

## Возможные ошибки:
| Текст ошибки | Решение |
|----------|----------|
| Ошибка CORS в консоли    | Проверить, что в allow_origins указан именно тот адрес, с которого открыт фронтенд (http://localhost:3000 или http://127.0.0.1:3000)  |
| 404 Not Found   | Проверить порт backend в UPLOAD_ENDPOINT (app.js, строка 6)  |
| Connection refused    | Проверить, запущен ли backend и соответствует ли порт значению в в UPLOAD_ENDPOINT (app.js, строка 6)   |
