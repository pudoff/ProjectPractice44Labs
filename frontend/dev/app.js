'use strict';

// ===== Конфигурация =====
const CONFIG = {
  // Адрес бэкенда
  BASE_URL: 'http://localhost:8000',
  // Адрес для отправки файла на распознавание
  UPLOAD_ENDPOINT: '/api/v1/jobs',
  // Получение статуса задания на обработку
  STATUS_ENDPOINT: '/api/v1/jobs/{id}',
  // Адрес для скачивания готового файла dxf
  DOWNLOAD_ENDPOINT: '/api/v1/jobs/{id}/artifacts/{name}',
  MAX_FILE_SIZE: 25 * 1024 * 1024,
  ALLOWED_EXTENSIONS: ['pdf', 'jpeg', 'jpg', 'png', 'tif', 'tiff', 'bmp'],
  ALLOWED_MIME: [
    'application/pdf',
    'image/jpeg',
    'image/png',
    'image/tiff',
    'image/bmp',
  ],
};

// ===== DOM-элементы =====
const form = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const dropzone = document.getElementById('dropzone');
const fileInfo = document.getElementById('fileInfo');
const fileNameEl = document.getElementById('fileName');
const clearFileBtn = document.getElementById('clearFile');
const submitBtn = document.getElementById('submitBtn');
const errorMsg = document.getElementById('errorMsg');
const loader = document.getElementById('loader');
const resultBlock = document.getElementById('resultBlock');
const downloadLink = document.getElementById('downloadLink');
const resetBtn = document.getElementById('resetBtn');

/** @type {File|null} */
let selectedFile = null;
/** @type {string|null} */
let currentBlobUrl = null;

// ===== Управление состояниями =====
/**
 * Переключает состояние интерфейса.
 * @param {'idle' | 'loading' | 'success' | 'error'} state
 */
function setUIState(state) {
  // 1. Прячем всё
  form.style.display = 'none';
  loader.style.display = 'none';
  resultBlock.style.display = 'none';

  // 2. Снимаем атрибут hidden (на всякий случай)
  form.hidden = true;
  loader.hidden = true;
  resultBlock.hidden = true;

  // 3. Показываем нужное
  if (state === 'idle' || state === 'error') {
    form.style.display = 'flex';
    form.hidden = false;
  } else if (state === 'loading') {
    loader.style.display = 'flex';
    loader.hidden = false;
  } else if (state === 'success') {
    resultBlock.style.display = 'flex';
    resultBlock.hidden = false; // Обязательно снимаем
  }
}

// ===== Вспомогательные функции =====
function showError(text) {
  errorMsg.textContent = text;
  errorMsg.hidden = false;
}

function hideError() {
  errorMsg.hidden = true;
  errorMsg.textContent = '';
}

function validateFile(file) {
  const ext = (file.name.split('.').pop() || '').toLowerCase();
  if (!CONFIG.ALLOWED_EXTENSIONS.includes(ext)) {
    return `Неподдерживаемый формат файла (.${ext}). Разрешены: ${CONFIG.ALLOWED_EXTENSIONS.join(', ')}.`;
  }
  if (file.type && !CONFIG.ALLOWED_MIME.includes(file.type)) {
    return 'Неподдерживаемый тип файла.';
  }
  if (file.size > CONFIG.MAX_FILE_SIZE) {
    return `Файл слишком большой. Максимум — ${CONFIG.MAX_FILE_SIZE / 1024 / 1024} МБ.`;
  }
  if (file.size === 0) {
    return 'Файл пустой.';
  }
  return null;
}

function setFile(file) {
  hideError();

  if (!file) {
    // Если файла нет - прячем плашку
    selectedFile = null;
    fileInput.value = ''; // Очищаем инпут
    fileInfo.style.display = 'none'; // Скрываем
    submitBtn.disabled = true;
    return;
  }

  const error = validateFile(file);
  if (error) {
    showError(error);
    selectedFile = null;
    fileInfo.style.display = 'none'; // Скрываем
    submitBtn.disabled = true;
    return;
  }

  // Если файл валиден - показываем плашку
  selectedFile = file;
  fileNameEl.textContent = file.name;
  fileInfo.style.display = 'flex'; // Показываем (так как .file-info имеет display: flex в CSS)
  submitBtn.disabled = false;
}

function revokeCurrentBlobUrl() {
  if (currentBlobUrl) {
    URL.revokeObjectURL(currentBlobUrl);
    currentBlobUrl = null;
  }
}

// ===== Запрос к серверу =====
async function recognizeDrawing(file) {
  // 1. Отправляем файл для распознавания
  const formData = new FormData();
  formData.append('file', file);
  formData.append('channel', 'web');

  const uploadResponse = await fetch(CONFIG.BASE_URL + CONFIG.UPLOAD_ENDPOINT, {
    method: 'POST',
    body: formData,
  });

  if (!uploadResponse.ok) {
    throw new Error(`Ошибка загрузки (${uploadResponse.status}). Проверьте формат файла.`);
  }

  const uploadData = await uploadResponse.json();
  // Проверить расположение параметра.
  const jobId = uploadData.id || uploadData.job_id || uploadData.taskId; 

  if (!jobId) {
    console.error('Ответ сервера:', uploadData);
    throw new Error('Сервер не вернул ID задачи.');
  }

  console.log(`Задача создана. ID: ${jobId}. Начинаем опрос статуса...`);

  // 2. Опрашиваем статус
  const statusUrl = CONFIG.BASE_URL + CONFIG.STATUS_ENDPOINT.replace('{id}', jobId);
  let isDone = false;
  let attempts = 0;
  const maxAttempts = 90; // 90 попыток * 2 сек = 3 минуты, проверить, достаточно ли времени на обработку

  while (!isDone && attempts < maxAttempts) {
    attempts++;
    
    // Ждем 2 секунды перед каждой проверкой
    await new Promise((resolve) => setTimeout(resolve, 2000));

    try {
      const statusResponse = await fetch(statusUrl);
      
      if (!statusResponse.ok) {
        throw new Error(`Ошибка при проверке статуса (${statusResponse.status})`);
      }

      const statusData = await statusResponse.json();
      
      // Берем статус из поля "state", приводим к нижнему регистру на всякий случай
      const currentStatus = (statusData.state || '').toLowerCase();
      
      console.log(`Проверка ${attempts}. Текущий статус: ${currentStatus}`);

      // Если задача успешно завершена (или завершена с замечаниями)
      if (currentStatus === 'succeeded' || currentStatus === 'needs_review') {
        isDone = true;
        console.log('Задача выполнена!');
      } 
      // Если задача упала с ошибкой
      else if (currentStatus === 'failed') {
        throw new Error('При распознавании чертежа произошла ошибка.');
      }
      // Статусы 'queued' и 'running' игнорируем - просто ждем следующего круга
      
    } catch (error) {
      // Если сеть моргнула - выводим в консоль, но не прерываем цикл
      console.warn('Сбой при проверке статуса:', error.message);
    }
  }

  if (!isDone) {
    throw new Error('Превышено время ожидания. Сервер слишком долго обрабатывает файл.');
  }

  // 3. Формируем ссылку на скачивание
  const downloadUrl = CONFIG.BASE_URL + CONFIG.DOWNLOAD_ENDPOINT
    .replace('{id}', jobId)
    .replace('{name}', 'result.dxf'); 

  return { downloadUrl: downloadUrl };
}

// ===== Обработчики событий =====

fileInput.addEventListener('change', () => {
  const file = fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;
  setFile(file);
});

['dragenter', 'dragover'].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.add('dropzone--dragover');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove('dropzone--dragover');
  });
});

dropzone.addEventListener('drop', (e) => {
  const files = e.dataTransfer && e.dataTransfer.files;
  const file = files && files.length ? files[0] : null;
  setFile(file);
});

// Очистка файла (крестик)
clearFileBtn.addEventListener('click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  setFile(null);
});

// Кнопка "Загрузить другой файл"
resetBtn.addEventListener('click', () => {
  setFile(null);
  setUIState('idle');
  revokeCurrentBlobUrl();
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

// Отправка формы (Кнопка Распознать)
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  hideError();

  if (!selectedFile) {
    showError('Сначала выберите файл.');
    return;
  }

  setUIState('loading');

  try {
    // Ждём, пока функция сделает POST, подождет и сформирует GET-ссылку
    const { downloadUrl } = await recognizeDrawing(selectedFile);

    // Отменяем старую скачанную ссылку (если была), чтобы не забивать память
    revokeCurrentBlobUrl(); 
    
    // Настраиваем кнопку скачивания
    downloadLink.href = downloadUrl;
    downloadLink.setAttribute('download', 'result.dxf');
    
    setUIState('success');
    
  } catch (err) {
    console.error(err);
    showError(err instanceof Error ? err.message : 'Неизвестная ошибка.');
    setUIState('error');
  }
});

// ===== Тема =====
const themeToggle = document.getElementById('themeToggle');
const themeIcon = themeToggle.querySelector('.theme-toggle__icon');
const themeText = themeToggle.querySelector('.theme-toggle__text');

function applyTheme(theme) {
  if (theme === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
    themeIcon.textContent = '☀️';
    themeText.textContent = 'Светлая';
    localStorage.setItem('theme', 'light');
  } else {
    document.documentElement.removeAttribute('data-theme');
    themeIcon.textContent = '🌙';
    themeText.textContent = 'Тёмная';
    localStorage.setItem('theme', 'dark');
  }
}

const currentTheme = document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
applyTheme(currentTheme);

themeToggle.addEventListener('click', () => {
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  applyTheme(isLight ? 'dark' : 'light');
});

window.addEventListener('beforeunload', revokeCurrentBlobUrl);

// ===== ИНИЦИАЛИЗАЦИЯ =====
setUIState('idle');
