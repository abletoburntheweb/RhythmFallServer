### RhythmFallServer — инструкция по запуску Flask-сервера

Эта инструкция поможет запустить Flask-сервер для анализа BPM и генерации нот.

#### 1. Создание виртуального окружения

Выполните в папке `RhythmFallServer`:

```bash
python -m venv .venv
```

#### 2. Активация виртуального окружения

- **Windows (cmd):**
  ```cmd
  .venv\Scripts\activate
  ```
- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
#### 3. Установка зависимостей

Убедитесь, что виртуальное окружение активировано, затем:

```bash
pip install -r requirements.txt
```

#### 4. Запуск сервера

```bash
python run.py
```

Сервер запустится на `http://127.0.0.1:5000`.

#### 5. Проверка работы

Откройте в браузере: `http://127.0.0.1:5000`

Должно отобразиться: `{"message": "RhythmFallServer is running"}`

#### 6. Остановка сервера

В окне терминала с запущенным сервером нажмите `Ctrl+C`.

#### 7. Деактивация виртуального окружения

```bash
deactivate
```

---
