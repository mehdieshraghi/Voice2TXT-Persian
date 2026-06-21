# Voice2TXT-Persian

تبدیل گفتار فارسی به متن — **آفلاین** با [Vosk](https://alphacephei.com/vosk/).

این پروژه دو لایه دارد:

| لایه | مسیر | توضیح |
|------|------|--------|
| **هسته (کتابخانه)** | `voice2txt/` | مستقل از Flask — قابل استفاده در CLI، API، یا برنامهٔ خودتان |
| **رابط وب (اختیاری)** | `app/` | Flask UI برای تنظیمات، آپلود فایل، و ضبط میکروفون |

## پیش‌نیازها

- Python 3.10+
- میکروفون (برای ضبط)
- مدل Vosk فارسی

### دانلود مدل

**روش خودکار (پیشنهادی):**

```bash
# CLI — با پرسش
python cli.py

# CLI — بدون پرسش
python cli.py -y

# CLI — مدل مشخص
python cli.py --install-model vosk-model-small-fa-0.42
```

در **رابط وب**، اگر مدل نباشد پنجرهٔ نصب باز می‌شود و می‌توانید دانلود را تأیید کنید.

**روش دستی:**

1. از [Vosk Models](https://alphacephei.com/vosk/models) مدل **`vosk-model-small-fa-0.42`** (یا نسخهٔ جدیدتر از کاتالوگ) را دانلود کنید.
2. فایل zip را extract کنید.
3. پوشهٔ مدل را در ریشهٔ پروژه قرار دهید:

```
Voice2TXT-Persian/
  vosk-model-small-fa-0.42/
    am/
    graph/
    ...
```

### کاتالوگ مدل‌ها (به‌روزرسانی بدون تغییر کد)

لیست مدل‌های فارسی در `voice2txt/data/models.catalog.json` است. URL دانلود از الگوی زیر ساخته می‌شود:

```
https://alphacephei.com/vosk/models/{model_id}.zip
```

برای دریافت لیست جدیدتر بدون انتشار نسخهٔ جدید برنامه، می‌توانید در `settings.json` فیلد `models_catalog_url` را به یک JSON خام روی GitHub (مثلاً همان فایل کاتالوگ در repo) تنظیم کنید.

## نصب

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

تنظیمات (اختیاری):

```bash
copy settings.example.json settings.json
```

## اجرا

### رابط وب (Flask)

```bash
python run_web.py
```

مرورگر: http://127.0.0.1:5000

امکانات UI:
- آپلود فایل WAV
- ضبط با میکروفون **مرورگر** (client-side → WAV 16kHz)
- ضبط با میکروفون **سرور** (ماشینی که Flask روی آن اجراست)
- مدیریت تنظیمات (مسیر مدل، پوشه خروجی، ...)
- نمایش، کپی، و ذخیرهٔ متن

### خط فرمان (CLI)

```bash
# ضبط ۵ ثانیه و تبدیل
python cli.py

# تبدیل فایل موجود
python cli.py --file audio.wav

# مدت ضبط دلخواه
python cli.py --duration 10

# لیست میکروفون‌ها
python cli.py --list-devices

# بدون ذخیره خودکار
python cli.py --no-save
```

## ساختار پروژه

```
Voice2TXT-Persian/
├── voice2txt/           # هسته — بدون وابستگی به Flask
│   ├── config.py        # Settings (JSON load/save)
│   ├── audio.py         # نرمال‌سازی WAV برای Vosk
│   ├── recorder.py      # ضبط میکروفون (sounddevice)
│   ├── transcriber.py   # Vosk + cache مدل
│   └── storage.py       # ذخیره transcript
├── app/                 # Flask UI (قابل جایگزینی)
│   ├── routes.py
│   ├── templates/
│   └── static/
├── cli.py               # ورودی CLI
├── run_web.py           # ورودی Flask
├── settings.example.json
└── code.py              # legacy → cli.py
```

## استفاده در کد خودتان

هستهٔ `voice2txt` مستقل است. Flask فقط یک adapter نمونه است.

```python
from voice2txt import Settings, Transcriber, Recorder, save_transcript

settings = Settings.load("settings.json")
transcriber = Transcriber(settings)

# از فایل
text = transcriber.transcribe_file("speech.wav")

# از ضبط میکروفون
recorder = Recorder(settings)
wav_bytes = recorder.record(duration=5)
text = transcriber.transcribe_wav(wav_bytes)

save_transcript(text, settings)
```

### Flask سفارشی

```python
from app import create_app
from voice2txt import Settings, Transcriber

settings = Settings.load()
app = create_app(settings=settings)

# یا routeهای خودتان را به app اضافه کنید
@app.route("/my-api")
def my_api():
    t = Transcriber(settings)
    return {"ready": t.is_model_ready()}
```

### جایگزین کردن Transcriber

کلاس `Transcriber` را subclass کنید یا adapter خودتان را بنویسید — فقط interface زیر را رعایت کنید:

- `transcribe_wav(source: bytes | Path) -> str`
- `transcribe_file(path: Path) -> str`
- `is_model_ready() -> bool`

## تنظیمات (`settings.json`)

| کلید | پیش‌فرض | توضیح |
|------|---------|--------|
| `model_path` | `vosk-model-small-fa-0.42` | مسیر پوشه مدل |
| `sample_rate` | `16000` | نرخ نمونه‌برداری |
| `record_duration` | `5` | مدت ضبط پیش‌فرض (ثانیه) |
| `output_dir` | `output` | پوشه ذخیره transcript |
| `chunk_frames` | `4000` | اندازه chunk برای Vosk |
| `max_upload_mb` | `25` | حداکثر آپلود در وب |
| `models_catalog_url` | `""` | URL اختیاری JSON کاتالوگ مدل‌ها |

## مجوز

[CC BY-NC 4.0](LICENSE) — استفاده غیرتجاری با ذکر منبع.

برای مجوز تجاری با maintainer تماس بگیرید.

## مشارکت

Issue و PR خوش‌آمد است. برای UI سفارشی، `voice2txt/` را import کنید و `app/` را نادیده بگیرید.
