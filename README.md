# mri-brain-tumors
Определение наличия и типов опухолей на мрт мозга
# Сегментация и классификация новообразований головного мозга по МРТ

Cистема для детекции и точной локализации опухолей головного мозга (глиома, менингиома, аденома гипофиза) по монохромным срезам МРТ.

[![Open Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SDEL3CeAGaQHK7YNh-pnwKLbk5V6PXVR?usp=sharing)
[![Open Full Pipeline in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12biIGSKEKvGF_6BlJyboRFQ5PzI0iJlq?usp=sharing)

---

## Быстрый запуск

Для оперативной проверки работы модели без ожидания обучения и скачивания датасетов:

1. Откройте блокнот **[test_ai.ipynb в Google Colab](https://colab.research.google.com/drive/1SDEL3CeAGaQHK7YNh-pnwKLbk5V6PXVR?usp=sharing)**.
2. В верхнем меню выберите **Среда выполнения (Runtime) -> Выполнить всё (Run all)**.
3. В выводе последней ячейки откроется веб-интерфейс Gradio с публичной ссылкой `*.gradio.live`.

---

# 🧠 Архитектура и технический разбор пайплайна обучения (Training Pipeline)

Данный документ содержит исчерпывающее техническое описание архитектуры, алгоритмов предварительной обработки данных, математического аппарата и протоколов обучения системы мультиклассовой диагностики патологий головного мозга по МРТ-сканам (`train_ai.ipynb` / `full_code_mri_ai.ipynb`).

---

## 📑 Содержание
1. [Общая архитектурная блок-схема (End-to-End Pipeline)](#1-общая-архитектурная-блок-схема-end-to-end-pipeline)
2. [Ячейка 1: Инфраструктура и загрузка сырых данных](#2-ячейка-1-инфраструктура-и-загрузка-сырых-данных)
3. [Ячейка 2: ETL, Санитизация, Фильтрация шумов и Дедупликация](#3-ячейка-2-etl-санитизация-фильтрация-шумов-и-дедупликация)
4. [Ячейка 3: Медицинский препроцессинг, Архитектуры сетей и Функции потерь](#4-ячейка-3-медицинский-препроцессинг-архитектуры-сетей-и-функции-потерь)
5. [Ячейка 4: Двухфазный протокол трансферного обучения классификатора](#5-ячейка-4-двухфазный-протокол-трансферного-обучения-классификатора)
6. [Ячейка 4.1: Калибровка входного фильтра Gatekeeper](#6-ячейка-41-калибровка-входного-фильтра-gatekeeper)
7. [Сводная таблица гиперпараметров и сигнатур](#7-сводная-таблица-гиперпараметров-и-сигнатур)

---

## 1. Общая архитектурная блок-схема (End-to-End Pipeline)

```mermaid
flowchart TD
    subgraph STAGE1["1. ETL, Очистка и Балансировка данных (Ячейки 1-2)"]
        A1["Google Drive Хранилище<br/>(mri_dataset.zip, not_mri_custom.zip, not_mri_nature.zip)"] -->|gdown.download| A2["ZIP-архивы в /content/"]
        A2 -->|safe_extract: декодирование кодировок, md5-пути| B1["Сырые файлы в RAW_DIR"]
        B1 --> C1{"is_valid_mri_quality<br/>Var(Laplacian) >= 15.0<br/>Entropy >= 2.5"}
        C1 -- Брак --> C2["Отфильтровано (Шум/Пустые кадры)"]
        C1 -- Валидно --> D1{"is_duplicate_and_register<br/>pHash 5 проекций (D4 группа)<br/>Hamming dist <= 2"}
        D1 -- Дубликат --> D2["Изоляция в /content/duplicates/"]
        D1 -- Уникален --> E1["Канонизация классов МРТ:<br/>['Глиома', 'Менингиома', 'Гипофиз', 'Опухоли нет']"]
        A2 -->|safe_extract| E2["Генерация OOD-класса 'Не МРТ'<br/>(Target: 2000 изображений)"]
        E1 & E2 --> F1["train_test_split (80% Train / 20% Val, seed=42)"]
        F1 --> F2["Структура CLEAN_DIR (/train, /val)"]
    end

    subgraph STAGE2["2. Предобработка и Тензорные датасеты (Ячейка 3)"]
        F2 --> G1["MedicalPreprocess:<br/>1. Порог бинаризации > 10 (Фон)<br/>2. Bounding Box кроп черепа (>1000 px)<br/>3. CLAHE (clip=1.2, grid=16x16)<br/>4. Letterbox Padding в 512x512 RGB"]
        G1 --> H1["data_transforms['train']:<br/>RandomFlip(0.5) + Rotation(10°) + RandomErasing(p=0.3) + Norm"]
        G1 --> H2["data_transforms['val']:<br/>ToTensor + ImageNet Normalize"]
        H1 & H2 --> I1["GatekeeperDataset & DataLoader<br/>(Batch=16, 2 класса)"]
        H1 & H2 --> I2["DiagnosticDataset & DataLoader<br/>(Batch=16, 4 класса)"]
    end

    subgraph STAGE3["3. Модели и Математический аппарат потерь (Ячейка 3)"]
        I1 --> J1["MobileNetV3-Small (Gatekeeper)<br/>Вход: (B, 3, 512, 512) -> Выход: (B, 2)"]
        J1 --> K1["Критерий: CrossEntropyLoss<br/>Оптимизатор: AdamW (lr=3e-4, wd=1e-3)"]
        I2 --> J2["ResNet-50 (Diagnostic Classifier)<br/>FC: Dropout(0.30) + Linear(2048 -> 4)"]
        J2 --> K2["Критерий: FocalLoss(gamma=2.0, alpha=[1.10, 1.30, 1.15, 0.65])<br/>AMP: GradScaler (FP16)"]
    end

    subgraph STAGE4["4. Обучение, Регуляризация и Сохранение (Ячейки 4, 4.1)"]
        K1 --> L1["Обучение Gatekeeper (3 эпохи, AMP)"]
        L1 --> L2["best_gatekeeper.pth"]
        K2 --> M1["Диагностика Фаза 1: Warmup Head (3 эпохи)<br/>Requires_grad: Backbone=False, Head=True<br/>AdamW(lr=3e-4), Grad Accumulation=2"]
        M1 --> M2["Диагностика Фаза 2: Deep Fine-Tuning (до 25 эпох)<br/>Дифференциальный AdamW: Backbone=1.2e-5, FC=8.0e-5<br/>Scheduler: CosineAnnealingWarmRestarts(T_0=5, T_mult=2)"]
        M2 --> M3{"Валидация: Расчет Macro F1"}
        M3 -->|Рекорд F1| M4["best_resnet50_mri_fixed.pth"]
        M3 -->|Patience = 7 эпох| M5["Ранняя остановка (Early Stopping)"]
        L2 -.-> N1["Ячейка 4.1: Калибровка Gatekeeper (2 эпохи)<br/>CrossEntropyLoss(weights=[1.5, 1.0])"]
        N1 --> N2["Обновленный best_gatekeeper.pth"]
    end
```

---

## 2. Ячейка 1: Инфраструктура и загрузка сырых данных

### 2.1. Назначение
Инициализирует окружение Google Colab, устанавливает актуальные версии библиотек компьютерного зрения и производит автоматическое скачивание архивов датасетов по прямым идентификаторам Google Drive.

```mermaid
flowchart LR
    A["Старт пайплайна"] --> B["Установка пакетов (albumentations, imagehash, opencv...)"]
    B --> C["Определение словаря DATASETS"]
    C --> D{"os.path.exists(dest_path)?"}
    D -- Нет -- > E["gdown.download(id, dest_path)"]
    D -- Да --> F["Пропуск загрузки"]
    E & F --> G["Архивы готовы к распаковке"]
```

### 2.2. Используемые переменные и структура словаря `DATASETS`
* `DATASETS` (`dict[str, str]`): маппинг локальных путей назначения и уникальных идентификаторов файлов Google Drive:
  * `"/content/mri_dataset.zip"` $\to$ `1fQh-pwXNEEOsVvvEz9xUZcPfsnVnhgSs` (основной мультиклассовый архив МРТ-сканов).
  * `"/content/not_mri_custom.zip"` $\to$ `1HNkX3akvRtW9Thkj8DmFeYOUpcL9CvbC` (пользовательские бытовые и фоновые снимки).
  * `"/content/not_mri_nature.zip"` $\to$ `1fGczBDMF3QO-GE55spClLMYD1htKN7jc` (природные текстуры и фотографии ImageNet).

---

## 3. Ячейка 2: ETL, Санитизация, Фильтрация шумов и Дедупликация

### 3.1. Архитектурная логика этапа
Медицинские датасеты из открытых источников страдают от трех системных проблем:
1. **Коллизии файловой системы**: поврежденные кодировки имен (кириллица DOS/Windows), метафайлы AppleDouble (`._*`, `__MACOSX`), спецсимволы и длинные пути, вызывающие `OSError: [Errno 36] File name too long`.
2. **Артефакты сканирования**: размытые сканы и пустые черные срезы без анатомических структур.
3. **Утечки данных (Data Leakage)**: идентичные срезы, повернутые на $90^\circ, 180^\circ$ или зеркально отраженные, попадающие одновременно в `train` и `val`.

```mermaid
flowchart TD
    A["RAW_DIR, CLEAN_DIR, DUPLICATES_DIR, TEMP_NOT_MRI"] --> B["safe_extract(ZIP_MRI, RAW_DIR)"]
    B --> C["Рекурсивная распаковка вложенных ZIP-архивов"]
    C --> D["Обход папок по именам через CLASS_CANONICAL"]
    D --> E["Итерация по графическим файлам (.jpg, .jpeg, .png)"]
    
    E --> F["is_valid_mri_quality(image_path)"]
    F -- "Var(L) < 15.0 или H < 2.5" --> G["Исключение: total_filtered_noise += 1"]
    F -- Качество подтверждено --> H["is_duplicate_and_register(image_path, threshold=2)"]
    
    H -- "Hamming Dist <= 2 к базе" --> I["Изоляция: duplicates_found += 1<br/>Перемещение в DUPLICATES_DIR"]
    H -- Уникальный снимок --> J["Регистрация pHash в seen_hashes<br/>Добавление в discovered_classes"]
    
    J --> K["Формирование класса 'Не МРТ' (ZIP_NATURE + ZIP_CUSTOM)"]
    K --> L["Ограничение TARGET_NOT_MRI_COUNT = 2000"]
    L --> M["train_test_split (test_size=0.2, random_state=42, shuffle=True)"]
    M --> N["Запись в /content/clean_dataset/train и /val"]
```

### 3.2. Детальный разбор функций

#### `safe_extract(zip_path, extract_to)`
* **Входные параметры**: `zip_path` (`str`), `extract_to` (`str`).
* **Логика работы**:
  * Игнорирует системный мусор: `__macosx` и файлы, начинающиеся с `._`.
  * Двухэтапная трансляция кодировок: `member.filename.encode('cp437').decode('cp866')`, при исключении — декодирование в `utf-8`.
  * Очистка регулярным выражением: `re.sub(r'[*?\"<>|]', '', filename).replace(':', '_')`.
  * MD5-хеширование сегментов путей длиннее 60 байт:
    $$part = f\_md5(part)[:10].ext$$

#### `is_valid_mri_quality(image_path)`
* **Входные параметры**: `image_path` (`str`).
* **Выходное значение**: `bool` (True — срез допущен к обучению).
* **Математический аппарат**:
  1. **Дисперсия дискретного оператора Лапласа** (оценка резкости границ):
     $$L(x, y) = \Delta I(x, y) = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$
     $$\text{Var}(L) = \frac{1}{N} \sum_{x,y} \left( L(x, y) - \mu_L \right)^2$$
     Порог отсечения: $\text{Var}(L) < 15.0$ (размытый кадр).
  2. **Энтропия Шеннона** (оценка информационной плотности гистограммы):
     $$p_i = \frac{h_i}{\sum_{k=0}^{255} h_k}, \quad H = -\sum_{i: p_i > 0} p_i \log_2(p_i)$$
     Порог отсечения: $H < 2.5$ (пустой шум сканера, монохромные артефакты).

#### `is_duplicate_and_register(image_path, hash_threshold=2)`
* **Входные параметры**: `image_path` (`str`), `hash_threshold` (`int = 2`).
* **Выходное значение**: `bool` (True — дубликат обнаружен).
* **Математический аппарат**:
  Для каждого изображения вычисляется перцептивный хеш на базе ДКП (`imagehash.phash`) по 5 базисным трансформациям диэдральной группы симметрий $D_4$:
  $$\mathcal{V}(I) = \left\{ \text{pHash}(I), \text{pHash}(I_{90^\circ}), \text{pHash}(I_{180^\circ}), \text{pHash}(I_{270^\circ}), \text{pHash}(I_{flip}) \right\}$$
  Расстояние Хэмминга между 64-битными векторами хешей:
  $$D_H(h_1, h_2) = \sum_{j=1}^{64} \left( h_1^{(j)} \oplus h_2^{(j)} \right)$$
  Если $\exists h \in \mathcal{V}(I), \exists h_{exist} \in seen\_hashes : D_H(h, h_{exist}) \le 2$, снимок отбраковывается в директорию `DUPLICATES_DIR`.

---

## 4. Ячейка 3: Медицинский препроцессинг, Архитектуры сетей и Функции потерь

### 4.1. Алгоритм класса `MedicalPreprocess`
В отличие от стандартного ресайза, приводящего к искажению геометрических пропорций новообразований, пайплайн выполняет адаптивное кадрирование паренхимы и Letterbox-центрирование:

```mermaid
flowchart TD
    A["PIL Image"] --> B["Конвертация в Gray"]
    B --> C["cv2.threshold(gray, 10, 255, THRESH_BINARY)"]
    C --> D["Поиск внешних контуров черепа"]
    D --> E{"Площадь контура > 1000 px?"}
    E -- Да --> F["Кадрирование по Bounding Box: img[y:y+h, x:x+w]"]
    E -- Нет --> G["Сохранение исходных границ"]
    F & G --> H["CLAHE: clipLimit=1.2, tileGridSize=(16, 16)"]
    H --> I["Конвертация в RGB"]
    I --> J["Вычисление масштабного коэффициента:<br/>scale = min(512/w, 512/h)"]
    J --> K["cv2.resize(INTER_AREA)"]
    K --> L["Создание черного холста 512x512x3"]
    L --> M["Центрированная вставка (Letterbox)"]
    M --> N["Выходной PIL Image (512x512)"]
```

### 4.2. Математическая формулировка `FocalLoss`
Для компенсации дисбаланса сложности патологий реализована взвешенная функция потерь Фокала:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

где $p_t = \exp(-\text{CE}(x, y))$, $\gamma = 2.0$.
Вектор весов классов $\alpha$:
$$\alpha = [\text{Гипофиз}: 1.10, \quad \text{Глиома}: 1.30, \quad \text{Менингиома}: 1.15, \quad \text{Опухоли нет}: 0.65]$$

> [!NOTE]
> Максимальный вес $\alpha = 1.30$ назначен глиоме вследствие ее инфильтративного характера роста, размытых границ и повышенной склонности к ложноотрицательным классификациям.

### 4.3. Архитектуры сверточных моделей

```mermaid
classDiagram
    class MobileNetV3_Gatekeeper {
        +Вход: Tensor (B, 3, 512, 512)
        +Backbone: Pretrained MobileNetV3-Small
        +Pooling: AdaptiveAvgPool2d(1, 1)
        +Classifier[0]: Linear(576, 1024)
        +Classifier[1]: Hardswish()
        +Classifier[2]: Dropout(p=0.20)
        +Classifier[3]: Linear(1024, 2)
        +Выход: Логиты (B, 2)
    }

    class ResNet50_Diagnostic {
        +Вход: Tensor (B, 3, 512, 512)
        +Backbone: Pretrained ResNet-50 (Conv1 -> Layer4)
        +Pooling: AdaptiveAvgPool2d(1, 1)
        +Flatten: Tensor (B, 2048)
        +FC[0]: Dropout(p=0.30)
        +FC[1]: Linear(2048, 4)
        +Выход: Логиты (B, 4)
    }
```

---

## 5. Ячейка 4: Двухфазный протокол трансферного обучения классификатора

Обучение диагностической сети ResNet-50 разделено на две изолированные фазы для сохранения низкоуровневых признаков сверточных фильтров.

```mermaid
flowchart TD
    subgraph Phase1["Фаза 1: Прогрев классификационной головы (3 эпохи)"]
        A1["Заморозка остова: param.requires_grad = False"] --> A2["Разморозка головы: model_diag.fc.requires_grad = True"]
        A2 --> A3["Оптимизатор: AdamW(model_diag.fc, lr=3e-4, wd=1e-3)"]
        A3 --> A4["Forward pass (FP16 AMP)"]
        A4 --> A5["loss = criterion_diag(outputs, labels) / 2"]
        A5 --> A6["Накопление градиентов (ACCUMULATION_STEPS = 2)"]
        A6 --> A7["scaler.step(optimizer_head) каждые 2 батча"]
    end

    subgraph Phase2["Фаза 2: Глубокая доводка всей сети (до 25 эпох)"]
        B1["Полная разморозка: все requires_grad = True"] --> B2["Дифференциальный AdamW:<br/>Backbone lr = 1.2e-5<br/>Classifier FC lr = 8.0e-5"]
        B2 --> B3["Scheduler: CosineAnnealingWarmRestarts<br/>T_0 = 5, T_mult = 2, eta_min = 1e-7"]
        B3 --> B4["Цикл эпох: Train и Val с AMP"]
        B4 --> B5["Расчет валидационного Macro F1-score"]
        B5 --> B6{"Macro F1 > best_diag_f1?"}
        B6 -- Да --> B7["best_diag_f1 = Macro F1<br/>Сохранение best_resnet50_mri_fixed.pth<br/>patience_counter = 0"]
        B6 -- Нет --> B8["patience_counter += 1"]
        B8 --> B9{"patience_counter >= 7?"}
        B9 -- Да --> B10["Early Stopping: Остановка обучения"]
        B9 -- Нет --> B4
    end

    Phase1 --> Phase2
```

### 5.1. Динамика шага обучения (Cosine Annealing with Warm Restarts)
Формула пересчета темпа обучения на шаге $t$:

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_i}\pi\right)\right)$$

* $T_0 = 5$ эпох (первый цикл рестарта).
* $T_{mult} = 2$ (длина каждого следующего цикла удваивается: $5 \to 10 \to 20$).
* $\eta_{min} = 10^{-7}$.

---

## 6. Ячейка 4.1: Калибровка входного фильтра Gatekeeper

### 6.1. Назначение
В клиническом контуре критически опасна ошибка пропуска немедицинского изображения (Out-of-Distribution) в диагностический модуль. Ячейка 4.1 производит дополнительную калибровку `MobileNetV3-Small` с асимметричным штрафом:

$$\mathcal{L}_{gate} = -\sum_{c=0}^{1} w_c \cdot y_c \log(p_c), \quad w = [w_{\text{Не МРТ}} = 1.5, \quad w_{\text{МРТ}} = 1.0]$$

```mermaid
flowchart LR
    A["Загрузка весов Gatekeeper"] --> B["Инициализация CrossEntropyLoss(weights=[1.5, 1.0])"]
    B --> C["Оптимизатор AdamW(lr=2e-4, wd=1e-3)"]
    C --> D["Обучение: 2 эпохи со смешанной точностью (AMP)"]
    D --> E["Оценка валидационной точности (Val Acc > 99.9%)"]
    E --> F["Перезапись чекпоинта 'best_gatekeeper.pth'"]
```

---

## 7. Сводная таблица гиперпараметров и сигнатур

| Компонент / Переменная | Тип / Размерность | Численное значение | Назначение в пайплайне |
| :--- | :--- | :--- | :--- |
| `TARGET_IMAGE_SIZE` | `tuple(int, int)` | `(512, 512)` | Входное пространственное разрешение сетей |
| `BATCH_SIZE` | `int` | `16` | Физический размер мини-батча на GPU |
| `ACCUMULATION_STEPS` | `int` | `2` | Виртуальный размер батча: $16 \times 2 = 32$ |
| `RandomErasing` | Атрибуты | `p=0.30, scale=(0.02, 0.15)` | Регуляризация устойчивости к локальным окклюзиям |
| `CLAHE` | Параметры | `clipLimit=1.2, grid=(16, 16)` | Нормализация контраста тканевых плотностей |
| `FocalLoss.gamma` | `float` | `2.0` | Степень подавления легких примеров |
| `diag_weights_tensor` | `torch.FloatTensor` | `[1.10, 1.30, 1.15, 0.65]` | Вектор балансировочных весов $\alpha$ |
| `lr (Backbone, Фаза 2)` | `float` | `1.2e-5` | Шаг сохранения низкоуровневых фильтров |
| `lr (Head FC, Фаза 2)` | `float` | `8.0e-5` | Шаг адаптации логитов классификатора |
| `patience` | `int` | `7` | Лимит эпох без роста Macro F1 (Early Stopping) |
| `best_gatekeeper.pth` | Файл весов PyTorch | `~9.3 MB` | Оптимальные веса фильтра Gatekeeper |
| `best_resnet50_mri_fixed.pth` | Файл весов PyTorch | `~94.4 MB` | Оптимальные веса диагностической модели |


## Формат выходных данных

При инференсе система формирует:
1. Визуализацию с наложенной маской патологии и bounding box.
2. Вероятностное распределение по всем классам.
3. Врачебную разметку в формате JSON (координаты полигонов и рамок).
