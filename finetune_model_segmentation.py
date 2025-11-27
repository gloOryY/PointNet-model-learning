"""

ДООБУЧЕНИЕ ДЛЯ TESLA T4 - ВЕРСИЯ С POINT-WISE SEGMENTATION

Главное отличие от classification:
- Модель предсказывает label для КАЖДОЙ точки отдельно (не для всего облака)
- Архитектура PointNet Segmentation с per-point classifier
- Loss и метрики считаются для каждой из 2048 точек

Совместимо с generate_detailed_city_scenes_v2.py
"""

# ============================================================================
# БЛОК 1: ИМПОРТ БИБЛИОТЕК И ИНИЦИАЛИЗАЦИЯ
# ============================================================================

import numpy as np
# numpy (np) — библиотека для работы с многомерными массивами

import torch
# torch — главная библиотека PyTorch для работы с тензорами и нейросетями

import torch.nn as nn
# nn (neural network) — модуль для определения архитектур нейросетей

import torch.nn.functional as F
# F (functional) — функции активации и потерь

from torch.utils.data import Dataset, DataLoader
# Dataset, DataLoader — утилиты для загрузки данных

import glob
# glob — поиск файлов по маскам

from tqdm import tqdm
# tqdm — визуализация прогресса обучения

from plyfile import PlyData
# plyfile — библиотека для чтения PLY файлов

import time
# time — модуль для работы со временем

import os
# os — работа с операционной системой

print("=" * 80)
print("POINTNET SEGMENTATION - ОБУЧЕНИЕ")
print("=" * 80)

# ============================================================================
# БЛОК 2: НАСТРОЙКА ПАРАМЕТРОВ ОБУЧЕНИЯ
# ============================================================================

# ========== ГИПЕРПАРАМЕТРЫ ОБУЧЕНИЯ ==========

BATCH_SIZE = 16
# НОВОЕ: Уменьшено с 32 до 16, потому что segmentation требует больше памяти
# Каждое облако теперь возвращает 2048 labels вместо одного

NUM_POINTS = 2048
# Количество точек в облаке (стандарт для PointNet)

NUM_EPOCHS = 20
# Количество эпох обучения

LEARNING_RATE = 0.0001
# Скорость обучения

VAL_SPLIT = 0.2
# Доля валидации (20%)

USE_MIXED_PRECISION = True
# Смешанная точность для ускорения

NUM_WORKERS = 2
# Потоки для загрузки данных

# ========== ПУТИ К ДАННЫМ ==========

MIXED_DATA_FOLDER = "dataset/mixed"
# Папка с PLY-файлами со смешанными labels

BASE_MODEL = "best_model_segmentation.pth"
# НОВОЕ: Другое имя файла для segmentation модели

# ============================================================================
# БЛОК 3: ВЫБОР УСТРОЙСТВА
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\nDevice: {device}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f} GB")

print(f"\nПараметры: BATCH={BATCH_SIZE}, WORKERS={NUM_WORKERS}, EPOCHS={NUM_EPOCHS}")

# ============================================================================
# БЛОК 4: ОПРЕДЕЛЕНИЕ АРХИТЕКТУРЫ TNet
# ============================================================================

class TNet(nn.Module):
    """
    TNet (Transformation Network) — вспомогательная сеть для PointNet
    Предсказывает матрицу трансформации для точек облака
    """

    def __init__(self, k=3):
        """
        Args:
            k (int): размерность входных данных (3 для xyz, 64 для признаков)
        """
        super(TNet, self).__init__()
        self.k = k

        # Сверточные слои
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # Полносвязные слои
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)

        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        """
        Прямой проход TNet

        Args:
            x: (batch, k, num_points)

        Returns:
            Матрица трансформации (batch, k, k)
        """
        batch_size = x.size(0)

        # Encoder: извлечение признаков
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global Max Pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)

        # Decoder: предсказание матрицы
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Инициализация единичной матрицей
        identity = torch.eye(self.k, device=x.device).flatten()
        x = x + identity
        x = x.view(batch_size, self.k, self.k)

        return x

# ============================================================================
# БЛОК 5: ОПРЕДЕЛЕНИЕ АРХИТЕКТУРЫ POINTNET SEGMENTATION
# ============================================================================

class PointNetSegmentation(nn.Module):
    """
    НОВАЯ АРХИТЕКТУРА: PointNet для сегментации

    Главное отличие от classification:
    - Возвращает labels для КАЖДОЙ точки (batch, num_points)
    - Использует комбинацию локальных и глобальных признаков
    - Per-point classifier вместо global classifier
    """

    def __init__(self, num_classes=2):
        """
        Args:
            num_classes: количество классов (2 для здание/не-здание)
        """
        super(PointNetSegmentation, self).__init__()

        # === INPUT TRANSFORM ===
        self.input_transform = TNet(k=3)

        # === ПЕРВАЯ ГРУППА СВЁРТОК (ЛОКАЛЬНЫЕ ПРИЗНАКИ) ===
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        # НОВОЕ: Третий слой теперь 128 (не 1024), так как нам нужны локальные признаки

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)

        # === FEATURE TRANSFORM ===
        self.feature_transform = TNet(k=64)

        # === ВТОРАЯ ГРУППА СВЁРТОК (ГЛУБОКИЕ ПРИЗНАКИ) ===
        self.conv4 = nn.Conv1d(128, 512, 1)
        self.conv5 = nn.Conv1d(512, 2048, 1)
        # НОВОЕ: Увеличиваем до 2048 для более богатых глобальных признаков

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)

        # === PER-POINT SEGMENTATION HEAD ===
        # НОВОЕ: Комбинируем локальные и глобальные признаки для каждой точки
        # Вход: 64 (локальные после conv1) + 2048 (глобальные) = 2112
        self.conv6 = nn.Conv1d(2112, 512, 1)
        self.conv7 = nn.Conv1d(512, 256, 1)
        self.conv8 = nn.Conv1d(256, 128, 1)
        self.conv9 = nn.Conv1d(128, num_classes, 1)
        # Выход: (batch, num_classes, num_points)

        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        """
        Прямой проход PointNet Segmentation

        Args:
            x: (batch, 3, num_points) — входное облако

        Returns:
            tuple:
                - logits: (batch, num_classes, num_points) — предсказания для каждой точки
                - trans: (batch, 3, 3) — матрица input transform
                - trans_feat: (batch, 64, 64) — матрица feature transform
        """
        batch_size = x.size(0)
        num_points = x.size(2)

        # === STEP 1: INPUT TRANSFORM ===
        trans = self.input_transform(x)
        x = torch.bmm(trans, x)

        # === STEP 2: ЛОКАЛЬНЫЕ ПРИЗНАКИ ===
        x = F.relu(self.bn1(self.conv1(x)))
        # ВАЖНО: Сохраняем локальные признаки после первого слоя
        local_features = x  # (batch, 64, num_points)

        # === STEP 3: FEATURE TRANSFORM ===
        trans_feat = self.feature_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)

        # === STEP 4: ГЛУБОКИЕ ЛОКАЛЬНЫЕ ПРИЗНАКИ ===
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # === STEP 5: ГЛОБАЛЬНЫЕ ПРИЗНАКИ ===
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Global Max Pooling: агрегация в глобальный вектор
        global_features = torch.max(x, 2, keepdim=True)[0]  # (batch, 2048, 1)

        # Размножить глобальные признаки для каждой точки
        global_features = global_features.repeat(1, 1, num_points)  # (batch, 2048, num_points)

        # === STEP 6: ОБЪЕДИНЕНИЕ ЛОКАЛЬНЫХ И ГЛОБАЛЬНЫХ ПРИЗНАКОВ ===
        # КЛЮЧЕВАЯ ИДЕЯ SEGMENTATION: каждая точка получает:
        # - Свои локальные признаки (что это за точка)
        # - Глобальный контекст (в каком облаке она находится)
        x = torch.cat([local_features, global_features], dim=1)  # (batch, 2112, num_points)

        # === STEP 7: PER-POINT CLASSIFICATION ===
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dropout(x)
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.dropout(x)
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.conv9(x)  # (batch, num_classes, num_points)

        # НОВОЕ: Не применяем softmax здесь, CrossEntropyLoss сделает это сам

        return x, trans, trans_feat

# ============================================================================
# БЛОК 6: ОПРЕДЕЛЕНИЕ ДАТАСЕТА СО СМЕШАННЫМИ LABELS
# ============================================================================


    """
    МОДИФИЦИРОВАННЫЙ ДАТАСЕТ для segmentation

    Главное отличие:
    - Возвращает labels для КАЖДОЙ точки (num_points,)
    - Не использует majority voting
    """

    def __init__(self, folder_path, num_points=2048, augment=True):
        """
        Args:
            folder_path: путь к папке с PLY файлами
            num_points: количество точек (2048)
            augment: применять ли аугментацию
        """
        self.num_points = num_points
        self.augment = augment

        print(f"\n📦 Загрузка датасета для segmentation:")

        self.files = glob.glob(os.path.join(folder_path, "*.ply"))

        if len(self.files) == 0:
            print(f"❌ Нет .ply файлов в {folder_path}!")
        else:
            print(f"  ✅ Найдено: {len(self.files)} файлов")

        self.total_samples = len(self.files)

    def _read_ply_with_labels(self, file_path):
        """
        Читает .ply файл с labels для каждой точки

        Returns:
            tuple: (points, labels) или (None, None)
        """
        try:
            plydata = PlyData.read(file_path)
            vertex = plydata['vertex']

            # Чтение координат
            points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

            # Чтение labels
            labels = None
            for label_field in ['label', 'class', 'classification', 'scalar_label']:
                if label_field in vertex.dtype.names:
                    labels = np.array(vertex[label_field])
                    break

            if labels is None:
                print(f"⚠️ Файл {os.path.basename(file_path)} не содержит поля label!")
                return None, None

            # Бинаризация labels
            labels = (labels == 1).astype(np.int64)

            return points, labels

        except Exception as e:
            print(f"⚠️ Ошибка чтения {os.path.basename(file_path)}: {e}")
            return None, None

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        МОДИФИЦИРОВАН для segmentation

        Returns:
            tuple:
                - points: (3, num_points) — облако точек
                - labels: (num_points,) — label для КАЖДОЙ точки
        """
        file_path = self.files[idx]
        points, labels = self._read_ply_with_labels(file_path)

        # Обработка ошибок
        if points is None or labels is None or len(points) == 0:
            points = np.random.randn(self.num_points, 3).astype(np.float32)
            labels = np.zeros(self.num_points, dtype=np.int64)

        # === НОРМАЛИЗАЦИЯ ===
        centroid = np.mean(points, axis=0)
        points = points - centroid

        m = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if m > 1e-8:
            points = points / m

        # === АУГМЕНТАЦИЯ ===
        if self.augment:
            # Поворот
            theta = np.random.uniform(0, 2 * np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rotation = np.array([
                [cos_t, -sin_t, 0],
                [sin_t, cos_t, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            points = points @ rotation.T

            # Масштаб
            points *= np.random.uniform(0.8, 1.2)

            # Шум
            points += np.random.normal(0, 0.02, points.shape).astype(np.float32)

        # === RESAMPLING ===
        if len(points) >= self.num_points:
            idx_sample = np.random.choice(len(points), self.num_points, replace=False)
        else:
            idx_sample = np.random.choice(len(points), self.num_points, replace=True)

        points = points[idx_sample]
        labels = labels[idx_sample]

        # === КОНВЕРТАЦИЯ В ТЕНЗОР ===
        points_tensor = torch.from_numpy(points.T.copy()).float()  # (3, num_points)
        labels_tensor = torch.from_numpy(labels.copy()).long()      # (num_points,)

        # НОВОЕ: Возвращаем labels для КАЖДОЙ точки, не majority label
        return points_tensor, labels_tensor
    
class SegmentationDataset(Dataset):
    """
    Датасет для point-wise segmentation.

    Главное:
    - Возвращает labels для КАЖДОЙ точки (num_points,)
    - Корректно читает поле label из PLY через vertex.data.dtype.names
    """

    def __init__(self, folder_path, num_points=2048, augment=True):
        self.num_points = num_points
        self.augment = augment

        print(f"\n📦 Загрузка датасета для segmentation:")

        self.files = glob.glob(os.path.join(folder_path, "*.ply"))

        if len(self.files) == 0:
            print(f"❌ Нет .ply файлов в {folder_path}!")
        else:
            print(f"  ✅ Найдено: {len(self.files)} файлов")

        self.total_samples = len(self.files)

    def _read_ply_with_labels(self, file_path):
        """
        Читает .ply файл с координатами и labels для каждой точки.

        Возвращает:
            points: np.ndarray (N, 3)
            labels: np.ndarray (N,)  (0/1)
        """
        try:
            # Важно: PlyData.read можно вызывать прямо по пути к файлу
            plydata = PlyData.read(file_path)
            vertex = plydata['vertex']          # PlyElement
            data = vertex.data                  # np.recarray с полями

            # Координаты
            points = np.vstack([
                data['x'],
                data['y'],
                data['z']
            ]).T.astype(np.float32)

            # Имена всех полей из vertex.data
            field_names = data.dtype.names

            # Поиск поля с метками
            labels = None
            for label_field in ['label', 'class', 'classification', 'scalar_label']:
                if label_field in field_names:
                    labels = np.array(data[label_field], dtype=np.int64)
                    break

            if labels is None:
                print(f"⚠️ Файл {os.path.basename(file_path)} не содержит поля label!")
                return None, None

            # Бинаризация: здания/стены = 1, всё остальное = 0
            labels = (labels == 1).astype(np.int64)

            # Защита от рассинхрона длин
            if len(points) != len(labels):
                print(f"⚠️ {os.path.basename(file_path)}: points({len(points)}) != labels({len(labels)})")
                n = min(len(points), len(labels))
                points = points[:n]
                labels = labels[:n]

            return points, labels

        except Exception as e:
            print(f"⚠️ Ошибка чтения {os.path.basename(file_path)}: {e}")
            return None, None

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        Возвращает:
            points: (3, num_points)
            labels: (num_points,)
        """
        file_path = self.files[idx]
        points, labels = self._read_ply_with_labels(file_path)

        # Если не удалось прочитать — генерируем заглушку
        if points is None or labels is None or len(points) == 0:
            points = np.random.randn(self.num_points, 3).astype(np.float32)
            labels = np.zeros(self.num_points, dtype=np.int64)

        # Нормализация
        centroid = np.mean(points, axis=0)
        points = points - centroid

        m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if m > 1e-8:
            points = points / m

        # Аугментации
        if self.augment:
            # Поворот вокруг Z
            theta = np.random.uniform(0, 2 * np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rotation = np.array([
                [cos_t, -sin_t, 0],
                [sin_t,  cos_t, 0],
                [0,      0,     1]
            ], dtype=np.float32)
            points = points @ rotation.T

            # Масштаб
            points *= np.random.uniform(0.8, 1.2)

            # Шум
            points += np.random.normal(0, 0.02, points.shape).astype(np.float32)

        # Resampling
        if len(points) >= self.num_points:
            idx_sample = np.random.choice(len(points), self.num_points, replace=False)
        else:
            idx_sample = np.random.choice(len(points), self.num_points, replace=True)

        points = points[idx_sample]
        labels = labels[idx_sample]

        # В тензоры
        points_tensor = torch.from_numpy(points.T.copy()).float()  # (3, num_points)
        labels_tensor = torch.from_numpy(labels.copy()).long()     # (num_points,)

        return points_tensor, labels_tensor

# ============================================================================
# БЛОК 7: ФУНКЦИИ ОБУЧЕНИЯ И ВАЛИДАЦИИ
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device, use_amp, epoch):
    """
    Одна эпоха обучения для segmentation

    ИЗМЕНЕНО: метрики считаются для КАЖДОЙ точки
    """
    model.train()

    total_loss, correct, total = 0.0, 0, 0
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} Train")

    for points, labels in pbar:
        # points: (batch, 3, num_points)
        # labels: (batch, num_points)

        points, labels = points.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast(enabled=True):
                outputs, trans, trans_feat = model(points)
                # outputs: (batch, num_classes, num_points)

                # НОВОЕ: Loss считается для КАЖДОЙ точки
                # Нужно переставить размерности для CrossEntropyLoss
                outputs = outputs.transpose(2, 1).contiguous()  # (batch, num_points, num_classes)
                outputs = outputs.view(-1, outputs.size(-1))     # (batch*num_points, num_classes)
                labels_flat = labels.view(-1)                    # (batch*num_points,)

                loss = criterion(outputs, labels_flat)

                # Регуляризация
                if trans_feat is not None:
                    k = trans_feat.size(1)
                    I = torch.eye(k, device=device).unsqueeze(0).repeat(trans_feat.size(0), 1, 1)
                    reg_loss = F.mse_loss(torch.bmm(trans_feat, trans_feat.transpose(2, 1)), I)
                    loss = loss + 0.001 * reg_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            outputs, trans, trans_feat = model(points)
            outputs = outputs.transpose(2, 1).contiguous()
            outputs = outputs.view(-1, outputs.size(-1))
            labels_flat = labels.view(-1)

            loss = criterion(outputs, labels_flat)

            if trans_feat is not None:
                k = trans_feat.size(1)
                I = torch.eye(k, device=device).unsqueeze(0).repeat(trans_feat.size(0), 1, 1)
                reg_loss = F.mse_loss(torch.bmm(trans_feat, trans_feat.transpose(2, 1)), I)
                loss = loss + 0.001 * reg_loss

            loss.backward()
            optimizer.step()

        # === МЕТРИКИ ===
        _, pred = torch.max(outputs, 1)  # (batch*num_points,)

        total += labels_flat.size(0)  # Всего точек
        correct += (pred == labels_flat).sum().item()  # Правильно классифицированных точек

        total_loss += loss.item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})

    return total_loss / len(loader), 100 * correct / total


def validate_epoch(model, loader, criterion, device, epoch):
    """
    Одна эпоха валидации для segmentation
    """
    model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} Val")

        for points, labels in pbar:
            points, labels = points.to(device), labels.to(device)

            outputs, _, _ = model(points)

            outputs = outputs.transpose(2, 1).contiguous()
            outputs = outputs.view(-1, outputs.size(-1))
            labels_flat = labels.view(-1)

            loss = criterion(outputs, labels_flat)

            _, pred = torch.max(outputs, 1)

            total += labels_flat.size(0)
            correct += (pred == labels_flat).sum().item()

            total_loss += loss.item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})

    return total_loss / len(loader), 100 * correct / total

# ============================================================================
# БЛОК 8: ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Главная функция: полный цикл обучения PointNet Segmentation"""

    # Проверка наличия данных
    if not os.path.exists(MIXED_DATA_FOLDER):
        print(f"❌ {MIXED_DATA_FOLDER} не найден!"); return

    print(f"\n✅ Папка с данными: {MIXED_DATA_FOLDER}/")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Загрузка датасета
    dataset = SegmentationDataset(MIXED_DATA_FOLDER, NUM_POINTS, True)

    if dataset.total_samples == 0:
        print("❌ Нет данных!"); return

    # Разделение train/val
    train_size = int((1-VAL_SPLIT)*len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])

    # DataLoaders
    train_loader = DataLoader(
        train_ds, BATCH_SIZE, True,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, BATCH_SIZE, False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # НОВАЯ МОДЕЛЬ: PointNet Segmentation
    model = PointNetSegmentation(2).to(device)
    print(f"\n🧠 Модель: PointNet Segmentation (point-wise classification)")

    # Попытка загрузить предыдущую модель
    if os.path.exists(BASE_MODEL):
        try:
            state = torch.load(BASE_MODEL, map_location=device)
            if isinstance(state, dict) and 'state_dict' in state:
                model.load_state_dict(state['state_dict'])
            else:
                model.load_state_dict(state)
            print(f"\n{'='*80}\n🔄 ДООБУЧЕНИЕ: {BASE_MODEL} загружен\n{'='*80}")
        except Exception as e:
            print(f"\n⚠️ Ошибка загрузки: {e}")
    else:
        print(f"\n⚠️ {BASE_MODEL} не найден - обучение с нуля")

    # Оптимизатор и планировщик
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    start = time.time()

    print(f"\n🚀 НАЧАЛО: {len(train_ds)} train, {len(val_ds)} val\n")

    # Главный цикл эпох
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'─'*80}\nEpoch {epoch+1}/{NUM_EPOCHS}\n{'─'*80}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, USE_MIXED_PRECISION, epoch)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, epoch)

        scheduler.step()

        print(f"\n📊 Train: {train_loss:.4f}, {train_acc:.2f}% | Val: {val_loss:.4f}, {val_acc:.2f}%")

        # Сохранение лучшей модели
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model_segmentation.pth')
            print(f"💾 Сохранено! Best: {best_acc:.2f}%")

    elapsed = (time.time() - start) / 60

    print(f"\n{'='*80}\n✅ ЗАВЕРШЕНО: {best_acc:.2f}%, {elapsed:.1f}мин\n{'='*80}")
    print(f"\n📁 best_model_segmentation.pth")
    print(f"\n🎯 Теперь модель классифицирует КАЖДУЮ точку отдельно!")
    print(f"   Вместо: облако → здание/не-здание")
    print(f"   Теперь: каждая точка → здание/не-здание")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================================
# БЛОК 9: ТОЧКА ВХОДА
# ============================================================================

if __name__ == "__main__":
    main()
