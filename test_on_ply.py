"""
Тестирование обученной PointNet Segmentation на файле data.ply
Классифицирует КАЖДУЮ точку как здание (1) / не-здание (0)
+ РАСЧЕТ МЕТРИК КАЧЕСТВА (если есть ground truth)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement  # ← Используется для чтения И записи
from tqdm import tqdm
import time


print("=" * 80)
print("ТЕСТИРОВАНИЕ POINTNET SEGMENTATION НА DATA.PLY")
print("=" * 80)


# === ПАРАМЕТРЫ ===
INPUT_FILE = "test.ply"  # ← Изменено на .ply
OUTPUT_FILE = "data_classified.ply"
MODEL_PATH = "best_model_segmentation.pth"
NUM_POINTS = 2048
BATCH_SIZE = 32


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ============================================================================
# АРХИТЕКТУРА (без изменений)
# ============================================================================

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k,   64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128,1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512,  256)
        self.fc3 = nn.Linear(256,  k*k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        identity = torch.eye(self.k, device=x.device).flatten()
        x = x + identity
        x = x.view(batch_size, self.k, self.k)
        return x


class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNetSegmentation, self).__init__()
        self.input_transform = TNet(k=3)
        self.conv1 = nn.Conv1d(3,   64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128,128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.feature_transform = TNet(k=64)
        self.conv4 = nn.Conv1d(128, 512,  1)
        self.conv5 = nn.Conv1d(512, 2048, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.conv6 = nn.Conv1d(2112, 512, 1)
        self.conv7 = nn.Conv1d(512,  256, 1)
        self.conv8 = nn.Conv1d(256,  128, 1)
        self.conv9 = nn.Conv1d(128,  num_classes, 1)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        trans = self.input_transform(x)
        x = torch.bmm(trans, x)
        x = F.relu(self.bn1(self.conv1(x)))
        local_features = x
        trans_feat = self.feature_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        global_features = torch.max(x, 2, keepdim=True)[0]
        global_features = global_features.repeat(1, 1, num_points)
        x = torch.cat([local_features, global_features], dim=1)
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dropout(x)
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.dropout(x)
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.conv9(x)
        return x, trans, trans_feat


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def normalize_point_cloud(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    m = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points = points / (m + 1e-8)
    return points


def segment_full_cloud(model, points, num_points, batch_size):
    N = len(points)
    labels = np.zeros(N, dtype=np.int32)
    confidences = np.zeros(N, dtype=np.float32)
    
    chunks = []
    for start in range(0, N, num_points):
        end = min(start + num_points, N)
        if end - start == 0:
            continue
        chunks.append((start, end))
    
    print(f"\nВсего точек: {N:,}")
    print(f"Чанк размером: {num_points}, всего чанков: {len(chunks)}")
    
    model.eval()
    
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(chunks), batch_size), desc="Сегментация"):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            batch_points = []
            batch_meta = []
            
            for (s, e) in batch_chunks:
                chunk_pts = points[s:e]
                chunk_norm = normalize_point_cloud(chunk_pts.copy())
                current = len(chunk_norm)
                if current < num_points:
                    idx = np.random.choice(current, num_points, replace=True)
                else:
                    idx = np.arange(current)[:num_points]
                
                chunk_resampled = chunk_norm[idx]
                batch_points.append(chunk_resampled)
                batch_meta.append((s, e, current))
            
            batch_tensor = torch.FloatTensor(np.array(batch_points)).to(device)
            batch_tensor = batch_tensor.transpose(2, 1).contiguous()
            
            logits, _, _ = model(batch_tensor)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            confs, _ = torch.max(probs, dim=1)
            
            preds_cpu = preds.cpu().numpy()
            confs_cpu = confs.cpu().numpy()
            
            for i, (s, e, real_n) in enumerate(batch_meta):
                labels[s:e] = preds_cpu[i, :real_n]
                confidences[s:e] = confs_cpu[i, :real_n]
    
    return labels, confidences


def calculate_metrics(gt_labels, pred_labels):
    """
    Расчет метрик сегментации
    
    Метрики:
    - Accuracy: процент правильно классифицированных точек
    - Precision: TP / (TP + FP) - точность предсказания класса "здание"
    - Recall: TP / (TP + FN) - полнота захвата класса "здание"
    - F1-Score: гармоническое среднее Precision и Recall
    - IoU (Intersection over Union): TP / (TP + FP + FN)
    """
    tp = np.sum((pred_labels == 1) & (gt_labels == 1))
    fp = np.sum((pred_labels == 1) & (gt_labels == 0))
    tn = np.sum((pred_labels == 0) & (gt_labels == 0))
    fn = np.sum((pred_labels == 0) & (gt_labels == 1))
    
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


def save_classified_cloud(points, labels, confidences, output_file):
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    colors[labels == 1] = [0, 255, 0]  # Зелёный = здание
    colors[labels == 0] = [255, 0, 0]  # Красный = не-здание
    
    vertex = np.array(
        [
            (points[i, 0], points[i, 1], points[i, 2],
             colors[i, 0], colors[i, 1], colors[i, 2],
             int(labels[i]), float(confidences[i]))
            for i in range(len(points))
        ],
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('label', 'i4'), ('confidence', 'f4')
        ]
    )
    
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(output_file)


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    # 1. Загрузка модели
    print(f"\nЗагрузка модели (segmentation): {MODEL_PATH}")
    model = PointNetSegmentation(num_classes=2).to(device)
    
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("✓ Модель загружена")
    except Exception as e:
        print(f"✗ Ошибка загрузки модели: {e}")
        return
    
    # 2. Загрузка PLY файла
    print(f"\nЗагрузка облака точек: {INPUT_FILE}")
    try:
        # ← ИЗМЕНЕНО: Чтение PLY вместо LAZ
        plydata = PlyData.read(INPUT_FILE)
        vertex_data = plydata['vertex']
        
        # Извлечение координат X, Y, Z
        points = np.vstack([
            vertex_data['x'],
            vertex_data['y'],
            vertex_data['z']
        ]).T.astype(np.float32)
        
        print(f"✓ Загружено {len(points):,} точек")
        print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
        
        # Попытка загрузить ground truth labels из PLY
        gt_labels = None
        possible_label_fields = ['label', 'classification', 'class', 'scalar_Classification']
        
        # Получаем список доступных полей
        available_fields = vertex_data.data.dtype.names
        print(f"\n  Доступные поля в PLY: {available_fields}")
        
        for field_name in possible_label_fields:
            if field_name in available_fields:
                gt_labels = np.array(vertex_data[field_name], dtype=np.int32)
                # Бинаризация: здания = 1, всё остальное = 0
                gt_labels = (gt_labels == 1).astype(np.int32)
                print(f"✓ Найдено поле ground truth: {field_name}")
                print(f"  Здания: {np.sum(gt_labels == 1):,} ({100*np.sum(gt_labels == 1)/len(gt_labels):.1f}%)")
                print(f"  Не-здания: {np.sum(gt_labels == 0):,} ({100*np.sum(gt_labels == 0)/len(gt_labels):.1f}%)")
                break
        
        if gt_labels is None:
            print("⚠️  Ground truth не найден - метрики не будут рассчитаны")
            
    except Exception as e:
        print(f"✗ Ошибка загрузки файла: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Сегментация всех точек
    print("\nНачало сегментации всего облака...")
    start_time = time.time()
    
    pred_labels, confidences = segment_full_cloud(
        model, points, NUM_POINTS, BATCH_SIZE
    )
    
    elapsed_time = time.time() - start_time
    
    # 4. Статистика
    num_buildings = np.sum(pred_labels == 1)
    num_non_buildings = np.sum(pred_labels == 0)
    avg_confidence = float(np.mean(confidences))
    
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ СЕГМЕНТАЦИИ")
    print("=" * 80)
    print(f"\nВсего точек: {len(points):,}")
    print(f"Здания (зелёный): {num_buildings:,} ({100*num_buildings/len(points):.1f}%)")
    print(f"Не-здания (красный): {num_non_buildings:,} ({100*num_non_buildings/len(points):.1f}%)")
    print(f"Средняя уверенность: {avg_confidence:.2%}")
    print(f"Время сегментации: {elapsed_time:.1f} секунд")
    print(f"Скорость: {len(points)/elapsed_time:.0f} точек/сек")
    
    # 5. Расчет метрик качества
    if gt_labels is not None:
        print("\n" + "=" * 80)
        print("МЕТРИКИ КАЧЕСТВА ДЕТЕКЦИИ")
        print("=" * 80)
        
        metrics = calculate_metrics(gt_labels, pred_labels)
        
        print(f"\n📊 Основные метрики:")
        print(f"  • Accuracy (точность):  {metrics['accuracy']:.2%}")
        print(f"    → Процент правильно классифицированных точек")
        print(f"    → Формула: (TP + TN) / Всего точек")
        
        print(f"\n  • Precision (точность класса 'здание'):  {metrics['precision']:.2%}")
        print(f"    → Из всех предсказанных зданий - сколько действительно здания")
        print(f"    → Формула: TP / (TP + FP)")
        
        print(f"\n  • Recall (полнота класса 'здание'):  {metrics['recall']:.2%}")
        print(f"    → Из всех реальных зданий - сколько модель нашла")
        print(f"    → Формула: TP / (TP + FN)")
        
        print(f"\n  • F1-Score (гармоническое среднее):  {metrics['f1_score']:.2%}")
        print(f"    → Баланс между Precision и Recall")
        print(f"    → Формула: 2 × (Precision × Recall) / (Precision + Recall)")
        
        print(f"\n  • IoU (Intersection over Union):  {metrics['iou']:.2%}")
        print(f"    → Площадь пересечения / Площадь объединения")
        print(f"    → Формула: TP / (TP + FP + FN)")
        
        print(f"\n📈 Confusion Matrix:")
        print(f"  ┌─────────────────┬──────────────┬──────────────┐")
        print(f"  │                 │  Pred: Здание│ Pred: Не-здан│")
        print(f"  ├─────────────────┼──────────────┼──────────────┤")
        print(f"  │ GT: Здание      │ {metrics['tp']:>12,} │ {metrics['fn']:>12,} │")
        print(f"  │ GT: Не-здание   │ {metrics['fp']:>12,} │ {metrics['tn']:>12,} │")
        print(f"  └─────────────────┴──────────────┴──────────────┘")
        
        print(f"\n  TP (True Positive):  {metrics['tp']:,} - Здание предсказано как здание ✓")
        print(f"  TN (True Negative):  {metrics['tn']:,} - Не-здание предсказано как не-здание ✓")
        print(f"  FP (False Positive): {metrics['fp']:,} - Не-здание предсказано как здание ✗")
        print(f"  FN (False Negative): {metrics['fn']:,} - Здание предсказано как не-здание ✗")
    
    # 6. Сохранение результата
    print(f"\n{'='*80}")
    print(f"Сохранение результата: {OUTPUT_FILE}")
    save_classified_cloud(points, pred_labels, confidences, OUTPUT_FILE)
    print("✓ Результат сохранён")
    
    print("\n" + "=" * 80)
    print("ЗАВЕРШЕНО")
    print("=" * 80)
    print(f"\nОткрой файл {OUTPUT_FILE} в CloudCompare или другом PLY viewer")
    print("Цвета:")
    print("  🟢 Зелёный = Здания/стены (label=1)")
    print("  🔴 Красный = Фон, не-здания (label=0)")


if __name__ == "__main__":
    main()
