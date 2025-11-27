"""
ГЕНЕРАТОР ДЕТАЛИЗИРОВАННЫХ ГОРОДСКИХ СЦЕН ДЛЯ ОБУЧЕНИЯ POINTNET

Особенности:
- Полные здания, разрушенные здания, углы (L-формат), отдельные стены
- Все здания и фрагменты стен = label 1
- Реалистичные автомобили (округлый кузов + колёса) = label 0
- Детализированные деревья = label 0
- Мусор / фон = label 0
- Высокая плотность точек
- Полная рандомизация

Разметка:
- 1 = здания и их фрагменты (включая отдельные стены, L-углы, разрушенные коробки)
- 0 = всё остальное (машины, деревья, мусор, фон)

Совместим с finetune_model.py и методом _read_ply_with_labels()
"""

import numpy as np
from plyfile import PlyData, PlyElement
import os
from tqdm import tqdm

# ============================================================================
# ПАРАМЕТРЫ ГЕНЕРАЦИИ
# ============================================================================

OUTPUT_FOLDER = "dataset/mixed"

NUM_SCENES = 1

# Диапазоны количества объектов
BUILDINGS_PER_SCENE = (2, 5)   # Включает разные типы зданий/стен
CARS_PER_SCENE = (0, 8)
TREES_PER_SCENE = (3, 12)
TRASH_PER_SCENE = (5, 20)

# Плотности точек
BUILDING_DENSITY = (300, 1000)
CAR_DENSITY = (200, 600)
TREE_DENSITY = (300, 800)
TRASH_DENSITY = (50, 200)

# Размер сцены (по X,Y)
SCENE_SIZE = 50.0

# Шум
NOISE_LEVEL = 0.02

# Вероятности типов зданий / стен
BUILDING_TYPE_PROBABILITIES = {
    'full': 0.5,        # 50% — полные коробки (4 стены + возможно крыша)
    'three_walls': 0.2, # 20% — разрушенные (3 стены)
    'two_walls': 0.2,   # 20% — углы (L-формат, 2 перпендикулярные стены)
    'single_wall': 0.1  # 10% — одна отдельная стена
}

print("=" * 80)
print("ГЕНЕРАТОР ДЕТАЛИЗИРОВАННЫХ ГОРОДСКИХ СЦЕН (WALL VARIANTS + REALISTIC CARS)")
print("=" * 80)
print(f"\nТипы зданий / стен:")
print(f" • Полные (4 стены): {BUILDING_TYPE_PROBABILITIES['full']*100:.0f}%")
print(f" • Разрушенные (3 стены): {BUILDING_TYPE_PROBABILITIES['three_walls']*100:.0f}%")
print(f" • Углы (2 стены, L): {BUILDING_TYPE_PROBABILITIES['two_walls']*100:.0f}%")
print(f" • Отдельные стены (1): {BUILDING_TYPE_PROBABILITIES['single_wall']*100:.0f}%")


# ============================================================================
# ФУНКЦИИ ГЕНЕРАЦИИ СТЕН И ВАРИАНТОВ ЗДАНИЙ
# ============================================================================

def generate_wall_with_windows(x0, y0, z0, wall_type, width, depth, height, density, windows=True):
    """
    Генерирует ОДНУ стену здания с (опционально) окнами.
    Это базовый строительный блок для:
    - полных зданий
    - разрушенных коробок
    - L-угловых стен
    - одиночных стен

    Все точки этой стены считаются частью здания => label=1 (назначается выше).

    Args:
        x0, y0, z0: позиция центра воображаемой коробки здания.
        wall_type: 'front', 'back', 'left', 'right'.
        width, depth, height: размеры коробки.
        density: плотность точек.
        windows: генерировать ли вырезы под окна.
    Returns:
        list из [x, y, z] точек.
    """
    points = []

    # Параметры окон (относительные)
    window_width = width * 0.15
    window_height = height * 0.12

    def is_inside_window(x, y, z, w_type):
        """Проверяет, попадает ли точка внутрь окна (тогда мы её вырезаем)."""
        if not windows:
            return False

        if w_type in ['front', 'back']:
            # Кол-во окон по ширине
            windows_per_wall = max(1, int(width / (window_width * 2)))
            for i in range(windows_per_wall):
                window_x = x0 - width / 2 + (i + 1) * (width / (windows_per_wall + 1))
                window_z_start = z0 + height * 0.2
                if (abs(x - window_x) < window_width / 2 and
                        window_z_start < z < window_z_start + window_height):
                    return True

        elif w_type in ['left', 'right']:
            # Кол-во окон по глубине
            windows_per_side = max(1, int(depth / (window_width * 2)))
            for i in range(windows_per_side):
                window_y = y0 - depth / 2 + (i + 1) * (depth / (windows_per_side + 1))
                window_z_start = z0 + height * 0.2
                if (abs(y - window_y) < window_width / 2 and
                        window_z_start < z < window_z_start + window_height):
                    return True

        return False

    # Генерация точек на выбранной стене
    if wall_type == 'front':
        # Передняя стена (y = y0 - depth/2)
        num_points = int(width * height * density / 100)
        for _ in range(num_points):
            x = np.random.uniform(x0 - width / 2, x0 + width / 2)
            z = np.random.uniform(z0, z0 + height)
            y = y0 - depth / 2
            if not is_inside_window(x, y, z, 'front'):
                points.append([x, y, z])

    elif wall_type == 'back':
        # Задняя стена (y = y0 + depth/2)
        num_points = int(width * height * density / 100)
        for _ in range(num_points):
            x = np.random.uniform(x0 - width / 2, x0 + width / 2)
            z = np.random.uniform(z0, z0 + height)
            y = y0 + depth / 2
            if not is_inside_window(x, y, z, 'back'):
                points.append([x, y, z])

    elif wall_type == 'left':
        # Левая стена (x = x0 - width/2)
        num_points = int(depth * height * density / 100)
        for _ in range(num_points):
            y = np.random.uniform(y0 - depth / 2, y0 + depth / 2)
            z = np.random.uniform(z0, z0 + height)
            x = x0 - width / 2
            if not is_inside_window(x, y, z, 'left'):
                points.append([x, y, z])

    elif wall_type == 'right':
        # Правая стена (x = x0 + width/2)
        num_points = int(depth * height * density / 100)
        for _ in range(num_points):
            y = np.random.uniform(y0 - depth / 2, y0 + depth / 2)
            z = np.random.uniform(z0, z0 + height)
            x = x0 + width / 2
            if not is_inside_window(x, y, z, 'right'):
                points.append([x, y, z])

    return points


def generate_building_variant(pos, width, depth, height, density, building_type):
    """
    Генерирует ЗДАНИЕ/ФРАГМЕНТ в зависимости от типа:
    - 'full'        : 4 стены + (часто) крыша
    - 'three_walls' : 3 стены, иногда крыша
    - 'two_walls'   : L-угол, 2 перпендикулярные стены (реально L-формат)
    - 'single_wall' : одна отдельная стена

    Все точки результата имеют label=1 (здание/фрагмент здания).

    Args:
        pos: (x, y, z) центр коробки.
        width, depth, height: размеры.
        density: плотность.
        building_type: строка одного из допустимых типов.

    Returns:
        points: np.ndarray (N, 3)
        labels: np.ndarray (N,) — все единицы.
    """
    x0, y0, z0 = pos
    points = []

    # Определяем, какие стены будут сгенерированы
    if building_type == 'full':
        # Полное здание: 4 стены + крыша
        walls = ['front', 'back', 'left', 'right']
        add_roof = True

    elif building_type == 'three_walls':
        # Разрушенное: 3 стены (одну случайно убираем)
        all_walls = ['front', 'back', 'left', 'right']
        removed_wall = np.random.choice(all_walls)
        walls = [w for w in all_walls if w != removed_wall]
        add_roof = np.random.random() > 0.5  # 50% есть крыша

    elif building_type == 'two_walls':
        # L-угол: 2 перпендикулярные стены
        corner_variants = [
            ['front', 'left'],   # Передний левый угол
            ['front', 'right'],  # Передний правый
            ['back', 'left'],    # Задний левый
            ['back', 'right']    # Задний правый
        ]
        idx = np.random.choice([0, 1, 2, 3])
        walls = corner_variants[idx]
        add_roof = False  # Обычно без крыши, чисто стеновой угол

    elif building_type == 'single_wall':
        # Одна отдельная стена (как standalone объект)
        walls = [np.random.choice(['front', 'back', 'left', 'right'])]
        add_roof = False

    else:
        # На всякий случай — дефолт: полная коробка
        walls = ['front', 'back', 'left', 'right']
        add_roof = True

    # Генерируем выбранные стены
    for wall_type in walls:
        wall_points = generate_wall_with_windows(
            x0, y0, z0, wall_type, width, depth, height, density, windows=True
        )
        points.extend(wall_points)

    # При необходимости добавляем крышу
    if add_roof:
        num_roof = int(width * depth * density / 200)
        for _ in range(num_roof):
            x = np.random.uniform(x0 - width / 2, x0 + width / 2)
            y = np.random.uniform(y0 - depth / 2, y0 + depth / 2)
            z = z0 + height
            points.append([x, y, z])

    points = np.array(points, dtype=np.float32)
    # ВАЖНО: все стены/здания/фрагменты = label 1
    labels = np.ones(len(points), dtype=np.int32)
    return points, labels


# ============================================================================
# РЕАЛИСТИЧНЫЕ МАШИНЫ (label=0)
# ============================================================================

def generate_detailed_car(pos, length, width, height, density):
    """
    Реалистичный автомобиль:
    - Нижний прямоугольный кузов (шасси)
    - Округлая кабина (часть эллипсоида сверху)
    - Четыре колеса (цилиндрические кластеры в углах)

    Все точки: label=0 (не здание).
    """
    x0, y0, z0 = pos
    points = []

    # 1. Кузов (прямоугольный параллелепипед)
    chassis_height = height * 0.4
    num_chassis_points = int(length * width * chassis_height * density / 8)

    for _ in range(num_chassis_points):
        face = np.random.randint(0, 6)
        if face == 0:  # передняя грань
            x = x0 - length / 2
            y = np.random.uniform(y0 - width / 2, y0 + width / 2)
            z = np.random.uniform(z0, z0 + chassis_height)
        elif face == 1:  # задняя
            x = x0 + length / 2
            y = np.random.uniform(y0 - width / 2, y0 + width / 2)
            z = np.random.uniform(z0, z0 + chassis_height)
        elif face == 2:  # левая
            x = np.random.uniform(x0 - length / 2, x0 + length / 2)
            y = y0 - width / 2
            z = np.random.uniform(z0, z0 + chassis_height)
        elif face == 3:  # правая
            x = np.random.uniform(x0 - length / 2, x0 + length / 2)
            y = y0 + width / 2
            z = np.random.uniform(z0, z0 + chassis_height)
        elif face == 4:  # верх шасси
            x = np.random.uniform(x0 - length / 2, x0 + length / 2)
            y = np.random.uniform(y0 - width / 2, y0 + width / 2)
            z = z0 + chassis_height
        else:  # низ
            x = np.random.uniform(x0 - length / 2, x0 + length / 2)
            y = np.random.uniform(y0 - width / 2, y0 + width / 2)
            z = z0 + np.random.uniform(0, chassis_height * 0.1)
        points.append([x, y, z])

    # 2. Округлая кабина (часть эллипсоида)
    cabin_height = height * 0.6
    cabin_center_z = z0 + chassis_height + cabin_height * 0.5
    cabin_rx = length * 0.35
    cabin_ry = width * 0.4
    cabin_rz = cabin_height * 0.6
    num_cabin_points = int(length * width * cabin_height * density / 10)

    for _ in range(num_cabin_points):
        for __ in range(10):
            dx = np.random.uniform(-cabin_rx, cabin_rx)
            dy = np.random.uniform(-cabin_ry, cabin_ry)
            dz = np.random.uniform(-cabin_rz, cabin_rz)
            if (dx * dx) / (cabin_rx * cabin_rx) + \
               (dy * dy) / (cabin_ry * cabin_ry) + \
               (dz * dz) / (cabin_rz * cabin_rz) <= 1.0:
                x = x0 + dx
                y = y0 + dy
                z = cabin_center_z + dz
                if z >= z0 + chassis_height:
                    points.append([x, y, z])
                break

    # 3. Колёса (4 цилиндра в углах)
    wheel_radius = min(length, width) * 0.12
    wheel_width = width * 0.20
    wheel_center_z = z0 + wheel_radius * 0.8
    dx = length * 0.35
    dy = width * 0.35

    wheel_centers = [
        (x0 - dx, y0 - dy, wheel_center_z),
        (x0 - dx, y0 + dy, wheel_center_z),
        (x0 + dx, y0 - dy, wheel_center_z),
        (x0 + dx, y0 + dy, wheel_center_z),
    ]

    num_wheel_points = int(wheel_radius * wheel_radius * density / 2)

    for cx, cy, cz in wheel_centers:
        for _ in range(num_wheel_points):
            theta = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, wheel_radius)
            x = cx + r * np.cos(theta)
            z = cz + r * np.sin(theta)
            y = cy + np.random.uniform(-wheel_width / 2, wheel_width / 2)
            points.append([x, y, z])

    points = np.array(points, dtype=np.float32)
    labels = np.zeros(len(points), dtype=np.int32)  # label=0
    return points, labels


# ============================================================================
# ДЕРЕВЬЯ И МУСОР (label=0)
# ============================================================================

def generate_detailed_tree(pos, radius, height, density):
    """
    Детализированное дерево (ствол, ветки, крона). Все точки label=0.
    """
    x0, y0, z0 = pos
    points = []

    trunk_height = height * 0.45
    trunk_radius = radius * 0.12

    # Ствол
    num_trunk = int(trunk_height * density)
    for _ in range(num_trunk):
        theta = np.random.uniform(0, 2 * np.pi)
        r_variation = np.random.uniform(0.8, 1.0)
        r = trunk_radius * r_variation
        x = x0 + r * np.cos(theta)
        y = y0 + r * np.sin(theta)
        z = np.random.uniform(z0, z0 + trunk_height)
        points.append([x, y, z])

    # Главные ветки
    num_main_branches = np.random.randint(3, 8)
    branch_start_height = z0 + trunk_height * 0.6

    for branch_idx in range(num_main_branches):
        branch_angle = (branch_idx / num_main_branches) * 2 * np.pi
        branch_tilt = np.random.uniform(np.pi / 6, np.pi / 3)
        branch_length = np.random.uniform(radius * 0.8, radius * 1.5)
        branch_radius = trunk_radius * np.random.uniform(0.3, 0.5)
        branch_start_z = branch_start_height + np.random.uniform(0, trunk_height * 0.3)
        num_branch_points = int(branch_length * density / 2)

        for i in range(num_branch_points):
            t = i / max(num_branch_points, 1)
            dist = t * branch_length

            x_dir = dist * np.sin(branch_tilt) * np.cos(branch_angle)
            y_dir = dist * np.sin(branch_tilt) * np.sin(branch_angle)
            z_dir = dist * np.cos(branch_tilt)

            curve = np.random.uniform(-0.1, 0.1)
            current_radius = branch_radius * (1 - t * 0.7)

            theta = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, current_radius)

            x = x0 + x_dir + r * np.cos(theta) + curve
            y = y0 + y_dir + r * np.sin(theta) + curve
            z = branch_start_z + z_dir
            points.append([x, y, z])

        # Мелкие веточки
        num_small_branches = np.random.randint(2, 5)
        for _ in range(num_small_branches):
            t_small = np.random.uniform(0.3, 1.0)
            dist_small = t_small * branch_length

            x_base = x0 + dist_small * np.sin(branch_tilt) * np.cos(branch_angle)
            y_base = y0 + dist_small * np.sin(branch_tilt) * np.sin(branch_angle)
            z_base = branch_start_z + dist_small * np.cos(branch_tilt)

            small_angle = branch_angle + np.random.uniform(-np.pi / 3, np.pi / 3)
            small_length = branch_length * np.random.uniform(0.2, 0.4)
            num_small_points = int(small_length * density / 4)

            for i in range(num_small_points):
                t_s = i / max(num_small_points, 1)
                dist_s = t_s * small_length
                x = x_base + dist_s * np.cos(small_angle)
                y = y_base + dist_s * np.sin(small_angle)
                z = z_base + dist_s * np.random.uniform(-0.2, 0.5)
                points.append([x, y, z])

    # Крона
    crown_center_z = z0 + trunk_height + radius * 0.5
    num_clusters = np.random.randint(5, 10)

    for _ in range(num_clusters):
        cluster_offset_x = np.random.uniform(-radius * 0.4, radius * 0.4)
        cluster_offset_y = np.random.uniform(-radius * 0.4, radius * 0.4)
        cluster_offset_z = np.random.uniform(-radius * 0.3, radius * 0.3)

        cluster_x = x0 + cluster_offset_x
        cluster_y = y0 + cluster_offset_y
        cluster_z = crown_center_z + cluster_offset_z

        cluster_radius = radius * np.random.uniform(0.4, 0.7)
        num_cluster_points = int(cluster_radius * cluster_radius * density * 2)

        for __ in range(num_cluster_points):
            while True:
                dx = np.random.uniform(-cluster_radius, cluster_radius)
                dy = np.random.uniform(-cluster_radius, cluster_radius)
                dz = np.random.uniform(-cluster_radius, cluster_radius)
                if dx * dx + dy * dy + dz * dz <= cluster_radius * cluster_radius:
                    points.append([cluster_x + dx, cluster_y + dy, cluster_z + dz])
                    break

    points = np.array(points, dtype=np.float32)
    labels = np.zeros(len(points), dtype=np.int32)  # label=0
    return points, labels


def generate_trash(pos, size, density):
    """
    Мусор: небольшие кластеры точек, все label=0.
    """
    x0, y0, z0 = pos
    num_points = int(size * density)
    points = []

    for _ in range(num_points):
        x = x0 + np.random.uniform(-size, size)
        y = y0 + np.random.uniform(-size, size)
        z = z0 + np.random.uniform(0, size * 0.5)
        points.append([x, y, z])

    points = np.array(points, dtype=np.float32)
    labels = np.zeros(len(points), dtype=np.int32)
    return points, labels


# ============================================================================
# ГЕНЕРАЦИЯ ПОЛНОЙ СЦЕНЫ
# ============================================================================

def generate_scene():
    """
    Генерирует одну городскую сцену:
    - Здания разных типов (полные, разрушенные, L, одиночные стены) — label=1
    - Автомобили — label=0
    - Деревья — label=0
    - Мусор — label=0
    """
    all_points = []
    all_labels = []

    building_types_generated = {
        'full': 0,
        'three_walls': 0,
        'two_walls': 0,
        'single_wall': 0
    }

    # Здания/стены
    num_buildings = np.random.randint(*BUILDINGS_PER_SCENE)
    for _ in range(num_buildings):
        x = np.random.uniform(-SCENE_SIZE / 2, SCENE_SIZE / 2)
        y = np.random.uniform(-SCENE_SIZE / 2, SCENE_SIZE / 2)
        z = 0.0

        width = np.random.uniform(3, 10)
        depth = np.random.uniform(3, 10)
        height = np.random.uniform(5, 25)
        density = np.random.randint(*BUILDING_DENSITY)

        building_type = np.random.choice(
            list(BUILDING_TYPE_PROBABILITIES.keys()),
            p=list(BUILDING_TYPE_PROBABILITIES.values())
        )
        building_types_generated[building_type] += 1

        points, labels = generate_building_variant(
            (x, y, z), width, depth, height, density, building_type
        )
        all_points.append(points)
        all_labels.append(labels)

    # Машины
    num_cars = np.random.randint(*CARS_PER_SCENE)
    for _ in range(num_cars):
        x = np.random.uniform(-SCENE_SIZE / 2, SCENE_SIZE / 2)
        y = np.random.uniform(-SCENE_SIZE / 2, SCENE_SIZE / 2)
        z = 0.0

        length = np.random.uniform(3, 5)
        width = np.random.uniform(1.5, 2.5)
        height = np.random.uniform(1.2, 2.0)
        density = np.random.randint(*CAR_DENSITY)

        points, labels = generate_detailed_car(
            (x, y, z), length, width, height, density
        )
        all_points.append(points)
        all_labels.append(labels)

    # Деревья
    num_trees = np.random.randint(*TREES_PER_SCENE)
    for _ in range(num_trees):
        x = np.random.uniform(-SCENE_SIZE / 2, SCENE_SIZE / 2)
        y = np.random.uniform(-SCENE_SIZE / 2, SCENE_SIZE / 2)
        z = 0.0

        radius = np.random.uniform(1, 3)
        height = np.random.uniform(4, 10)
        density = np.random.randint(*TREE_DENSITY)

        points, labels = generate_detailed_tree(
            (x, y, z), radius, height, density
        )
        all_points.append(points)
        all_labels.append(labels)

    # Мусор
    num_trash = np.random.randint(*TRASH_PER_SCENE)
    for _ in range(num_trash):
        x = np.random.uniform(-SCENE_SIZE / 2, SCENE_SIZE / 2)
        y = np.random.uniform(-SCENE_SIZE / 2, SCENE_SIZE / 2)
        z = 0.0

        size = np.random.uniform(0.2, 1.0)
        density = np.random.randint(*TRASH_DENSITY)

        points, labels = generate_trash(
            (x, y, z), size, density
        )
        all_points.append(points)
        all_labels.append(labels)

    # Объединение
    all_points = np.vstack(all_points)
    all_labels = np.concatenate(all_labels)

    # Добавляем шум
    noise = np.random.normal(0, NOISE_LEVEL, all_points.shape).astype(np.float32)
    all_points += noise

    return all_points, all_labels, building_types_generated


# ============================================================================
# СОХРАНЕНИЕ PLY
# ============================================================================

def save_ply(filename, points, labels):
    """
    Сохраняет облако точек с метками в формат PLY.
    """
    vertex = np.zeros(
        len(points),
        dtype=[
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4'),
            ('label', 'i4')
        ]
    )

    vertex['x'] = points[:, 0]
    vertex['y'] = points[:, 1]
    vertex['z'] = points[:, 2]
    vertex['label'] = labels

    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(filename)


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print(f"\n📁 Папка: {OUTPUT_FOLDER}/")
    print(f"🎲 Генерация {NUM_SCENES} детализированных сцен...\n")

    total_building_points = 0
    total_other_points = 0
    total_points_all = 0

    total_building_types = {
        'full': 0,
        'three_walls': 0,
        'two_walls': 0,
        'single_wall': 0
    }

    points_per_file = []

    for i in tqdm(range(NUM_SCENES), desc="Генерация"):
        points, labels, building_types = generate_scene()

        num_points = len(points)
        points_per_file.append(num_points)

        building_points = np.sum(labels == 1)
        other_points = np.sum(labels == 0)

        total_building_points += building_points
        total_other_points += other_points
        total_points_all += num_points

        for btype, count in building_types.items():
            total_building_types[btype] += count

        filename = os.path.join(OUTPUT_FOLDER, f"scene_{i:04d}.ply")
        save_ply(filename, points, labels)

        if (i + 1) % 50 == 0 or i == 0:
            print(
                f"\n 📄 scene_{i:04d}.ply: {num_points:,} точек "
                f"(здания: {building_points:,}, фон: {other_points:,})"
            )

    print("\n" + "=" * 80)
    print("✅ ГЕНЕРАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 80)

    print(f"\n📊 Общая статистика:")
    print(f" • Всего сцен: {NUM_SCENES}")
    print(f" • Всего точек: {total_points_all:,}")
    print(f" • Средняя точек/сцену: {total_points_all // NUM_SCENES:,}")
    print(f" • Мин точек/сцену: {min(points_per_file):,}")
    print(f" • Макс точек/сцену: {max(points_per_file):,}")

    print(f"\n🏗️ Статистика зданий/стен:")
    total_buildings = sum(total_building_types.values())
    if total_buildings > 0:
        print(f" • Всего объектов (зданий/фрагментов): {total_buildings}")
        print(f" • Полные (4 стены): {total_building_types['full']} "
              f"({total_building_types['full']/total_buildings*100:.1f}%)")
        print(f" • Разрушенные (3 стены): {total_building_types['three_walls']} "
              f"({total_building_types['three_walls']/total_buildings*100:.1f}%)")
        print(f" • Углы (2 стены, L): {total_building_types['two_walls']} "
              f"({total_building_types['two_walls']/total_buildings*100:.1f}%)")
        print(f" • Отдельные стены (1): {total_building_types['single_wall']} "
              f"({total_building_types['single_wall']/total_buildings*100:.1f}%)")

    print(f"\n📈 Баланс классов:")
    print(f" • Здания/стены (label=1): {total_building_points:,}")
    print(f" • Фон (label=0): {total_other_points:,}")
    ratio = total_building_points / (total_building_points + total_other_points) * 100
    print(f" • Соотношение: {ratio:.1f}%")
    if 30 <= ratio <= 70:
        print(" ✅ Баланс хороший (30-70%)")
    else:
        print(f" ⚠️ Дисбаланс! Текущий {ratio:.1f}%")

    print(f"\n🚀 Что добавлено:")
    print(f" ✅ Полные здания, разрушенные, L-углы, одиночные стены (все label=1)")
    print(f" ✅ Реалистичные автомобили с колёсами (label=0)")
    print(f" ✅ Детализированные деревья и мусор (label=0)")

    print(f"\n📝 Следующие шаги:")
    print(f" 1. MIXED_DATA_FOLDER = '{OUTPUT_FOLDER}'")
    print(f" 2. python finetune_model.py")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
