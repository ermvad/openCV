import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFileDialog, QComboBox, QSlider, QScrollArea,
                             QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox,
                             QTabWidget, QRadioButton, QButtonGroup, QListWidget, QListWidgetItem,
                             QSplitter, QSizePolicy, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QSize

import filters
from filter_category import FilterCategory
from filter_manager import Filter, FilterManager
from logger import logger


class OpenCVFilters(QMainWindow):
    """
    Основной класс приложения для обработки изображений с использованием OpenCV.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV Filters")
        self.setGeometry(100, 100, 1300, 800)
        self.filter_manager = FilterManager()
        logger.info("Приложение запущено")
        self.init_ui()
        self.populate_filters()
        self.add_filter_button.setEnabled(False)  # Отключаем кнопку добавления фильтра до загрузки изображения

    def init_ui(self):
        """Инициализирует пользовательский интерфейс."""
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        self.load_button = QPushButton("Загрузить изображение")
        self.load_button.clicked.connect(self.load_image)
        left_layout.addWidget(self.load_button)
        filter_selection_group = QGroupBox("Выбор фильтров")
        filter_selection_layout = QVBoxLayout()

        self.category_list = QListWidget()
        self.category_list.setMaximumHeight(150)

        for category in FilterCategory.get_all():
            item = QListWidgetItem(category)
            self.category_list.addItem(item)

        self.category_list.itemClicked.connect(self.on_category_selected)
        filter_selection_layout.addWidget(QLabel("Категории фильтров:"))
        filter_selection_layout.addWidget(self.category_list)

        # список доступных фильтров в выбранной категории
        filter_selection_layout.addWidget(QLabel("Доступные фильтры:"))
        self.filter_combo = QComboBox()
        self.filter_combo.setMinimumWidth(200)
        filter_selection_layout.addWidget(self.filter_combo)

        # Кнопка добавления фильтра
        self.add_filter_button = QPushButton("Добавить выбранный фильтр")
        self.add_filter_button.clicked.connect(self.add_selected_filter)
        filter_selection_layout.addWidget(self.add_filter_button)

        filter_selection_group.setLayout(filter_selection_layout)
        left_layout.addWidget(filter_selection_group)

        # Словарь для хранения комбобоксов по категориям
        self.category_combos = {}

        # Область для списка примененных фильтров
        filters_header = QLabel("Примененные фильтры")
        filters_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        left_layout.addWidget(filters_header)

        self.filters_group = QWidget()
        self.filters_layout = QVBoxLayout()
        self.filters_group.setLayout(self.filters_layout)

        # Скролл область для фильтров
        filters_scroll = QScrollArea()
        filters_scroll.setWidgetResizable(True)
        filters_scroll.setWidget(self.filters_group)
        left_layout.addWidget(filters_scroll)

        # Кнопки действий
        buttons_layout = QHBoxLayout()

        # Кнопка сброса фильтров
        self.reset_button = QPushButton("Сбросить все фильтры")
        self.reset_button.clicked.connect(self.reset_filters)
        buttons_layout.addWidget(self.reset_button)

        # Кнопка сохранения результата
        self.save_button = QPushButton("Сохранить результат")
        self.save_button.clicked.connect(self.save_image)
        buttons_layout.addWidget(self.save_button)

        left_layout.addLayout(buttons_layout)

        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(400)

        # Правая панель для отображения изображений
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        # Только предпросмотр на всю ширину
        filtered_group = QGroupBox("Предпросмотр")
        filtered_layout = QVBoxLayout()
        self.filtered_label = QLabel()
        self.filtered_label.setAlignment(Qt.AlignCenter)
        self.filtered_label.setMinimumSize(600, 500)  # Увеличиваем размер
        filtered_layout.addWidget(self.filtered_label)
        filtered_group.setLayout(filtered_layout)

        right_layout.addWidget(filtered_group, 1)  # Растягиваем по вертикали (1)

        # Информация об изображении - компактный вариант
        info_widget = QWidget()
        info_widget.setMaximumHeight(40)  # Ограничение высоты
        info_layout = QHBoxLayout(info_widget)
        info_layout.setContentsMargins(5, 0, 5, 0)

        # Используем один label для всей информации
        self.image_info_label = QLabel("Нет изображения")
        font = QFont()
        font.setPointSize(9)
        self.image_info_label.setFont(font)
        self.image_info_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(self.image_info_label)

        right_layout.addWidget(info_widget)
        right_panel.setLayout(right_layout)

        # Добавление панелей в основную компоновку
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Деактивация кнопок до загрузки изображения
        self.reset_button.setEnabled(False)
        self.save_button.setEnabled(False)

        # Подключение сигнала обновления
        self.filter_manager.signals.updated.connect(self.update_preview)

    def populate_filters(self):
        """Заполняет словарь доступных фильтров по категориям."""
        # Словарь фильтров по категориям
        self.filters_by_category = {}

        # Размытие и сглаживание
        self.filters_by_category[FilterCategory.BLUR] = [
            "Размытие по Гауссу",
            "Медианный фильтр",
            "Двустороннее размытие",
            "Размытие среднего",
            "Размытие по Блоку",
            "Фильтр Собеля",
            "Фильтр Лапласа",
            "Размытие по Стэку",
            "Фильтр НЧМ (Нижних частот)",
        ]

        # Обнаружение краев и контуров
        self.filters_by_category[FilterCategory.EDGES] = [
            "Canny",
            "Собель X",
            "Собель Y",
            "Собель XY",
            "Лаплас",
            "Scharr X",
            "Scharr Y",
            "Scharr XY",
            "Детектор углов Харриса",
            "Детектор углов Ши-Томаси",
            "Выделение контуров"
        ]

        # Цветовые преобразования
        self.filters_by_category[FilterCategory.COLOR] = [
            "Оттенки серого",
            "Сепия",
            "HSV",
            "LAB",
            "YCrCb",
            "Негатив",
            "Изменение насыщенности",
            "Изменение оттенка",
            "Повышение контраста CLAHE",
            "Псевдоцвета (Jet)",
            "Псевдоцвета (Hot)",
            "Псевдоцвета (Cool)",
            "Изменение баланса RGB"
        ]

        # Морфологические операции
        self.filters_by_category[FilterCategory.MORPHOLOGY] = [
            "Эрозия",
            "Дилатация",
            "Открытие",
            "Закрытие",
            "Градиент",
            "Верхняя шляпа",
            "Черная шляпа",
            "Скелетизация",
            "Утончение",
        ]

        # Геометрические преобразования
        self.filters_by_category[FilterCategory.TRANSFORM] = [
            "Масштабирование",
            "Поворот",
            "Отражение",
            "Сдвиг",
            "Перспективное преобразование",
            "Аффинное преобразование",
            "Коррекция перспективы"
        ]

        # Улучшение изображения
        self.filters_by_category[FilterCategory.ENHANCEMENT] = [
            "Гамма-коррекция",
            "Яркость/Контраст",
            "Автоматический баланс белого",
            "Автоконтраст",
            "Поиск и удаление шума",
            "Повышение резкости",
            "Детализация",
            "HDR-эффект",
            "Выравнивание гистограммы"
        ]

        # Сегментация и пороги
        self.filters_by_category[FilterCategory.SEGMENTATION] = [
            "Бинаризация (пороговая)",
            "Адаптивный порог",
            "Порог Отсу",
            "Водораздел",
            "Mean Shift",
            "K-Means",
            "GrabCut",
            "Детектор движений"
        ]

        # Выделение признаков
        self.filters_by_category[FilterCategory.FEATURE] = [
            "SIFT-дескриптор",
            "SURF-дескриптор",
            "ORB-дескриптор",
            "HOG-дескриптор",
            "Детектор лиц Haar Cascade",
            "Детектор глаз Haar Cascade",
            "Выделение линий Хафа",
            "Выделение окружностей Хафа"
        ]

        # Пользовательские фильтры
        self.filters_by_category[FilterCategory.CUSTOM] = [
            "Карта глубины",
            "Мультиплексирование каналов",
            "Картунизация (Cartoon-эффект)",
            "Пикселизация",
            "Виньетка",
            "Размытие в движении",
            "Акварельный эффект",
            "Эффект масляной живописи",
            "Стилизация под карандашный рисунок"
        ]

        # Выбираем первую категорию по умолчанию
        if self.category_list.count() > 0:
            self.category_list.setCurrentRow(0)
            self.on_category_selected(self.category_list.item(0))

    def load_image(self):
        """Загружает изображение из файла."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Открыть изображение", "",
                                                       "Изображения (*.png *.jpg *.jpeg *.bmp *.tiff)")
            if file_path:
                logger.info(f"Загрузка изображения: {file_path}")
                image = cv2.imread(file_path)
                if image is None:
                    logger.error(f"Не удалось загрузить изображение: {file_path}")
                    QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить изображение: {file_path}")
                    return

                self.filter_manager.set_original_image(image)

                # Обновление информации об изображении
                height, width = image.shape[:2]
                channels = image.shape[2] if len(image.shape) > 2 else 1
                dtype = image.dtype

                # Компактная информация в одной строке
                info_text = f"Размер: {width}x{height}  |  Каналы: {channels} ({self.get_channel_description(channels)})  |  Тип: {dtype}"
                self.image_info_label.setText(info_text)

                # Включить кнопки
                self.reset_button.setEnabled(True)
                self.save_button.setEnabled(True)
                self.add_filter_button.setEnabled(True)

                # Обновление предпросмотра
                self.update_preview()

                logger.info(f"Изображение успешно загружено: {width}x{height}, {channels} каналов")
        except Exception as e:
            logger.error(f"Ошибка при загрузке изображения: {e}")
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при загрузке изображения: {e}")

    def get_channel_description(self, channels):
        """Возвращает описание каналов изображения."""
        if channels == 1:
            return "Оттенки серого"
        elif channels == 3:
            return "BGR"
        elif channels == 4:
            return "BGRA (с прозрачностью)"
        else:
            return f"{channels} каналов"

    def on_category_selected(self, item):
        """Обработчик выбора категории фильтров."""
        category = item.text()
        # Обновляем выпадающий список фильтров
        self.filter_combo.clear()
        if category in self.filters_by_category:
            self.filter_combo.addItems(self.filters_by_category[category])

    def add_selected_filter(self):
        """Добавляет выбранный фильтр в список активных фильтров."""
        if self.filter_manager.get_original_image() is None or self.category_list.currentItem() is None:
            return

        category = self.category_list.currentItem().text()
        filter_name = self.filter_combo.currentText()

        self.add_filter(category, filter_name)

    def add_filter(self, category, filter_name=None):
        """Добавляет выбранный фильтр в список активных фильтров."""
        if self.filter_manager.get_original_image() is None:
            return

        if filter_name is None:
            # Для обратной совместимости со старым кодом
            combo = self.category_combos[category]
            filter_name = combo.currentText()

        # Создание фильтра в зависимости от выбора и категории
        if category == FilterCategory.BLUR:
            self.add_blur_filter(filter_name)
        elif category == FilterCategory.EDGES:
            self.add_edge_filter(filter_name)
        elif category == FilterCategory.COLOR:
            self.add_color_filter(filter_name)
        elif category == FilterCategory.MORPHOLOGY:
            self.add_morphology_filter(filter_name)
        elif category == FilterCategory.TRANSFORM:
            self.add_transform_filter(filter_name)
        elif category == FilterCategory.ENHANCEMENT:
            self.add_enhancement_filter(filter_name)
        elif category == FilterCategory.SEGMENTATION:
            self.add_segmentation_filter(filter_name)
        elif category == FilterCategory.FEATURE:
            self.add_feature_filter(filter_name)
        elif category == FilterCategory.CUSTOM:
            self.add_custom_filter(filter_name)

    # Методы добавления фильтров по категориям
    def add_blur_filter(self, filter_name):
        """Добавляет фильтр размытия."""
        if filter_name == "Размытие по Гауссу":
            new_filter = Filter("Размытие по Гауссу", filters.apply_gaussian_blur,
                                {"ksize": (5, 5), "sigma": 0})
            self.add_filter_widget(new_filter, {"ksize_x": (1, 99, 2), "ksize_y": (1, 99, 2),
                                                "sigma": (0, 20, 0.1)})

        elif filter_name == "Медианный фильтр":
            new_filter = Filter("Медианный фильтр", filters.apply_median_blur, {"ksize": 5})
            self.add_filter_widget(new_filter, {"ksize": (1, 99, 2)})

        elif filter_name == "Двустороннее размытие":
            new_filter = Filter("Двустороннее размытие", filters.apply_bilateral_filter,
                                {"d": 9, "sigma_color": 75, "sigma_space": 75})
            self.add_filter_widget(new_filter, {"d": (5, 50, 2),
                                                "sigma_color": (10, 200, 5),
                                                "sigma_space": (10, 200, 5)})

        elif filter_name == "Размытие среднего":
            new_filter = Filter("Размытие среднего", filters.apply_blur, {"ksize": (5, 5)})
            self.add_filter_widget(new_filter, {"ksize_x": (1, 99, 2), "ksize_y": (1, 99, 2)})

        elif filter_name == "Размытие по Блоку":
            new_filter = Filter("Размытие по Блоку", filters.apply_box_filter, {"ksize": (5, 5)})
            self.add_filter_widget(new_filter, {"ksize_x": (1, 99, 2), "ksize_y": (1, 99, 2)})

        elif filter_name == "Фильтр Собеля":
            new_filter = Filter("Фильтр Собеля", filters.apply_sobel, {"dx": 1, "dy": 1, "ksize": 3})
            self.add_filter_widget(new_filter, {"dx": (0, 3, 1), "dy": (0, 3, 1),
                                                "ksize": [1, 3, 5, 7]})

        elif filter_name == "Фильтр Лапласа":
            new_filter = Filter("Фильтр Лапласа", filters.apply_laplacian, {"ksize": 3})
            self.add_filter_widget(new_filter, {"ksize": [1, 3, 5, 7]})

        elif filter_name == "Размытие по Стэку":
            new_filter = Filter("Размытие по Стэку", filters.apply_stack_blur, {"ksize": (21, 21)})
            self.add_filter_widget(new_filter, {"ksize_x": (1, 99, 2), "ksize_y": (1, 99, 2)})

        elif filter_name == "Фильтр НЧМ (Нижних частот)":
            new_filter = Filter("Фильтр НЧМ", filters.apply_lowpass_filter, {"cutoff": 30})
            self.add_filter_widget(new_filter, {"cutoff": (5, 200, 1)})

        self.filter_manager.add_filter(new_filter)

    def add_edge_filter(self, filter_name):
        """Добавляет фильтр обнаружения краев."""
        if filter_name == "Canny":
            new_filter = Filter("Canny", filters.apply_canny_edge,
                                {"threshold1": 100, "threshold2": 200})
            self.add_filter_widget(new_filter, {"threshold1": (0, 500, 1),
                                                "threshold2": (0, 500, 1)})

        elif filter_name == "Собель X":
            new_filter = Filter("Собель X", filters.apply_sobel_x, {"ksize": 3})
            self.add_filter_widget(new_filter, {"ksize": [1, 3, 5, 7]})

        elif filter_name == "Собель Y":
            new_filter = Filter("Собель Y", filters.apply_sobel_y, {"ksize": 3})
            self.add_filter_widget(new_filter, {"ksize": [1, 3, 5, 7]})

        elif filter_name == "Собель XY":
            new_filter = Filter("Собель XY", filters.apply_sobel_xy, {"ksize": 3})
            self.add_filter_widget(new_filter, {"ksize": [1, 3, 5, 7]})

        elif filter_name == "Лаплас":
            new_filter = Filter("Лаплас", filters.apply_laplacian, {"ksize": 3})
            self.add_filter_widget(new_filter, {"ksize": [1, 3, 5, 7]})

        elif filter_name == "Scharr X":
            new_filter = Filter("Scharr X", filters.apply_scharr_x)
            self.add_filter_widget(new_filter)

        elif filter_name == "Scharr Y":
            new_filter = Filter("Scharr Y", filters.apply_scharr_y)
            self.add_filter_widget(new_filter)

        elif filter_name == "Scharr XY":
            new_filter = Filter("Scharr XY", filters.apply_scharr_xy)
            self.add_filter_widget(new_filter)

        elif filter_name == "Детектор углов Харриса":
            new_filter = Filter("Детектор углов Харриса", filters.apply_harris_corner,
                                {"block_size": 2, "ksize": 3, "k": 0.04})
            self.add_filter_widget(new_filter, {"block_size": (2, 10, 1),
                                                "ksize": [1, 3, 5, 7],
                                                "k": (0.01, 0.1, 0.01)})

        elif filter_name == "Детектор углов Ши-Томаси":
            new_filter = Filter("Детектор углов Ши-Томаси", filters.apply_shi_tomasi,
                                {"max_corners": 100, "quality_level": 0.01, "min_distance": 10})
            self.add_filter_widget(new_filter, {"max_corners": (10, 500, 10),
                                                "quality_level": (0.001, 0.1, 0.001),
                                                "min_distance": (1, 100, 1)})

        elif filter_name == "Выделение контуров":
            new_filter = Filter("Выделение контуров", filters.apply_find_contours,
                                {"mode": cv2.RETR_TREE, "method": cv2.CHAIN_APPROX_SIMPLE,
                                 "threshold1": 100, "threshold2": 200})
            self.add_filter_widget(new_filter, {"threshold1": (0, 500, 1),
                                                "threshold2": (0, 500, 1),
                                                "mode": ["RETR_EXTERNAL", "RETR_LIST", "RETR_TREE"],
                                                "method": ["CHAIN_APPROX_NONE", "CHAIN_APPROX_SIMPLE"]})

        self.filter_manager.add_filter(new_filter)

    def add_color_filter(self, filter_name):
        """Добавляет фильтр цветового преобразования."""
        if filter_name == "Оттенки серого":
            new_filter = Filter("Оттенки серого", filters.apply_grayscale)
            self.add_filter_widget(new_filter)

        elif filter_name == "Сепия":
            new_filter = Filter("Сепия", filters.apply_sepia, {"intensity": 1.0})
            self.add_filter_widget(new_filter, {"intensity": (0.0, 1.0, 0.05)})

        elif filter_name == "HSV":
            new_filter = Filter("HSV", filters.apply_hsv)
            self.add_filter_widget(new_filter)

        elif filter_name == "LAB":
            new_filter = Filter("LAB", filters.apply_lab)
            self.add_filter_widget(new_filter)

        elif filter_name == "YCrCb":
            new_filter = Filter("YCrCb", filters.apply_ycrcb)
            self.add_filter_widget(new_filter)

        elif filter_name == "Негатив":
            new_filter = Filter("Негатив", filters.apply_negative)
            self.add_filter_widget(new_filter)

        elif filter_name == "Изменение насыщенности":
            new_filter = Filter("Изменение насыщенности", filters.apply_saturation, {"scale": 1.5})
            self.add_filter_widget(new_filter, {"scale": (0.0, 3.0, 0.05)})

        elif filter_name == "Изменение оттенка":
            new_filter = Filter("Изменение оттенка", filters.apply_hue_shift, {"shift": 10})
            self.add_filter_widget(new_filter, {"shift": (0, 180, 1)})

        elif filter_name == "Повышение контраста CLAHE":
            new_filter = Filter("CLAHE", filters.apply_clahe,
                                {"clip_limit": 2.0, "tile_grid_size": 8})
            self.add_filter_widget(new_filter, {"clip_limit": (0.5, 10.0, 0.5),
                                                "tile_grid_size": (2, 16, 2)})

        elif filter_name == "Псевдоцвета (Jet)":
            new_filter = Filter("Псевдоцвета (Jet)", filters.apply_colormap_jet)
            self.add_filter_widget(new_filter)

        elif filter_name == "Псевдоцвета (Hot)":
            new_filter = Filter("Псевдоцвета (Hot)", filters.apply_colormap_hot)
            self.add_filter_widget(new_filter)

        elif filter_name == "Псевдоцвета (Cool)":
            new_filter = Filter("Псевдоцвета (Cool)", filters.apply_colormap_cool)
            self.add_filter_widget(new_filter)

        elif filter_name == "Изменение баланса RGB":
            new_filter = Filter("Изменение баланса RGB", filters.apply_rgb_balance,
                                {"r_scale": 1.0, "g_scale": 1.0, "b_scale": 1.0})
            self.add_filter_widget(new_filter, {"r_scale": (0.0, 2.0, 0.05),
                                                "g_scale": (0.0, 2.0, 0.05),
                                                "b_scale": (0.0, 2.0, 0.05)})

        self.filter_manager.add_filter(new_filter)

    def add_morphology_filter(self, filter_name):
        """Добавляет морфологический фильтр."""
        if filter_name == "Эрозия":
            new_filter = Filter("Эрозия", filters.apply_erosion, {"ksize": 5, "iterations": 1})
            self.add_filter_widget(new_filter, {"ksize": (1, 21, 2), "iterations": (1, 10, 1)})

        elif filter_name == "Дилатация":
            new_filter = Filter("Дилатация", filters.apply_dilation, {"ksize": 5, "iterations": 1})
            self.add_filter_widget(new_filter, {"ksize": (1, 21, 2), "iterations": (1, 10, 1)})

        elif filter_name == "Открытие":
            new_filter = Filter("Открытие", filters.apply_opening, {"ksize": 5, "iterations": 1})
            self.add_filter_widget(new_filter, {"ksize": (1, 21, 2), "iterations": (1, 10, 1)})

        elif filter_name == "Закрытие":
            new_filter = Filter("Закрытие", filters.apply_closing, {"ksize": 5, "iterations": 1})
            self.add_filter_widget(new_filter, {"ksize": (1, 21, 2), "iterations": (1, 10, 1)})

        elif filter_name == "Градиент":
            new_filter = Filter("Градиент", filters.apply_morph_gradient, {"ksize": 5})
            self.add_filter_widget(new_filter, {"ksize": (1, 21, 2)})

        elif filter_name == "Верхняя шляпа":
            new_filter = Filter("Верхняя шляпа", filters.apply_tophat, {"ksize": 5})
            self.add_filter_widget(new_filter, {"ksize": (1, 21, 2)})

        elif filter_name == "Черная шляпа":
            new_filter = Filter("Черная шляпа", filters.apply_blackhat, {"ksize": 5})
            self.add_filter_widget(new_filter, {"ksize": (1, 21, 2)})

        elif filter_name == "Скелетизация":
            new_filter = Filter("Скелетизация", filters.apply_skeleton, {"iterations": 10})
            self.add_filter_widget(new_filter, {"iterations": (1, 30, 1)})

        elif filter_name == "Утончение":
            new_filter = Filter("Утончение", filters.apply_thinning, {"iterations": 10})
            self.add_filter_widget(new_filter, {"iterations": (1, 30, 1)})

        self.filter_manager.add_filter(new_filter)

    def add_transform_filter(self, filter_name):
        """Добавляет геометрический фильтр трансформации."""
        if filter_name == "Масштабирование":
            new_filter = Filter("Масштабирование", filters.apply_resize, {"scale": 1.0})
            self.add_filter_widget(new_filter, {"scale": (0.1, 3.0, 0.1)})
        elif filter_name == "Поворот":
            new_filter = Filter("Поворот", filters.apply_rotate, {"angle": 0})
            self.add_filter_widget(new_filter, {"angle": (-180, 180, 1)})
        elif filter_name == "Отражение":
            new_filter = Filter("Отражение", filters.apply_flip, {"mode": "Горизонтально"})
            self.add_filter_widget(new_filter, {"mode": ["Вертикально", "Горизонтально", "Оба"]})
        elif filter_name == "Сдвиг":
            new_filter = Filter("Сдвиг", filters.apply_translate, {"x_shift": 0, "y_shift": 0})
            self.add_filter_widget(new_filter, {"x_shift": (-100, 100, 1), "y_shift": (-100, 100, 1)})
        elif filter_name == "Перспективное преобразование":
            new_filter = Filter("Перспективное преобразование", filters.apply_perspective_transform)
            self.add_filter_widget(new_filter)
        elif filter_name == "Аффинное преобразование":
            new_filter = Filter("Аффинное преобразование", filters.apply_affine_transform)
            self.add_filter_widget(new_filter)
        elif filter_name == "Коррекция перспективы":
            new_filter = Filter("Коррекция перспективы", filters.apply_perspective_correction)
            self.add_filter_widget(new_filter)

        self.filter_manager.add_filter(new_filter)

    def add_enhancement_filter(self, filter_name):
        """Добавляет фильтр улучшения изображения."""
        if filter_name == "Гамма-коррекция":
            new_filter = Filter("Гамма-коррекция", filters.apply_gamma_correction, {"gamma": 1.0})
            self.add_filter_widget(new_filter, {"gamma": (0.1, 5.0, 0.1)})
        elif filter_name == "Яркость/Контраст":
            new_filter = Filter("Яркость/Контраст", filters.apply_brightness_contrast, {"brightness": 0, "contrast": 0})
            self.add_filter_widget(new_filter, {"brightness": (-100, 100, 1), "contrast": (-100, 100, 1)})
        elif filter_name == "Автоматический баланс белого":
            new_filter = Filter("Баланс белого", filters.apply_auto_white_balance)
            self.add_filter_widget(new_filter)
        elif filter_name == "Автоконтраст":
            new_filter = Filter("Автоконтраст", filters.apply_auto_contrast)
            self.add_filter_widget(new_filter)
        elif filter_name == "Поиск и удаление шума":
            new_filter = Filter("Удаление шума", filters.apply_denoise)
            self.add_filter_widget(new_filter)
        elif filter_name == "Повышение резкости":
            new_filter = Filter("Резкость", filters.apply_sharpen)
            self.add_filter_widget(new_filter)
        elif filter_name == "Детализация":
            new_filter = Filter("Детализация", filters.apply_detail)
            self.add_filter_widget(new_filter)
        elif filter_name == "HDR-эффект":
            new_filter = Filter("HDR-эффект", filters.apply_hdr)
            self.add_filter_widget(new_filter)
        elif filter_name == "Выравнивание гистограммы":
            new_filter = Filter("Гистограмма", filters.apply_histogram_equalization)
            self.add_filter_widget(new_filter)

        self.filter_manager.add_filter(new_filter)

    def add_segmentation_filter(self, filter_name):
        """Добавляет фильтр сегментации."""
        if filter_name == "Бинаризация (пороговая)":
            new_filter = Filter("Бинаризация", filters.apply_threshold, {"thresh": 127})
            self.add_filter_widget(new_filter, {"thresh": (0, 255, 1)})
        elif filter_name == "Адаптивный порог":
            new_filter = Filter("Адаптивный порог", filters.apply_adaptive_threshold, {"block_size": 11, "C": 2})
            self.add_filter_widget(new_filter, {"block_size": (3, 31, 2), "C": (-10, 10, 1)})
        elif filter_name == "Порог Отсу":
            new_filter = Filter("Порог Отсу", filters.apply_otsu_threshold)
            self.add_filter_widget(new_filter)
        elif filter_name == "Водораздел":
            new_filter = Filter("Водораздел", filters.apply_watershed)
            self.add_filter_widget(new_filter)
        elif filter_name == "Mean Shift":
            new_filter = Filter("Mean Shift", filters.apply_mean_shift)
            self.add_filter_widget(new_filter)
        elif filter_name == "K-Means":
            new_filter = Filter("K-Means", filters.apply_kmeans, {"K": 4})
            self.add_filter_widget(new_filter, {"K": (2, 10, 1)})
        elif filter_name == "GrabCut":
            new_filter = Filter("GrabCut", filters.apply_grabcut)
            self.add_filter_widget(new_filter)
        elif filter_name == "Детектор движений":
            new_filter = Filter("Детектор движений", filters.apply_motion_detector)
            self.add_filter_widget(new_filter)

        self.filter_manager.add_filter(new_filter)

    def add_feature_filter(self, filter_name):
        """Добавляет фильтр выделения признаков."""
        if filter_name == "SIFT-дескриптор":
            new_filter = Filter("SIFT", filters.apply_sift)
            self.add_filter_widget(new_filter)
        elif filter_name == "SURF-дескриптор":
            new_filter = Filter("SURF", filters.apply_surf)
            self.add_filter_widget(new_filter)
        elif filter_name == "ORB-дескриптор":
            new_filter = Filter("ORB", filters.apply_orb)
            self.add_filter_widget(new_filter)
        elif filter_name == "HOG-дескриптор":
            new_filter = Filter("HOG", filters.apply_hog)
            self.add_filter_widget(new_filter)
        elif filter_name == "Детектор лиц Haar Cascade":
            new_filter = Filter("Лицо Haar", filters.apply_face_detection)
            self.add_filter_widget(new_filter)
        elif filter_name == "Детектор глаз Haar Cascade":
            new_filter = Filter("Глаза Haar", filters.apply_eye_detection)
            self.add_filter_widget(new_filter)
        elif filter_name == "Выделение линий Хафа":
            new_filter = Filter("Линии Хафа", filters.apply_hough_lines)
            self.add_filter_widget(new_filter)
        elif filter_name == "Выделение окружностей Хафа":
            new_filter = Filter("Окружности Хафа", filters.apply_hough_circles)
            self.add_filter_widget(new_filter)

        self.filter_manager.add_filter(new_filter)

    def add_custom_filter(self, filter_name):
        """Добавляет пользовательский фильтр."""
        if filter_name == "Карта глубины":
            new_filter = Filter("Карта глубины", filters.apply_depth_map)
            self.add_filter_widget(new_filter)
        elif filter_name == "Мультиплексирование каналов":
            new_filter = Filter("Мультиплексирование", filters.apply_channel_multiplex)
            self.add_filter_widget(new_filter)
        elif filter_name == "Картунизация (Cartoon-эффект)":
            new_filter = Filter("Картунизация", filters.apply_cartoon)
            self.add_filter_widget(new_filter)
        elif filter_name == "Пикселизация":
            new_filter = Filter("Пикселизация", filters.apply_pixelation, {"block_size": 10})
            self.add_filter_widget(new_filter, {"block_size": (2, 50, 1)})
        elif filter_name == "Виньетка":
            new_filter = Filter("Виньетка", filters.apply_vignette)
            self.add_filter_widget(new_filter)
        elif filter_name == "Размытие в движении":
            new_filter = Filter("Размытие в движении", filters.apply_motion_blur, {"kernel_size": 15})
            self.add_filter_widget(new_filter, {"kernel_size": (3, 50, 2)})
        elif filter_name == "Акварельный эффект":
            new_filter = Filter("Акварель", filters.apply_watercolor)
            self.add_filter_widget(new_filter)
        elif filter_name == "Эффект масляной живописи":
            new_filter = Filter("Масляная живопись", filters.apply_oil_painting)
            self.add_filter_widget(new_filter)
        elif filter_name == "Стилизация под карандашный рисунок":
            new_filter = Filter("Карандаш", filters.apply_pencil_sketch)
            self.add_filter_widget(new_filter)

        self.filter_manager.add_filter(new_filter)

    def update_preview(self):
        """Обновляет предпросмотр изображения с примененными фильтрами."""
        try:
            img = self.filter_manager.apply_filters()
            if img is not None:
                qt_img = self.convert_cv_qt(img)
                self.filtered_label.setPixmap(qt_img)
                logger.debug("Превью обновлено успешно")
        except Exception as e:
            logger.error(f"Ошибка при обновлении превью: {e}")
            QMessageBox.warning(self, "Ошибка", f"Не удалось обновить изображение: {e}")

    def convert_cv_qt(self, cv_img):
        """Конвертирует изображение из формата OpenCV в QPixmap."""
        if cv_img.ndim == 2:
            qformat = QImage.Format_Grayscale8
        else:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            qformat = QImage.Format_RGB888
        height, width = cv_img.shape[:2]
        bytes_per_line = 3 * width if cv_img.ndim == 3 else width
        q_img = QImage(cv_img.data, width, height, bytes_per_line, qformat)
        q_pixmap = QPixmap.fromImage(q_img).scaled(self.filtered_label.width(), self.filtered_label.height(),
                                                   Qt.KeepAspectRatio)
        return q_pixmap

    def reset_filters(self):
        """Сбрасывает все примененные фильтры."""
        self.filter_manager.reset_filters()
        for i in reversed(range(self.filters_layout.count())):
            widget = self.filters_layout.takeAt(i).widget()
            if widget is not None:
                widget.deleteLater()

    def save_image(self):
        """Сохраняет изображение с примененными фильтрами."""
        try:
            if self.filter_manager.get_current_image() is None:
                return
            file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить изображение", "",
                                                       "Изображения (*.png *.jpg *.jpeg *.bmp *.tiff)")
            if file_path:
                logger.info(f"Сохранение изображения: {file_path}")
                cv2.imwrite(file_path, self.filter_manager.get_current_image())
                logger.info(f"Изображение успешно сохранено: {file_path}")
                QMessageBox.information(self, "Информация", f"Изображение успешно сохранено в {file_path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении изображения: {e}")
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при сохранении изображения: {e}")

    def add_filter_widget(self, filter_obj, widget_params=None):
        """Добавляет виджет для управления параметрами фильтра."""
        container = QGroupBox(filter_obj.name)
        layout = QVBoxLayout()

        # Флажок активации фильтра
        checkbox = QCheckBox("Активен")
        checkbox.setChecked(True)
        checkbox.stateChanged.connect(lambda state: self.filter_manager.toggle_filter(filter_obj, state == Qt.Checked))
        layout.addWidget(checkbox)

        if widget_params:
            for param, config in widget_params.items():
                h_layout = QHBoxLayout()
                label = QLabel(param)
                h_layout.addWidget(label)

                if isinstance(config, tuple) and len(config) == 3:
                    min_val, max_val, step = config
                    default = filter_obj.params.get(param, min_val)
                    if isinstance(default, float) or isinstance(step, float):
                        spin = QDoubleSpinBox()
                        spin.setDecimals(2)
                    else:
                        spin = QSpinBox()
                    spin.setMinimum(min_val)
                    spin.setMaximum(max_val)
                    spin.setSingleStep(step)
                    spin.setValue(default)
                    spin.valueChanged.connect(
                        lambda val, p=param: self.filter_manager.update_filter_param(filter_obj, p, val))
                    h_layout.addWidget(spin)
                elif isinstance(config, list):
                    combo = QComboBox()
                    combo.addItems(config)
                    combo.currentTextChanged.connect(
                        lambda val, p=param: self.filter_manager.update_filter_param(filter_obj, p, val))
                    h_layout.addWidget(combo)
                layout.addLayout(h_layout)

        remove_btn = QPushButton("Удалить фильтр")
        remove_btn.clicked.connect(lambda: self.remove_filter(filter_obj, container))
        layout.addWidget(remove_btn)

        container.setLayout(layout)
        self.filters_layout.addWidget(container)

    def remove_filter(self, filter_obj, widget):
        """Удаляет фильтр из списка активных фильтров."""
        self.filter_manager.remove_filter(filter_obj)
        widget.deleteLater()