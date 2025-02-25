import numpy as np
import traceback
from signals import FilterSignals
from logger import logger, log_filter_error


class Filter:
    """
    Класс фильтра, содержащий информацию о фильтре и его параметрах.
    """
    def __init__(self, name, function, params=None):
        self.name = name
        self.function = function
        self.params = params or {}
        self.is_active = True

    def apply(self, image):
        """
        Применяет фильтр к изображению.
        """
        if not self.is_active:
            return image
        try:
            logger.info(f"Применение фильтра: {self.name}")
            return self.function(image.copy(), **self.params)
        except Exception as e:
            stack_trace = traceback.format_exc()
            log_filter_error(self.name, e, stack_trace)
            # Возвращаем исходное изображение в случае ошибки
            return image

    def __str__(self):
        return self.name


class FilterManager:
    """
    Класс для управления фильтрами: добавление, удаление, применение фильтров к изображению.
    """
    def __init__(self):
        self.filters = []
        self.signals = FilterSignals()
        self.original_image = None
        self.current_image = None

    def set_original_image(self, image):
        """
        Устанавливает оригинальное изображение.
        """
        self.original_image = image
        self.current_image = image.copy() if image is not None else None

    def add_filter(self, new_filter):
        """
        Добавляет новый фильтр в список.
        """
        self.filters.append(new_filter)
        self.signals.updated.emit()

    def remove_filter(self, filter_obj):
        """
        Удаляет фильтр из списка.
        """
        if filter_obj in self.filters:
            self.filters.remove(filter_obj)
            self.signals.updated.emit()

    def toggle_filter(self, filter_obj, is_active):
        """
        Включает или выключает фильтр.
        """
        filter_obj.is_active = is_active
        self.signals.updated.emit()

    def update_filter_param(self, filter_obj, param, value):
        """
        Обновляет параметр фильтра.
        """
        if "_" in param:
            base = param.split("_")[0]
            current = filter_obj.params.get(base, (0, 0))
            if isinstance(current, tuple) and len(current) == 2:
                if param.endswith("x"):
                    filter_obj.params[base] = (value, current[1])
                elif param.endswith("y"):
                    filter_obj.params[base] = (current[0], value)
        else:
            filter_obj.params[param] = value
        self.signals.updated.emit()

    def reset_filters(self):
        """
        Сбрасывает все фильтры.
        """
        self.filters = []
        self.signals.updated.emit()

    def apply_filters(self):
        """
        Применяет все активные фильтры к оригинальному изображению.
        """
        if self.original_image is None:
            return None

        img = self.original_image.copy()
        for filt in self.filters:
            img = filt.apply(img)
        self.current_image = img
        return img

    def get_original_image(self):
        """
        Возвращает оригинальное изображение.
        """
        return self.original_image

    def get_current_image(self):
        """
        Возвращает текущее изображение с примененными фильтрами.
        """
        return self.current_image