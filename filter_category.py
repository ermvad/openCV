class FilterCategory:
    """Категории фильтров в приложении."""
    BLUR = "Размытие и сглаживание"
    EDGES = "Обнаружение краев и контуров"
    COLOR = "Цветовые преобразования"
    MORPHOLOGY = "Морфологические операции"
    TRANSFORM = "Геометрические преобразования"
    ENHANCEMENT = "Улучшение изображения"
    SEGMENTATION = "Сегментация и пороги"
    FEATURE = "Выделение признаков"
    CUSTOM = "Пользовательские фильтры"

    @classmethod
    def get_all(cls):
        """Возвращает список всех категорий."""
        return [
            cls.BLUR,
            cls.EDGES,
            cls.COLOR,
            cls.MORPHOLOGY,
            cls.TRANSFORM,
            cls.ENHANCEMENT,
            cls.SEGMENTATION,
            cls.FEATURE,
            cls.CUSTOM
        ]