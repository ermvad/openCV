from PyQt5.QtCore import QObject, pyqtSignal

class FilterSignals(QObject):
    """Класс сигналов для обновления интерфейса при изменении фильтров."""
    updated = pyqtSignal()