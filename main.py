import sys
from PyQt5.QtWidgets import QApplication
from ui import OpenCVFilters
from config import DARK_STYLE
from logger import logger

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        app.setStyleSheet(DARK_STYLE)
        logger.info("Тёмная тема применена")

        window = OpenCVFilters()
        window.setStyleSheet(DARK_STYLE)
        window.show()

        exit_code = app.exec_()
        logger.info(f"Приложение завершило работу с кодом: {exit_code}")
        sys.exit(exit_code)

    except Exception as e:
        logger.critical(f"Критическая ошибка при запуске приложения: {e}")
        raise