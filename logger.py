import logging
import os
import sys
import traceback
from datetime import datetime

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)


def setup_logger():
    log_filename = os.path.join(log_dir, f"opencv_filters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Настройка основного логгера
    logger = logging.getLogger("opencv_filters")
    logger.setLevel(logging.DEBUG)

    # Форматирование логов
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Вывод в файл
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Вывод в консоль
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Добавляем обработчики к логгеру
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger



logger = setup_logger()


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Необработанное исключение:", exc_info=(exc_type, exc_value, exc_traceback))

    error_msg = f"Произошла критическая ошибка: {exc_type.__name__}: {exc_value}"
    print(error_msg)

    logger.debug("Стек вызовов:")
    for line in traceback.format_tb(exc_traceback):
        logger.debug(line.strip())


sys.excepthook = handle_exception

def log_filter_error(filter_name, error, stack_trace=None):
    """
    Логирует ошибку при применении фильтра

    Args:
        filter_name (str): Имя фильтра
        error (Exception): Объект исключения
        stack_trace (str, optional): Стек вызовов
    """
    logger.error(f"Ошибка в фильтре '{filter_name}': {error}")
    if stack_trace:
        logger.debug(f"Стек вызовов для '{filter_name}':")
        for line in stack_trace.split('\n'):
            logger.debug(line.strip())