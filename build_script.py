#!/usr/bin/env python3
"""
Скрипт для компиляции OpenCV Filters в исполняемый файл с помощью Nuitka.
"""

import os
import sys
import shutil
import platform
import subprocess
from datetime import datetime

# Текущая директория проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Настройки сборки
OUTPUT_DIR = os.path.join(BASE_DIR, "build")
DIST_DIR = os.path.join(OUTPUT_DIR, "dist")
ICON_PATH = os.path.join(BASE_DIR, "icon.ico")
APP_NAME = "OpenCV_Filters"
VERSION = "1.0.0"

# Создаем папку сборки, если не существует
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(message):
    """Вывод сообщений с меткой времени"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def check_requirements():
    """Проверка наличия необходимых зависимостей"""
    log("Проверка зависимостей...")
    
    # Проверяем Nuitka
    try:
        subprocess.run([sys.executable, "-m", "nuitka", "--version"], 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        log("Ошибка: Nuitka не установлен. Установите его командой:")
        log("pip install nuitka")
        return False
    
    # Проверяем зависимости
    try:
        import PyQt5
        import cv2
        import numpy
    except ImportError as e:
        log(f"Ошибка: Отсутствует зависимость: {e}")
        log("Установите все зависимости командой:")
        log("pip install PyQt5 opencv-python numpy")
        return False
    
    log("Все зависимости установлены.")
    return True

def clean_build_dir():
    """Очистка директории сборки"""
    log("Очистка директории сборки...")
    
    # Удаляем только содержимое директории, сохраняя саму директорию
    for item in os.listdir(OUTPUT_DIR):
        item_path = os.path.join(OUTPUT_DIR, item)
        try:
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            log(f"Ошибка при удалении {item_path}: {e}")

def build_application(onefile=True, console=False):
    """Сборка приложения с помощью Nuitka"""
    log("Начало компиляции приложения...")
    
    # Основные опции Nuitka
    nuitka_options = [
        "--standalone",                  # Автономное приложение
        "--enable-plugin=pyqt5",         # Поддержка PyQt5
        f"--output-dir={OUTPUT_DIR}",    # Директория вывода
        "--follow-imports",              # Следовать за импортами
        "--show-progress",               # Отображать прогресс
        "--include-package=cv2",         # Включить OpenCV
        "--include-package=numpy",       # Включить NumPy
        "--include-data-dir=logs=logs",  # Включить директорию logs
    ]
    
    # Дополнительные опции
    if platform.system() == "Windows":
        nuitka_options.append("--mingw64")  # Использовать MinGW64 на Windows
        if os.path.exists(ICON_PATH):
            nuitka_options.append(f"--windows-icon-from-ico={ICON_PATH}")
        if not console:
            nuitka_options.append("--windows-disable-console")
    
    # Оптимизации
    nuitka_options.extend([
        "--lto=yes",                     # Link Time Optimization
        f"--jobs={os.cpu_count() + 1}",  # Многопоточная компиляция
        "--remove-output",               # Удалить временные файлы после сборки
    ])
    
    # Режим onefile (один исполняемый файл)
    if onefile:
        nuitka_options.append("--onefile")
        nuitka_options.append(f"--output-filename={APP_NAME}")
    
    # Финальная команда
    cmd = [sys.executable, "-m", "nuitka"] + nuitka_options + ["main.py"]
    
    log("Запуск компиляции со следующими параметрами:")
    log(" ".join(cmd))
    
    # Запуск процесса компиляции
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                  universal_newlines=True, bufsize=1)
        
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode == 0:
            log("Компиляция успешно завершена!")
            return True
        else:
            log(f"Ошибка компиляции! Код возврата: {process.returncode}")
            return False
            
    except Exception as e:
        log(f"Ошибка во время компиляции: {e}")
        return False

def copy_additional_files():
    """Копирование дополнительных файлов в директорию сборки"""
    log("Копирование дополнительных файлов...")
    
    # Создаем README.txt с информацией об установке
    readme_content = f"""
OpenCV Filters {VERSION}
=======================

Спасибо за использование OpenCV Filters!

Запуск:
- Просто запустите {APP_NAME}.exe

При первом запуске может потребоваться некоторое время для извлечения библиотек.

Системные требования:
- Windows 7/8/10/11 (64-bit)
- Минимум 4 ГБ ОЗУ
- Поддержка OpenGL 2.0+

При возникновении проблем посетите репозиторий проекта или обратитесь к разработчику.
"""
    
    # Определяем директорию, куда копировать файлы
    target_dir = DIST_DIR if os.path.exists(DIST_DIR) else OUTPUT_DIR
    
    # Записываем README.txt
    with open(os.path.join(target_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    log("Дополнительные файлы скопированы.")

def main():
    """Основная функция сборки"""
    log(f"Начало сборки {APP_NAME} v{VERSION}")
    
    # Проверка зависимостей
    if not check_requirements():
        return 1
    
    # Спрашиваем пользователя о параметрах сборки
    onefile = input("Собрать в один файл? (y/n) [y]: ").lower() != 'n'
    console = input("Показывать консоль? (y/n) [n]: ").lower() == 'y'
    clean = input("Очистить директорию сборки перед началом? (y/n) [y]: ").lower() != 'n'
    
    # Очистка директории сборки
    if clean:
        clean_build_dir()
    
    # Сборка приложения
    if not build_application(onefile, console):
        return 1
    
    # Копирование дополнительных файлов
    copy_additional_files()
    
    log(f"Сборка {APP_NAME} v{VERSION} успешно завершена!")
    log(f"Исполняемый файл находится в директории: {OUTPUT_DIR}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
