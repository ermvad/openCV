import os
import aiosqlite
import json
import time
from datetime import datetime
import uuid
from logger import logger

# Путь к базе данных статистики
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats.db")

async def init_db():
    """Инициализирует базу данных для веб-приложения."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            # Таблица сессий пользователей
            await db.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                ip_address TEXT,
                user_agent TEXT,
                image_count INTEGER DEFAULT 0
            )
            ''')
            
            # Таблица использования фильтров
            await db.execute('''
            CREATE TABLE IF NOT EXISTS filter_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                filter_name TEXT NOT NULL,
                filter_category TEXT NOT NULL,
                parameters TEXT,
                execution_time REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN DEFAULT 1
            )
            ''')
            
            # Таблица информации об обрабатываемых изображениях
            await db.execute('''
            CREATE TABLE IF NOT EXISTS image_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                image_id TEXT NOT NULL,
                original_filename TEXT,
                width INTEGER,
                height INTEGER,
                channels INTEGER,
                file_size_kb REAL,
                filter_count INTEGER DEFAULT 0,
                processing_time REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Таблица общих событий приложения
            await db.execute('''
            CREATE TABLE IF NOT EXISTS app_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                session_id TEXT,
                description TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            await db.commit()
            logger.info("База данных статистики успешно инициализирована")
            return True
    except Exception as e:
        logger.error(f"Ошибка при инициализации базы данных статистики: {e}")
        return False

async def create_session(ip_address=None, user_agent=None):
    """
    Создает новую сессию пользователя.
    
    Args:
        ip_address: IP-адрес пользователя
        user_agent: User-Agent браузера пользователя
    
    Returns:
        str: ID сессии
    """
    try:
        session_id = str(uuid.uuid4())
        
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO sessions (session_id, ip_address, user_agent) VALUES (?, ?, ?)",
                (session_id, ip_address, user_agent)
            )
            await db.commit()
            
            # Логируем событие
            await log_app_event("session_start", session_id, f"Новая сессия начата")
            
            logger.info(f"Создана новая сессия: {session_id}")
            return session_id
    except Exception as e:
        logger.error(f"Ошибка при создании сессии: {e}")
        return None

async def end_session(session_id):
    """
    Завершает сессию пользователя.
    
    Args:
        session_id: ID сессии
    """
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE sessions SET end_time = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session_id,)
            )
            await db.commit()
            
            # Получаем статистику сессии
            async with db.execute(
                "SELECT start_time, image_count FROM sessions WHERE session_id = ?",
                (session_id,)
            ) as cursor:
                session_data = await cursor.fetchone()
            
            if session_data:
                start_time, image_count = session_data
                description = f"Сессия завершена. Обработано изображений: {image_count}"
                await log_app_event("session_end", session_id, description)
            
            logger.info(f"Сессия {session_id} завершена")
    except Exception as e:
        logger.error(f"Ошибка при завершении сессии: {e}")

async def log_filter_usage(session_id, filter_name, filter_category, parameters=None, execution_time=0, success=True):
    """
    Логирует использование фильтра.
    
    Args:
        session_id: ID сессии
        filter_name: Название фильтра
        filter_category: Категория фильтра
        parameters: Параметры фильтра (словарь или список)
        execution_time: Время выполнения в миллисекундах
        success: Успешно ли выполнен фильтр
    """
    try:
        # Преобразуем параметры в JSON-строку
        if parameters:
            if isinstance(parameters, (dict, list)):
                params_json = json.dumps(parameters)
            else:
                params_json = str(parameters)
        else:
            params_json = None
        
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """INSERT INTO filter_usage 
                (session_id, filter_name, filter_category, parameters, execution_time, success) 
                VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, filter_name, filter_category, params_json, execution_time, success)
            )
            await db.commit()
            
            logger.debug(f"Использование фильтра {filter_name} записано в БД")
            
            # Обновляем счетчик фильтров для текущего изображения в сессии
            # Находим последнее обработанное изображение в сессии
            async with db.execute(
                """SELECT id FROM image_stats 
                WHERE session_id = ? 
                ORDER BY timestamp DESC LIMIT 1""",
                (session_id,)
            ) as cursor:
                image_record = await cursor.fetchone()
            
            if image_record:
                image_id = image_record[0]
                await db.execute(
                    "UPDATE image_stats SET filter_count = filter_count + 1 WHERE id = ?",
                    (image_id,)
                )
                await db.commit()
    except Exception as e:
        logger.error(f"Ошибка при записи использования фильтра в БД: {e}")

async def log_image_processing(session_id, image_id, original_filename=None, width=0, height=0, 
                              channels=0, file_size_kb=0, processing_time=0):
    """
    Логирует информацию об обработке изображения.
    
    Args:
        session_id: ID сессии
        image_id: ID изображения
        original_filename: Оригинальное имя файла
        width: Ширина изображения
        height: Высота изображения
        channels: Количество каналов
        file_size_kb: Размер файла в КБ
        processing_time: Время обработки в миллисекундах
    """
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """INSERT INTO image_stats 
                (session_id, image_id, original_filename, width, height, channels, file_size_kb, processing_time) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, image_id, original_filename, width, height, channels, file_size_kb, processing_time)
            )
            await db.commit()
            
            # Увеличиваем счетчик изображений в сессии
            await db.execute(
                "UPDATE sessions SET image_count = image_count + 1 WHERE session_id = ?",
                (session_id,)
            )
            await db.commit()
            
            logger.debug(f"Информация об изображении {image_id} записана в БД")
    except Exception as e:
        logger.error(f"Ошибка при записи информации об изображении в БД: {e}")

async def log_app_event(event_type, session_id=None, description=None):
    """
    Логирует событие приложения.
    
    Args:
        event_type: Тип события
        session_id: ID сессии
        description: Описание события
    """
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO app_events (event_type, session_id, description) VALUES (?, ?, ?)",
                (event_type, session_id, description)
            )
            await db.commit()
            
            logger.debug(f"Событие {event_type} записано в БД")
    except Exception as e:
        logger.error(f"Ошибка при записи события в БД: {e}")

async def get_filter_stats():
    """
    Получает статистику использования фильтров.
    
    Returns:
        list: Список статистики использования фильтров
    """
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT filter_category, filter_name, COUNT(*) as usage_count, 
                AVG(execution_time) as avg_execution_time,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as error_count
                FROM filter_usage 
                GROUP BY filter_category, filter_name
                ORDER BY usage_count DESC"""
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Ошибка при получении статистики фильтров: {e}")
        return []

async def get_app_stats():
    """
    Получает общую статистику приложения.
    
    Returns:
        dict: Словарь с общей статистикой
    """
    try:
        stats = {}
        
        async with aiosqlite.connect(DB_PATH) as db:
            # Общее количество сессий
            async with db.execute("SELECT COUNT(*) FROM sessions") as cursor:
                stats['total_sessions'] = (await cursor.fetchone())[0]
                
            # Общее количество обработанных изображений
            async with db.execute("SELECT COUNT(*) FROM image_stats") as cursor:
                stats['total_images'] = (await cursor.fetchone())[0]
                
            # Общее количество примененных фильтров
            async with db.execute("SELECT COUNT(*) FROM filter_usage") as cursor:
                stats['total_filters_applied'] = (await cursor.fetchone())[0]
                
            # Среднее время обработки изображения
            async with db.execute("SELECT AVG(processing_time) FROM image_stats") as cursor:
                stats['avg_image_processing_time'] = (await cursor.fetchone())[0] or 0
                
            # Среднее время применения фильтра
            async with db.execute("SELECT AVG(execution_time) FROM filter_usage") as cursor:
                stats['avg_filter_execution_time'] = (await cursor.fetchone())[0] or 0
                
            # Топ-5 самых популярных фильтров
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT filter_name, COUNT(*) as count 
                FROM filter_usage 
                GROUP BY filter_name 
                ORDER BY count DESC LIMIT 5"""
            ) as cursor:
                rows = await cursor.fetchall()
                stats['top_filters'] = [dict(row) for row in rows]
                
            # Активность по дням
            async with db.execute(
                """SELECT DATE(timestamp) as date, COUNT(*) as count 
                FROM filter_usage 
                GROUP BY DATE(timestamp) 
                ORDER BY date DESC LIMIT 7"""
            ) as cursor:
                rows = await cursor.fetchall()
                stats['daily_activity'] = [dict(row) for row in rows]
                
            return stats
    except Exception as e:
        logger.error(f"Ошибка при получении общей статистики: {e}")
        return {}
