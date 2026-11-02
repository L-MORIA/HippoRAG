"""
Утилита для автоматического управления LM Studio и моделями.
Проверяет запущен ли LM Studio, запускает его при необходимости,
загружает необходимые модели.
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class LMStudioManager:
    """Менеджер для управления LM Studio и моделями."""

    def __init__(self, base_url: str = "http://localhost:1234", project_root: Optional[str] = None):
        """
        Инициализация менеджера.

        Args:
            base_url: URL LM Studio API
            project_root: Корневая папка проекта (для хранения моделей)
        """
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/v1"

        if project_root is None:
            # Определяем корень проекта (3 уровня вверх от этого файла)
            project_root = Path(__file__).parent.parent.parent.parent
        self.project_root = Path(project_root)
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(exist_ok=True)

        # Пути к LM Studio на Windows
        self.lm_studio_paths = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "LM Studio" / "LM Studio.exe",
            Path(os.environ.get("PROGRAMFILES", "")) / "LM Studio" / "LM Studio.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", "")) / "LM Studio" / "LM Studio.exe",
        ]

    def is_lm_studio_running(self) -> bool:
        """Проверяет, запущен ли LM Studio."""
        try:
            response = requests.get(f"{self.api_url}/models", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def find_lm_studio_executable(self) -> Optional[Path]:
        """Находит исполняемый файл LM Studio."""
        for path in self.lm_studio_paths:
            if path.exists():
                logger.info(f"Найден LM Studio: {path}")
                return path

        logger.warning("LM Studio не найден в стандартных путях")
        return None

    def start_lm_studio(self, wait_time: int = 10) -> bool:
        """
        Запускает LM Studio.

        Args:
            wait_time: Время ожидания запуска в секундах

        Returns:
            True если успешно запущен, False иначе
        """
        if self.is_lm_studio_running():
            logger.info("LM Studio уже запущен")
            return True

        lm_studio_exe = self.find_lm_studio_executable()
        if lm_studio_exe is None:
            logger.error("Не удалось найти LM Studio. Установите его с https://lmstudio.ai/")
            return False

        try:
            logger.info(f"Запуск LM Studio: {lm_studio_exe}")
            # Запускаем LM Studio в фоновом режиме
            subprocess.Popen(
                [str(lm_studio_exe)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )

            # Ждем запуска
            logger.info(f"Ожидание запуска LM Studio ({wait_time} сек)...")
            for i in range(wait_time):
                time.sleep(1)
                if self.is_lm_studio_running():
                    logger.info("LM Studio успешно запущен")
                    return True
                if i % 3 == 0:
                    logger.info(f"Ожидание... {i}/{wait_time} сек")

            logger.warning(
                "LM Studio запущен, но API не отвечает. Возможно, требуется больше времени."
            )
            return False

        except Exception as e:
            logger.error(f"Ошибка при запуске LM Studio: {e}")
            return False

    def get_loaded_model(self) -> Optional[str]:
        """Получает имя загруженной модели в LM Studio."""
        try:
            response = requests.get(f"{self.api_url}/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("data") and len(data["data"]) > 0:
                    model_id = data["data"][0].get("id")
                    logger.info(f"Загружена модель: {model_id}")
                    return model_id
            return None
        except Exception as e:
            logger.error(f"Ошибка при получении списка моделей: {e}")
            return None

    def load_model_in_lm_studio(self, model_name: str, max_retries: int = 3) -> bool:
        """
        Загружает модель в LM Studio через API (или ждёт ручной загрузки).

        Пытается загрузить модель программно (LM Studio 0.3+), если API не
        поддерживает — ждёт, пока пользователь загрузит вручную.

        Args:
            model_name: Имя модели для загрузки
            max_retries: Максимальное количество попыток

        Returns:
            True если модель загружена, False иначе
        """
        loaded_model = self.get_loaded_model()
        if loaded_model and model_name in loaded_model:
            logger.info(f"Модель {model_name} уже загружена")
            return True

        # 1. Попытка программной загрузки (LM Studio 0.3+)
        logger.info(f"Попытка автоматической загрузки модели {model_name}...")
        try:
            resp = requests.post(
                f"{self.api_url}/models/load",
                json={"model": model_name},
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info(f"Модель {model_name} загружена через API")
                time.sleep(3)  # даём время на инициализацию
                return True
        except Exception:
            pass

        # 2. Альтернативный эндпоинт (некоторые сборки LM Studio)
        try:
            resp = requests.post(
                f"{self.base_url}/models/load",
                json={"model": model_name},
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info(f"Модель {model_name} загружена через альтернативный API")
                time.sleep(3)
                return True
        except Exception:
            pass

        # 3. Ручная загрузка
        logger.warning(f"Модель {model_name} не загружена в LM Studio")
        logger.warning("Пожалуйста, загрузите модель вручную:")
        logger.warning("1. Откройте LM Studio")
        logger.warning("2. Перейдите в раздел 'Local Server'")
        logger.warning(f"3. Выберите модель '{model_name}' и нажмите 'Start Server'")
        logger.info("Ожидание загрузки модели...")

        for i in range(max_retries * 10):
            time.sleep(3)
            loaded_model = self.get_loaded_model()
            if loaded_model:
                logger.info(f"Модель загружена: {loaded_model}")
                return True
            if i % 5 == 0:
                logger.info(f"Ожидание... ({i*3} сек)")

        logger.error("Модель не была загружена. Продолжение без LLM.")
        return False

    def ensure_lm_studio_ready(self, model_name: str = "gemma-4-e4b") -> bool:
        """
        Убеждается, что LM Studio запущен и модель загружена.

        Args:
            model_name: Имя модели для загрузки

        Returns:
            True если все готово, False иначе
        """
        logger.info("Проверка LM Studio...")

        # 1. Проверяем/запускаем LM Studio
        if not self.is_lm_studio_running():
            logger.info("LM Studio не запущен, запускаем...")
            if not self.start_lm_studio(wait_time=15):
                logger.error("Не удалось запустить LM Studio")
                return False
        else:
            logger.info("LM Studio уже запущен")

        # 2. Проверяем/загружаем модель
        loaded_model = self.get_loaded_model()
        if loaded_model is None:
            logger.warning("В LM Studio не загружена ни одна модель")
            return self.load_model_in_lm_studio(model_name)

        # Сверяем загруженную модель с запрошенной (LM Studio может вернуть id вида 'author/model')
        if model_name.lower() not in loaded_model.lower():
            logger.warning(
                f"В LM Studio загружена '{loaded_model}', а не запрошенная '{model_name}' — "
                "пытаемся загрузить нужную модель"
            )
            return self.load_model_in_lm_studio(model_name)

        logger.info(f"LM Studio готов к работе с моделью: {loaded_model}")
        return True


def ensure_embedding_model_downloaded(model_name: str, cache_dir: Optional[str] = None) -> bool:
    """
    Убеждается, что embedding модель скачана локально.

    Args:
        model_name: Имя модели (например, 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        cache_dir: Директория для кэша моделей

    Returns:
        True если модель доступна, False иначе
    """
    try:
        from sentence_transformers import SentenceTransformer

        if cache_dir is None:
            # Используем папку models в корне проекта
            project_root = Path(__file__).parent.parent.parent.parent
            cache_dir = str(project_root / "models" / "embeddings")

        os.makedirs(cache_dir, exist_ok=True)

        logger.info(f"Проверка embedding модели: {model_name}")
        logger.info(f"Директория кэша: {cache_dir}")

        # Пытаемся загрузить модель (скачается автоматически если нужно)
        SentenceTransformer(model_name, cache_folder=cache_dir)
        logger.info(f"Embedding модель {model_name} готова к использованию")
        return True

    except Exception as e:
        logger.error(f"Ошибка при загрузке embedding модели: {e}")
        return False


def setup_environment(
    llm_model_name: str = "gemma-4-e4b",
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    lm_studio_url: str = "http://localhost:1234",
) -> Tuple[bool, bool]:
    """
    Настраивает окружение: запускает LM Studio и загружает модели.

    Args:
        llm_model_name: Имя LLM модели
        embedding_model_name: Имя embedding модели
        lm_studio_url: URL LM Studio

    Returns:
        Tuple[bool, bool]: (LM Studio готов, Embedding модель готова)
    """
    logger.info("=" * 60)
    logger.info("Настройка окружения для HippoRAG")
    logger.info("=" * 60)

    # 1. Настройка LM Studio
    lm_manager = LMStudioManager(base_url=lm_studio_url)
    lm_ready = lm_manager.ensure_lm_studio_ready(model_name=llm_model_name)

    # 2. Загрузка embedding модели
    embedding_ready = ensure_embedding_model_downloaded(embedding_model_name)

    logger.info("=" * 60)
    logger.info(f"LM Studio: {'✓ Готов' if lm_ready else '✗ Не готов'}")
    logger.info(f"Embedding модель: {'✓ Готова' if embedding_ready else '✗ Не готова'}")
    logger.info("=" * 60)

    return lm_ready, embedding_ready


if __name__ == "__main__":
    # Настройка логирования для тестирования
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Тестирование
    setup_environment()
