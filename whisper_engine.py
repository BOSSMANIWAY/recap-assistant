"""STT-движок на faster-whisper.

Модель small скачана локально в models/faster-whisper-small (см. models/download.py),
чтобы не качать её при каждом запуске.
"""
import logging
import os
import threading

import numpy as np
from faster_whisper import WhisperModel

log = logging.getLogger("assist.stt")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "faster-whisper-small")
LANGUAGE = "ru"


class WhisperEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self.model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray) -> str:
        """Транскрибировать аудио (16 кГц) и вернуть текст.

        Вход — уже нормализованный float32 в диапазоне [-1, 1]
        (так его присылает браузер), поэтому деление на 32768 убрано.
        """
        with self._lock:
            if len(audio) < 800:
                return ""
            audio = np.asarray(audio, dtype=np.float32)
            segments, _ = self.model.transcribe(
                audio,
                language=LANGUAGE,
                beam_size=5,
                # отключаем temperature-fallback: он порождает галлюцинации
                # вроде «Игорь Негода» / «До свидания» на шумовых фрагментах
                temperature=(0.0,),
                vad_filter=True,
                vad_parameters=dict(min_speech_duration_ms=250, threshold=0.4),
                log_prob_threshold=-1.0,
                no_speech_threshold=0.5,
            )
            texts = []
            for s in segments:
                # фильтруем низкокачественные сегменты — источник галлюцинаций
                if s.no_speech_prob > 0.5:
                    log.debug("отброшен сегмент (no_speech=%.2f): %r", s.no_speech_prob, s.text[:40])
                    continue
                if s.avg_logprob < -1.0:
                    log.debug("отброшен сегмент (logprob=%.2f): %r", s.avg_logprob, s.text[:40])
                    continue
                texts.append(s.text.strip())
            return " ".join(t for t in texts if t).strip()
