"""Пакетирование аудио из worklet-чанков, VAD и накопление фраз."""
import logging
from collections import deque

import numpy as np

log = logging.getLogger("assist.vad")

SAMPLE_RATE = 16000
# Гистерезис: шум этого микрофона ≈ 0.002-0.006, speech ≈ 0.02-0.13.
# Вход в речь: rms > START. Выход (тишина): rms < STOP.
# С одним порогом шум случайно дёргал состояние и рвал фразу.
START_THRESHOLD = 0.012   # начало речи (выше шума)
STOP_THRESHOLD = 0.008    # конец речи (тоже выше шумового пола)
MIN_PHRASE = 0.3          # минимальная длина фразы, сек
SILENCE_TAIL = 1.0        # сек тишины = перерыв между репликами
MAX_PHRASE = 15.0         # сек: без паузы длиннее — отдаём кусок и пишем дальше
PRESPEECH = 0.3           # сек кольца перед речью — чтобы не съесть начало слова
AMPLIFY = 1.0


class PhraseAssembler:
    """VAD с гистерезисом и pre-speech кольцом.
    На входе float32-чанки из браузера, на выходе — готовые фразы."""

    def __init__(self):
        self.speaking = False
        self.voice: list[np.ndarray] = []
        self.voice_dur = 0.0
        self.silence = 0.0
        self.pre: deque[np.ndarray] = deque()
        self.pre_dur = 0.0
        self._rms_report = 0

    def add(self, chunk: np.ndarray) -> np.ndarray | None:
        dur = len(chunk) / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        self._rms_report += 1
        if self._rms_report >= 10:
            log.debug("rms=%.5f %s", rms, "речь" if self.speaking else "тишина")
            self._rms_report = 0

        if not self.speaking:
            if rms > START_THRESHOLD:
                # старт речи: pre-speech кольцо + текущий чанк
                log.debug("старт речи (rms=%.5f)", rms)
                self.speaking = True
                self.voice = list(self.pre) + [chunk]
                self.voice_dur = self.pre_dur + dur
                self.silence = 0.0
                self.pre.clear()
                self.pre_dur = 0.0
                return None
            # в тишине копим pre-speech кольцо
            self.pre.append(chunk)
            self.pre_dur += dur
            while self.pre_dur > PRESPEECH and len(self.pre) > 1:
                self.pre_dur -= len(self.pre.popleft()) / SAMPLE_RATE
            return None

        # уже говорим
        self.voice.append(chunk)
        self.voice_dur += dur
        if rms < STOP_THRESHOLD:
            self.silence += dur
            if self.silence > SILENCE_TAIL:
                log.debug("пауза %.1f сек — фраза закончилась", self.silence)
                return self.flush()
        else:
            self.silence = 0.0
        if self.voice_dur >= MAX_PHRASE:
            log.info("фраза > %.0f сек — завершаю принудительно", MAX_PHRASE)
            return self.flush()
        return None

    def flush(self) -> np.ndarray | None:
        self.speaking = False
        self.silence = 0.0
        if not self.voice:
            return None
        phrase = np.concatenate(self.voice)
        self.voice.clear()
        self.voice_dur = 0.0
        if len(phrase) < MIN_PHRASE * SAMPLE_RATE:
            log.debug("сброшено: короче %.1f сек", MIN_PHRASE)
            return None
        if AMPLIFY != 1.0:
            phrase = np.clip(phrase * AMPLIFY, -1.0, 1.0)
        log.debug("фраза готова: %.1f сек", len(phrase) / SAMPLE_RATE)
        return phrase.astype(np.float32)
