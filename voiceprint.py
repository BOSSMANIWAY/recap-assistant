"""Идентификация говорящих (диаризация) на чистом numpy.

Без torch/speechbrain: признаки — MFCC-статистика (среднее + станд. отклонение)
по озвончённым кадрам; метод — косинусное сходство с эталонами.
Профиль говорящего — до MAX_REFS эталонов (разные фразы): голос меняется
от фразы к фразе, один эталон даёт ложные «новые голоса».
"""
import logging

import numpy as np

log = logging.getLogger("assist.vp")

SR = 16000
N_FFT = 512
HOP = 256               # 16 мс между кадрами
FRAME = 400             # 25 мс
N_MELS = 40
N_MFCC = 20
MIN_EMBED = 0.35        # сек: короче — вектор не строим (мало данных)
MAX_REFS = 8            # эталонов на говорящего
# косинус: выше — тот же голос. Зазор между «свой» (~1.0) и «чужой»
# (~0.83, общий спектральный наклон) — порог в центре зазора.
# Фактические сходства пишутся в лог DEBUG — порог подстраивается по ним.
THRESHOLD = 0.90


def _mel_filters():
    """Треугольные фильтры мел-шкалы: (N_MELS, N_FFT//2+1)."""
    lo, hi = 60.0, SR / 2 - 200
    mlo = 2595 * np.log10(1 + lo / 700)
    mhi = 2595 * np.log10(1 + hi / 700)
    mels = np.linspace(mlo, mhi, N_MELS + 2)
    hz = 700 * (10 ** (mels / 2595) - 1)
    bins = np.clip(np.floor((N_FFT + 1) * hz / SR).astype(int), 0, N_FFT // 2)
    f = np.zeros((N_MELS, N_FFT // 2 + 1))
    for i in range(N_MELS):
        l, c, r = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        c = max(c, l + 1)
        r = max(r, c + 1)
        if c <= N_FFT // 2:
            f[i, l:c] = np.linspace(0, 1, c - l, endpoint=False)
        if r <= N_FFT // 2:
            f[i, c:r] = np.linspace(1, 0, r - c, endpoint=False)
    return f


_MEL = _mel_filters()

_n = np.arange(N_MFCC)[:, None]
_k = np.arange(N_MELS)[None, :]
_DCT = np.cos(np.pi * (2 * _k + 1) * _n / (2 * N_MELS)) * np.sqrt(2.0 / N_MELS)
_DCT[0] /= np.sqrt(2.0)

_WIN = np.hamming(FRAME)


def embed(audio: np.ndarray) -> np.ndarray | None:
    """Голосовой отпечаток фразы: нормализованный 38-мерный вектор или None."""
    a = np.asarray(audio, dtype=np.float64)
    if len(a) < int(MIN_EMBED * SR):
        return None
    n = 1 + (len(a) - FRAME) // HOP
    if n < 5:
        return None
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    frames = a[idx] * _WIN
    # энергетический VAD: отбрасываем 30% самых тихих кадров
    e = np.sqrt((frames**2).mean(axis=1))
    frames = frames[e > max(float(np.percentile(e, 30)), 1e-5)]
    if len(frames) < 5:
        return None
    spec = np.abs(np.fft.rfft(frames, N_FFT, axis=1)) ** 2
    mel = np.log(np.clip(spec @ _MEL.T, 1e-10, None))
    mfcc = (mel @ _DCT.T)[:, 1:]            # c0 — уровень записи, не тембр
    v = np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0)])
    nrm = float(np.linalg.norm(v))
    if nrm <= 0:
        return None
    return v / nrm


class SpeakerTracker:
    """Трекер говорящих одной сессии: identify(фраза) -> индекс говорящего."""

    def __init__(self, threshold: float = THRESHOLD):
        self.threshold = threshold
        self.profiles: list[list[np.ndarray]] = []

    def identify(self, audio: np.ndarray) -> int:
        v = embed(audio)
        if v is None:
            return 0  # слишком короткая — относим к первому, не плодим фантомов
        best, best_sim = -1, -1.0
        for i, refs in enumerate(self.profiles):
            sim = max(float(v @ r) for r in refs)
            if sim > best_sim:
                best, best_sim = i, sim
        if best >= 0 and best_sim >= self.threshold:
            if len(self.profiles[best]) < MAX_REFS:
                self.profiles[best].append(v)
            log.debug("говорящий %d (сходство=%.2f)", best + 1, best_sim)
            return best
        self.profiles.append([v])
        log.info(
            "новый говорящий #%d (лучшее сходство=%.2f, порог=%.2f)",
            len(self.profiles), best_sim, self.threshold,
        )
        return len(self.profiles) - 1

    @staticmethod
    def name(index: int) -> str:
        return f"Говорящий {index + 1}"
