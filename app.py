"""
AI Audio Noise Removal & Speech Enhancer
Real-time tuning with traffic noise removal and Hindi speech enhancement.
"""

import sys
import os
import time
import traceback
import numpy as np
import soundfile as sf
import sounddevice as sd
import noisereduce as nr
from scipy.signal import butter, sosfilt, iirpeak, tf2sos, resample_poly
from pydub import AudioSegment
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog, QGroupBox, QFrame, QProgressBar, QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor, QShortcut, QKeySequence
import pyqtgraph as pg


# ─────────────────────────────────────────────
# Audio Processor — Two stage: denoise (slow, cached) + filters (instant)
# ─────────────────────────────────────────────

class AudioProcessor:
    def __init__(self):
        self.original = None      # Raw loaded audio
        self.denoised = None      # After noisereduce (cached, expensive)
        self.enhanced = None      # After all filters (cheap)
        self.sr = 16000
        self.loaded = False

        self._nr_key = -1.0       # Cached noise strength
        self._denoise_token = 0   # Invalidate in-flight workers on load

        # Params
        self.noise_strength = 0.5
        self.traffic = 0.4
        self.hiss = 0.35
        self.clarity = 0.45
        self.speech = 0.7
        self.gate = 0.05
        self.volume = 1.0

    def load(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.m4a', '.mp3', '.aac', '.ogg', '.wma'):
            seg = AudioSegment.from_file(path)
            seg = seg.set_channels(1).set_frame_rate(16000)
            # Scale based on actual sample width to keep audio in [-1, 1].
            max_val = float(1 << (8 * seg.sample_width - 1))
            s = np.array(seg.get_array_of_samples(), dtype=np.float32) / max_val
            self.sr = 16000
        else:
            s, self.sr = sf.read(path, dtype='float32')
            if s.ndim > 1:
                s = s.mean(axis=1)
        self.original = s
        self.denoised = None
        self.enhanced = s.copy()
        self._nr_key = -1.0
        self._denoise_token += 1
        self.loaded = True
        return s

    @staticmethod
    def compute_denoise(original, sr, key):
        """SLOW: run noisereduce (two passes for deep cleaning)."""
        if key > 0.01:
            # Pass 1: Stationary noise reduction (constant background noise)
            audio = nr.reduce_noise(
                y=original, sr=sr,
                prop_decrease=key,
                stationary=True,
                n_std_thresh_stationary=max(0.3, 0.5 + (1 - key) * 2.0),
            ).astype(np.float32)

            # Pass 2: Non-stationary reduction (variable noise like traffic)
            # Only apply when strength > 30% to avoid over-processing
            if key > 0.3:
                audio = nr.reduce_noise(
                    y=audio, sr=sr,
                    prop_decrease=key * 0.8,
                    stationary=False,
                ).astype(np.float32)
        else:
            audio = original.copy()
        return audio

    def apply_filters(self):
        """FAST: apply filters to denoised audio. Call on main thread."""
        base = self.denoised if self.denoised is not None else self.original
        if base is None:
            return
        audio = base.copy()

        # Traffic / horns: remove rumble and horn-heavy low-mids with wet/dry blend
        nyq = self.sr / 2
        if self.traffic > 0.001:
            amt = self.traffic
            hp = 55 + amt * 145  # 55-200 Hz
            hp_audio = self._bfilt(audio, hp, 'high')
            # Horn energy often peaks around 350-900 Hz; tame gently.
            horn_tamed = self._peak_eq(hp_audio, 420, 1.0, -5.5 * amt)
            horn_tamed = self._peak_eq(horn_tamed, 760, 1.1, -4.0 * amt)
            audio = self._mix(audio, horn_tamed, 0.22 + 0.58 * amt)

        # Hiss: de-emphasize upper bands with speech-safe blend
        if self.hiss > 0.001:
            amt = self.hiss
            lp_max = min(0.98 * nyq, 9200)
            lp_min = min(0.78 * nyq, 6800)
            lp = lp_max - amt * (lp_max - lp_min)
            lp_audio = self._bfilt(audio, lp, 'low')
            hiss_tamed = self._peak_eq(lp_audio, 6500, 1.0, -4.5 * amt)
            if nyq > 9000:
                hiss_tamed = self._peak_eq(hiss_tamed, 9000, 1.2, -3.0 * amt)
            audio = self._mix(audio, hiss_tamed, 0.18 + 0.52 * amt)

        # Hindi speech enhancement (presence + clarity, reduce mud)
        if self.speech > 0.01:
            amt = self.speech
            audio = self._peak_eq(audio, 320, 1.1, -4.0 * amt)
            audio = self._peak_eq(audio, 3000, 1.2, 6.0 * amt)
            audio = self._peak_eq(audio, 5000, 1.1, 3.0 * amt)
            audio = self._peak_eq(audio, 8000, 0.9, 2.0 * amt)

        # Compression (voice clarity)
        if self.clarity > 0.01:
            comp = self._compress(audio, self.clarity)
            audio = self._mix(audio, comp, 0.22 + 0.48 * self.clarity)

        # Speech-optimized noise gate (gentle, RMS-based)
        if self.gate > 0.01:
            g = self._gate(audio, self.gate)
            audio = self._mix(audio, g, 0.25 + 0.65 * self.gate)

        # Volume
        if abs(self.volume - 1.0) > 0.01:
            audio = np.clip(audio * self.volume, -1, 1)

        self.enhanced = np.clip(audio, -1, 1).astype(np.float32)

    def _bfilt(self, data, cutoff, btype, order=6):
        nyq = self.sr / 2
        cutoff = max(20, min(cutoff, nyq - 100))
        n = cutoff / nyq
        if n <= 0 or n >= 1:
            return data
        sos = butter(order, n, btype=btype, output='sos')
        return sosfilt(sos, data).astype(np.float32)

    def _compress(self, data, amt):
        th = 0.3 * (1 - amt * 0.5)
        r = 1 + amt * 3
        g = 1 + amt * 0.8
        out = data.copy()
        m = np.abs(out) > th
        out[m] = np.sign(out[m]) * (th + (np.abs(out[m]) - th) / r)
        out *= g
        return np.tanh(out).astype(np.float32)

    @staticmethod
    def _mix(dry, wet, amt):
        a = float(np.clip(amt, 0.0, 1.0))
        return ((1.0 - a) * dry + a * wet).astype(np.float32)

    def _peak_eq(self, data, f0, q, gain_db):
        nyq = self.sr / 2
        if f0 <= 20 or f0 >= nyq - 100 or abs(gain_db) < 0.01:
            return data
        w0 = f0 / nyq
        b, a = iirpeak(w0, q)
        sos = tf2sos(b, a)
        bp = sosfilt(sos, data).astype(np.float32)
        gain = 10 ** (gain_db / 20.0) - 1.0
        return (data + gain * bp).astype(np.float32)

    def _gate(self, data, amt):
        # RMS-based downward expander. Stronger than a binary gate and
        # safer for delicate speech tails.
        win = max(1, int(self.sr * 0.012))
        x2 = data * data
        # Fast moving average using cumulative sum
        c = np.cumsum(np.insert(x2, 0, 0.0))
        ma = (c[win:] - c[:-win]) / win
        # Pad to match length
        pad_left = win // 2
        pad_right = len(data) - len(ma) - pad_left
        if pad_right < 0:
            pad_right = 0
        ma = np.pad(ma, (pad_left, pad_right), mode='edge')
        rms = np.sqrt(ma + 1e-12)
        thr_db = -54 + amt * 20
        thr = 10 ** (thr_db / 20.0)
        floor_db = -30 - amt * 20
        floor = 10 ** (floor_db / 20.0)
        ratio = 1.5 + 5.0 * amt
        gain = np.ones_like(rms, dtype=np.float32)
        low = rms < thr
        gain[low] = np.power(np.maximum(rms[low] / (thr + 1e-12), 1e-6), ratio - 1.0)
        gain = np.clip(gain, floor, 1.0)
        return (data * gain).astype(np.float32)

    def save(self, path):
        if self.enhanced is not None:
            sf.write(path, self.enhanced, self.sr)
            return True
        return False


# ─────────────────────────────────────────────
# Denoise Worker (background thread, only for noisereduce)
# ─────────────────────────────────────────────

class DenoiseWorker(QThread):
    done = pyqtSignal(int, object, float)
    error = pyqtSignal(str)

    def __init__(self, proc, token, original, sr, key):
        super().__init__()
        self.proc = proc
        self.token = token
        self.original = original
        self.sr = sr
        self.key = key

    def run(self):
        try:
            audio = self.proc.compute_denoise(self.original, self.sr, self.key)
            self.done.emit(self.token, audio, self.key)
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────
# AI Enhance Worker (DeepFilterNet)
# ─────────────────────────────────────────────

class AIEnhanceWorker(QThread):
    done = pyqtSignal(object, str)
    error = pyqtSignal(str)

    def __init__(self, original, sr, out_path, dialogue_recovery=0.6):
        super().__init__()
        self.original = original
        self.sr = sr
        self.out_path = out_path
        self.dialogue_recovery = float(np.clip(dialogue_recovery, 0.0, 1.0))

    @staticmethod
    def _resample(audio, orig_sr, target_sr):
        if orig_sr == target_sr:
            return audio
        from math import gcd
        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        return resample_poly(audio, up, down).astype(np.float32)

    @staticmethod
    def _recover_dialogue(ai_audio, original_audio, recovery_amt):
        """Recover delicate dialogue after strong denoising."""
        if ai_audio is None or len(ai_audio) == 0:
            return ai_audio
        out = np.asarray(ai_audio, dtype=np.float32)
        orig = np.asarray(original_audio, dtype=np.float32)
        amt = float(np.clip(recovery_amt, 0.0, 1.0))

        # 1) Blend back a little original to recover low-energy consonants/breath.
        n = min(len(out), len(orig))
        if n > 0:
            out_n = out[:n]
            orig_n = orig[:n]
            blend = 0.05 + 0.20 * amt
            out[:n] = (1.0 - blend) * out_n + blend * orig_n

        # 2) Energy-based make-up gain with safe clamp.
        p95 = float(np.percentile(np.abs(out), 95))
        if p95 > 1e-6:
            target_p95 = 0.18 + 0.10 * amt
            gain = target_p95 / p95
            gain = float(np.clip(gain, 0.9, 2.4))
            out *= gain

        # 3) Gentle soft limiter + final peak protection.
        drive = 1.05 + 0.25 * amt
        out = np.tanh(drive * out) / np.tanh(drive)
        peak = float(np.max(np.abs(out))) if len(out) else 1.0
        if peak > 0.96:
            out = out * (0.96 / peak)
        return np.clip(out, -1, 1).astype(np.float32)

    def run(self):
        try:
            from df import enhance, init_df
            from df.config import config as df_config
            import torch

            # Force DeepFilterNet internal device routing to CPU so that
            # model weights, features and tensors all share the same device.
            os.environ["DEVICE"] = "cpu"
            model, df_state, _ = init_df()
            df_config.set("DEVICE", "cpu", str, section="train")
            audio = self.original.astype(np.float32)
            target_sr = df_state.sr()
            if self.sr != target_sr:
                audio_rs = self._resample(audio, self.sr, target_sr)
            else:
                audio_rs = audio

            def _extract_audio_to_numpy(x):
                # Handle various return signatures from df.enhance()
                if torch.is_tensor(x):
                    return x.detach().cpu().numpy()
                if isinstance(x, (tuple, list)):
                    if not x:
                        raise ValueError("AI enhance returned empty sequence")
                    return _extract_audio_to_numpy(x[0])
                if isinstance(x, dict):
                    for key in ("enhanced", "audio", "y", "out"):
                        if key in x:
                            return _extract_audio_to_numpy(x[key])
                    # fallback: first value
                    for v in x.values():
                        return _extract_audio_to_numpy(v)
                    raise ValueError("AI enhance returned empty dict")
                if hasattr(x, "detach") and hasattr(x, "cpu"):
                    return x.detach().cpu().numpy()
                return np.asarray(x)

            def _run_once():
                audio_t = torch.from_numpy(audio_rs).to(torch.device("cpu"))
                if audio_t.ndim == 1:
                    audio_t = audio_t.unsqueeze(0)  # [channels, samples]
                out = enhance(model, df_state, audio_t)
                out_np = _extract_audio_to_numpy(out)
                out_np = np.asarray(out_np, dtype=np.float32)
                if out_np.ndim == 2:
                    # Convert [channels, samples] to mono safely.
                    out_np = out_np.mean(axis=0)
                elif out_np.ndim > 2:
                    out_np = np.asarray(out_np).reshape(-1)
                return np.asarray(out_np, dtype=np.float32).reshape(-1)

            enhanced = _run_once()
            enhanced = np.asarray(enhanced, dtype=np.float32).reshape(-1)

            if self.sr != target_sr:
                enhanced = self._resample(enhanced, target_sr, self.sr)

            enhanced = self._recover_dialogue(enhanced, self.original, self.dialogue_recovery)
            enhanced = np.clip(enhanced, -1, 1).astype(np.float32)
            sf.write(self.out_path, enhanced, self.sr)
            self.done.emit(enhanced, self.out_path)
        except ModuleNotFoundError as e:
            if getattr(e, "name", "") == "df":
                self.error.emit(
                    "DeepFilterNet is not installed in this Python environment.\n"
                    f"Interpreter: {sys.executable}\n"
                    "Install with: python -m pip install deepfilternet==0.5.6"
                )
            else:
                self.error.emit(f"{e}\n{traceback.format_exc()}")
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")

# ─────────────────────────────────────────────
# Simple Slider
# ─────────────────────────────────────────────

_CSS = """
QSlider::groove:horizontal {
    border:none; height:8px;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1a2a3a, stop:1 #2a3a4a);
    border-radius:4px;
}
QSlider::handle:horizontal {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #5ecfff, stop:1 #3aa0d8);
    border:2px solid #2a4a6a; width:18px; height:18px; margin:-6px 0; border-radius:10px;
}
QSlider::handle:horizontal:hover {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #7edfff, stop:1 #5abfe8);
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2a6090, stop:1 #4a90c0);
    border-radius:4px;
}"""


class Knob(QWidget):
    changed = pyqtSignal(float)

    def __init__(self, label, icon, lo, hi, default=50):
        super().__init__()
        self.setMinimumHeight(88)
        ly = QVBoxLayout(self)
        ly.setContentsMargins(0, 5, 0, 5)
        ly.setSpacing(2)
        h = QHBoxLayout()
        t = QLabel(f"{icon}  {label}")
        t.setFont(QFont("Segoe UI", 12, QFont.Weight.DemiBold))
        t.setStyleSheet("color:#e0e6ed;")
        self.vl = QLabel(f"{default}%")
        self.vl.setFont(QFont("Segoe UI Semibold", 12))
        self.vl.setStyleSheet("color:#7ecfff;")
        self.vl.setAlignment(Qt.AlignmentFlag.AlignRight)
        h.addWidget(t); h.addWidget(self.vl)
        ly.addLayout(h)
        self.s = QSlider(Qt.Orientation.Horizontal)
        self.s.setRange(0, 100); self.s.setValue(default)
        self.s.setStyleSheet(_CSS)
        self.s.valueChanged.connect(self._c)
        ly.addWidget(self.s)
        r = QHBoxLayout()
        a = QLabel(lo); a.setFont(QFont("Segoe UI", 9)); a.setStyleSheet("color:#6e7681;")
        b = QLabel(hi); b.setFont(QFont("Segoe UI", 9)); b.setStyleSheet("color:#6e7681;")
        b.setAlignment(Qt.AlignmentFlag.AlignRight)
        r.addWidget(a); r.addWidget(b)
        ly.addLayout(r)

    def _c(self, v):
        self.vl.setText(f"{v}%")
        self.changed.emit(v / 100.0)


class VolKnob(QWidget):
    changed = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(88)
        ly = QVBoxLayout(self)
        ly.setContentsMargins(0, 5, 0, 5)
        ly.setSpacing(2)
        h = QHBoxLayout()
        t = QLabel("🔊  Volume")
        t.setFont(QFont("Segoe UI", 12, QFont.Weight.DemiBold))
        t.setStyleSheet("color:#e0e6ed;")
        self.vl = QLabel("1.0x")
        self.vl.setFont(QFont("Segoe UI Semibold", 12))
        self.vl.setStyleSheet("color:#7ecfff;")
        self.vl.setAlignment(Qt.AlignmentFlag.AlignRight)
        h.addWidget(t); h.addWidget(self.vl)
        ly.addLayout(h)
        self.s = QSlider(Qt.Orientation.Horizontal)
        self.s.setRange(0, 100); self.s.setValue(20)
        self.s.setStyleSheet(_CSS)
        self.s.valueChanged.connect(self._c)
        ly.addWidget(self.s)
        r = QHBoxLayout()
        a = QLabel("Quiet"); a.setFont(QFont("Segoe UI", 9)); a.setStyleSheet("color:#6e7681;")
        b = QLabel("Loud"); b.setFont(QFont("Segoe UI", 9)); b.setStyleSheet("color:#6e7681;")
        b.setAlignment(Qt.AlignmentFlag.AlignRight)
        r.addWidget(a); r.addWidget(b)
        ly.addLayout(r)

    def _c(self, v):
        vol = 0.5 + v / 100.0 * 2.5
        self.vl.setText(f"{vol:.1f}x")
        self.changed.emit(vol)


# ─────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.proc = AudioProcessor()
        self._worker = None
        self._ai_worker = None
        self._duration = 0.0
        self._sel_start = 0.0
        self._sel_end = 0.0
        self._play_vol_orig = 1.0
        self._play_vol_enh = 1.0
        self._is_playing = False
        self._play_started_at = 0.0
        self._play_offset_sec = 0.0
        self._play_len_sec = 0.0
        self._loaded_path = None
        self._dialogue_recovery = 0.7

        # Debounce only for noise reduction slider
        self._nr_debounce = QTimer()
        self._nr_debounce.setSingleShot(True)
        self._nr_debounce.setInterval(800)
        self._nr_debounce.timeout.connect(self._run_denoise)

        # Debounce fast filters to keep UI responsive
        self._filter_debounce = QTimer()
        self._filter_debounce.setSingleShot(True)
        self._filter_debounce.setInterval(90)
        self._filter_debounce.timeout.connect(self._apply_filters_now)

        # Playback cursor timer
        self._play_timer = QTimer()
        self._play_timer.setInterval(40)
        self._play_timer.timeout.connect(self._update_play_cursor)

        self.setWindowTitle("🎙️ AI Audio Enhancer")
        self.setMinimumSize(1320, 900)
        self._build_ui()
        self._apply_theme()
        self._bind_shortcuts()

        # Auto-load
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Voice 260228_185452.m4a")
        if os.path.exists(f):
            QTimer.singleShot(300, lambda: self._load(f))

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background:#0d1117; }
            QWidget { background:#0d1117; color:#c9d1d9; font-family:'Segoe UI'; }
            QGroupBox {
                background:#161b22; border:1px solid #21262d; border-radius:12px;
                padding:20px; padding-top:34px; margin-top:10px;
                font-size:14px; font-weight:bold; color:#58a6ff;
            }
            QGroupBox::title {
                subcontrol-origin:margin; subcontrol-position:top left;
                padding:4px 14px; background:#1a2332; border-radius:6px; color:#58a6ff;
            }
            QPushButton {
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #1f3044, stop:1 #162436);
                color:#7ecfff; border:1px solid #2a4a6a; border-radius:8px;
                padding:12px 22px; font-size:14px; font-weight:bold; min-height:22px;
            }
            QPushButton:hover {
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #2a4a64, stop:1 #1f3a52);
                border:1px solid #4a7aaa; color:#a0dfff;
            }
            QPushButton:pressed { background:#0d2030; }
            QPushButton:disabled { background:#161b22; color:#484f58; border:1px solid #21262d; }
            QLabel { background:transparent; }
            QProgressBar {
                border:1px solid #2a4a6a; border-radius:7px; background:#111822; height:12px;
                text-align:center; color:#9ccfff;
            }
            QProgressBar::chunk {
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2b6cb0, stop:1 #4ac0ff);
                border-radius:6px;
            }
        """)

    def _build_ui(self):
        c = QWidget(); self.setCentralWidget(c)
        root = QVBoxLayout(c)
        root.setContentsMargins(24, 18, 24, 18); root.setSpacing(12)

        # ── Top bar ──
        top = QHBoxLayout()
        self.b_load = QPushButton("📂  Load Audio"); self.b_load.setFixedWidth(160)
        self.b_load.clicked.connect(self._on_load_click)
        top.addWidget(self.b_load)
        self.file_lbl = QLabel("No file loaded")
        self.file_lbl.setFont(QFont("Segoe UI", 12))
        self.file_lbl.setStyleSheet("color:#6e7681; padding-left:12px;")
        top.addWidget(self.file_lbl, 1)
        self.stat = QLabel("")
        self.stat.setFont(QFont("Segoe UI", 11))
        top.addWidget(self.stat)
        root.addLayout(top)

        self.busy = QProgressBar()
        self.busy.setRange(0, 0)  # indeterminate
        self.busy.setVisible(False)
        self.busy.setFixedHeight(12)
        root.addWidget(self.busy)

        # ── Waveforms ──
        wg = QGroupBox("📊  Audio Waveform")
        wl = QVBoxLayout(wg)
        pg.setConfigOptions(antialias=False, background='#0d1117', foreground='#c9d1d9')

        wl.addWidget(self._lbl("🔴  Original Audio", "#f08060"))
        self.pw1 = pg.PlotWidget(); self._init_pw(self.pw1); wl.addWidget(self.pw1)
        self.cursor1 = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#f6c177", width=1.4))
        self.cursor1.setVisible(False)
        self.cursor1.setZValue(20)
        self.pw1.addItem(self.cursor1)

        wl.addWidget(self._lbl("🔵  Enhanced Audio", "#4ac0ff"))
        self.pw2 = pg.PlotWidget(); self._init_pw(self.pw2); wl.addWidget(self.pw2)
        self.cursor2 = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#f6c177", width=1.4))
        self.cursor2.setVisible(False)
        self.cursor2.setZValue(20)
        self.pw2.addItem(self.cursor2)
        self.region = pg.LinearRegionItem(
            values=[0, 1],
            movable=True,
            brush=(74, 192, 255, 40),
            pen=pg.mkPen('#4ac0ff', width=1.2),
        )
        self.region.setZValue(10)
        self.region.sigRegionChanged.connect(self._on_region_changed)
        self.region.setVisible(False)
        self.pw2.addItem(self.region)
        root.addWidget(wg, 2)

        # ── Bottom: sliders + buttons ──
        bot = QHBoxLayout(); bot.setSpacing(16)

        sg = QGroupBox("🎛️  Tuning Controls")
        sl = QVBoxLayout(sg); sl.setSpacing(6)

        quick = QHBoxLayout()
        quick.setSpacing(10)
        self.b_preset = QPushButton("Traffic Horns Preset")
        self.b_preset.clicked.connect(self._apply_traffic_preset)
        self.b_preset.setStyleSheet(self._cbss("#d29922", "#8b5a15"))
        quick.addWidget(self.b_preset, 1)

        self.b_ai = QPushButton("AI Enhance (DeepFilterNet)")
        self.b_ai.clicked.connect(self._run_ai_enhance)
        self.b_ai.setStyleSheet(self._cbss("#7ecfff", "#2a4a6a"))
        quick.addWidget(self.b_ai, 1)
        sl.addLayout(quick)

        self.k_noise = Knob("Noise Removal", "🤖", "None", "Maximum", 50)
        self.k_noise.changed.connect(self._on_noise)
        sl.addWidget(self.k_noise)

        self.k_traffic = Knob("Traffic / Horns Removal", "🚗", "None", "Maximum", 40)
        self.k_traffic.changed.connect(self._on_filter)
        sl.addWidget(self.k_traffic)

        self.k_hiss = Knob("Hiss / Background Removal", "🌬️", "None", "Maximum", 35)
        self.k_hiss.changed.connect(self._on_filter)
        sl.addWidget(self.k_hiss)

        self.k_clarity = Knob("Voice Clarity", "🎙️", "Natural", "Extra Clear", 45)
        self.k_clarity.changed.connect(self._on_filter)
        sl.addWidget(self.k_clarity)

        self.k_speech = Knob("Hindi Speech Enhancer", "🗣️", "Natural", "Focused", 70)
        self.k_speech.changed.connect(self._on_filter)
        sl.addWidget(self.k_speech)

        self.k_dialog = Knob("Dialogue Recovery", "💬", "Light", "Strong", 70)
        self.k_dialog.changed.connect(self._on_dialogue_recovery)
        sl.addWidget(self.k_dialog)

        self.k_gate = Knob("Noise Gate", "🚪", "Open", "Tight", 5)
        self.k_gate.changed.connect(self._on_filter)
        sl.addWidget(self.k_gate)

        self.k_vol = VolKnob()
        self.k_vol.changed.connect(self._on_filter)
        sl.addWidget(self.k_vol)

        sg_scroll = QScrollArea()
        sg_scroll.setWidgetResizable(True)
        sg_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sg_scroll.setWidget(sg)
        bot.addWidget(sg_scroll, 5)

        # Playback panel
        pg2 = QGroupBox("▶️  Playback")
        pl = QVBoxLayout(pg2); pl.setSpacing(10)
        pg2.setMinimumWidth(440)

        self.b1 = QPushButton("▶  Play Original"); self.b1.setEnabled(False)
        self.b1.clicked.connect(self._play_orig)
        self.b1.setStyleSheet(self._cbss("#f08060", "#c05030"))
        pl.addWidget(self.b1)

        self.b2 = QPushButton("▶  Play Enhanced"); self.b2.setEnabled(False)
        self.b2.clicked.connect(self._play_enh)
        self.b2.setStyleSheet(self._cbss("#4ac0ff", "#2a80c0"))
        pl.addWidget(self.b2)

        self.b3 = QPushButton("⏹  Stop"); self.b3.setEnabled(False)
        self.b3.clicked.connect(self._stop)
        pl.addWidget(self.b3)

        pv1 = self._make_playback_slider("Original Volume", "#f08060", 100)
        self.pv_orig = pv1["slider"]; self.pv_orig_lbl = pv1["label"]
        self.pv_orig.valueChanged.connect(self._on_play_vol_orig)
        pl.addWidget(pv1["widget"])

        pv2 = self._make_playback_slider("Enhanced Volume", "#4ac0ff", 100)
        self.pv_enh = pv2["slider"]; self.pv_enh_lbl = pv2["label"]
        self.pv_enh.valueChanged.connect(self._on_play_vol_enh)
        pl.addWidget(pv2["widget"])

        pl.addStretch()
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background:#21262d; max-height:1px;")
        pl.addWidget(sep)

        self.b4 = QPushButton("💾  Export Clean Audio"); self.b4.setEnabled(False)
        self.b4.clicked.connect(self._export)
        self.b4.setStyleSheet(self._cbss("#3fb950", "#2a8040"))
        pl.addWidget(self.b4)

        self.b5 = QPushButton("▶  Play Selection"); self.b5.setEnabled(False)
        self.b5.clicked.connect(self._play_selection)
        self.b5.setStyleSheet(self._cbss("#4ac0ff", "#2a80c0"))
        pl.addWidget(self.b5)

        self.b6 = QPushButton("💾  Export Selection"); self.b6.setEnabled(False)
        self.b6.clicked.connect(self._export_selection)
        self.b6.setStyleSheet(self._cbss("#3fb950", "#2a8040"))
        pl.addWidget(self.b6)

        dl = QLabel(""); dl.setFont(QFont("Segoe UI", 11)); dl.setStyleSheet("color:#6e7681;")
        dl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.d_lbl = dl; pl.addWidget(dl)
        self.play_pos_lbl = QLabel("0:00 / 0:00")
        self.play_pos_lbl.setFont(QFont("Segoe UI", 11))
        self.play_pos_lbl.setStyleSheet("color:#9ccfff;")
        self.play_pos_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pl.addWidget(self.play_pos_lbl)
        self.sel_lbl = QLabel("Selection: --"); self.sel_lbl.setFont(QFont("Segoe UI", 10))
        self.sel_lbl.setStyleSheet("color:#6e7681;")
        self.sel_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pl.addWidget(self.sel_lbl)
        self.hint_lbl = QLabel("Shortcuts: Space Play/Stop • Ctrl+O Load • Ctrl+S Export Selection")
        self.hint_lbl.setFont(QFont("Segoe UI", 9))
        self.hint_lbl.setStyleSheet("color:#6e7681;")
        self.hint_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pl.addWidget(self.hint_lbl)
        pl.addStretch()
        pg2_scroll = QScrollArea()
        pg2_scroll.setWidgetResizable(True)
        pg2_scroll.setFrameShape(QFrame.Shape.NoFrame)
        pg2_scroll.setWidget(pg2)
        bot.addWidget(pg2_scroll, 3)
        root.addLayout(bot, 3)

        self.b_load.setToolTip("Load an audio file (Ctrl+O)")
        self.b_ai.setToolTip("Run DeepFilterNet AI enhancement on loaded file")
        self.b_preset.setToolTip("Apply tuned settings for traffic horns and delicate speech")
        self.b1.setToolTip("Play original track")
        self.b2.setToolTip("Play enhanced track (Space)")
        self.b4.setToolTip("Export full enhanced audio")
        self.b5.setToolTip("Play selected region only")
        self.b6.setToolTip("Export selected region (Ctrl+S)")

    def _lbl(self, text, color):
        l = QLabel(text)
        l.setFont(QFont("Segoe UI Semibold", 10))
        l.setStyleSheet(f"color:{color};")
        return l

    def _init_pw(self, pw):
        pw.setBackground('#0d1117'); pw.setFixedHeight(108)
        pw.showGrid(x=False, y=True, alpha=0.1)
        pw.setMouseEnabled(x=False, y=False); pw.hideButtons()
        pw.getAxis('left').setWidth(40)
        pw.getAxis('bottom').setStyle(showValues=False)

    def _cbss(self, c, d):
        return f"""
        QPushButton {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {d}, stop:1 #161b22);
            color:{c}; border:1px solid {d}; border-radius:8px; padding:12px 20px; font-size:14px; font-weight:bold; min-height:24px; }}
        QPushButton:hover {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {c}, stop:1 {d}); color:#fff; }}
        QPushButton:disabled {{ background:#161b22; color:#484f58; border:1px solid #21262d; }}
        """

    @staticmethod
    def _fmt(s): return f"{int(s // 60)}:{int(s % 60):02d}"

    @staticmethod
    def _fmt_pos(cur, total):
        cur = max(0.0, float(cur))
        total = max(0.0, float(total))
        return f"{App._fmt(cur)} / {App._fmt(total)}"

    def _bind_shortcuts(self):
        QShortcut(QKeySequence("Space"), self, activated=self._toggle_play_pause)
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self._on_load_click)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._export_selection)

    def _toggle_play_pause(self):
        if not self.proc.loaded:
            return
        if self._is_playing:
            self._stop()
        else:
            self._play_enh()

    def _set_busy(self, active, msg=None):
        self.busy.setVisible(active)
        self.b_load.setEnabled(not active)
        self.b_ai.setEnabled(not active)
        self.b_preset.setEnabled(not active)
        if msg is not None:
            self._msg(msg, "#d29922" if active else "#3fb950")

    def _make_playback_slider(self, title, color, default):
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(0, 0, 0, 0)
        ly.setSpacing(4)
        h = QHBoxLayout()
        t = QLabel(title); t.setFont(QFont("Segoe UI", 10)); t.setStyleSheet("color:#c9d1d9;")
        v = QLabel(f"{default}%"); v.setFont(QFont("Segoe UI", 10))
        v.setStyleSheet(f"color:{color};")
        v.setAlignment(Qt.AlignmentFlag.AlignRight)
        h.addWidget(t); h.addWidget(v)
        ly.addLayout(h)
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(0, 150); s.setValue(default)
        s.setStyleSheet(_CSS)
        ly.addWidget(s)
        return {"widget": w, "slider": s, "label": v}

    # ── Load ──

    def _on_load_click(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Load Audio", "",
            "Audio (*.m4a *.mp3 *.wav *.flac *.aac *.ogg);;All (*)")
        if fp:
            self._load(fp)

    def _load(self, fp):
        self._stop()
        self._msg("Loading...", "#d29922")
        QApplication.processEvents()

        try:
            audio = self.proc.load(fp)
        except Exception as e:
            self._msg(f"❌ Load failed: {e}", "#f85149")
            return

        self._loaded_path = fp
        self._duration = len(audio) / self.proc.sr
        self.file_lbl.setText(f"📄 {os.path.basename(fp)}  •  {self._duration:.1f}s")
        self.file_lbl.setStyleSheet("color:#c9d1d9; padding-left:12px;")
        self.d_lbl.setText(self._fmt(self._duration))
        self.play_pos_lbl.setText(self._fmt_pos(0.0, self._duration))

        self._draw_wave(self.pw1, audio, '#f08060')
        self._init_selection()

        self.b1.setEnabled(True); self.b2.setEnabled(True)
        self.b3.setEnabled(True); self.b4.setEnabled(True)
        self.b5.setEnabled(True); self.b6.setEnabled(True)

        # Run initial denoise (in background)
        self._run_denoise()

    # ── Waveform ──

    def _draw_wave(self, pw, audio, color):
        pw.clear()
        n = 5000
        if len(audio) > n:
            d = audio[::max(1, len(audio)//n)]
        else:
            d = audio
        t = np.linspace(0, len(audio) / self.proc.sr, len(d))
        pw.plot(t, d, pen=pg.mkPen(color=color, width=1.2))
        pw.setYRange(-1, 1)
        if pw is self.pw1:
            pw.addItem(self.cursor1)
        if pw is self.pw2:
            pw.addItem(self.region)
            pw.addItem(self.cursor2)

    # ── Processing ──

    def _on_noise(self, v):
        """Noise slider changed → debounce → re-run noisereduce (slow)."""
        self.proc.noise_strength = v
        self._nr_debounce.start()

    def _on_filter(self, _=None):
        """Any filter slider changed → debounce → apply filters (fast)."""
        self.proc.traffic = self.k_traffic.s.value() / 100.0
        self.proc.hiss = self.k_hiss.s.value() / 100.0
        self.proc.clarity = self.k_clarity.s.value() / 100.0
        self.proc.speech = self.k_speech.s.value() / 100.0
        self.proc.gate = self.k_gate.s.value() / 100.0
        self.proc.volume = 0.5 + self.k_vol.s.value() / 100.0 * 2.5

        if self.proc.loaded:
            self._filter_debounce.start()

    def _on_dialogue_recovery(self, v):
        self._dialogue_recovery = v
        if self.proc.loaded:
            self._msg("✅ Dialogue recovery updated (applies on next AI Enhance)", "#3fb950")

    def _apply_filters_now(self):
        if not self.proc.loaded:
            return
        self.proc.apply_filters()
        self._draw_wave(self.pw2, self.proc.enhanced, '#4ac0ff')
        self._msg("✅ Ready", "#3fb950")

    def _run_denoise(self):
        """Run noisereduce in a background thread."""
        if not self.proc.loaded:
            return
        if self._worker is not None and self._worker.isRunning():
            # Already running, will re-queue on completion
            self._nr_debounce.start()
            return

        key = round(self.proc.noise_strength, 2)
        if self.proc.denoised is not None and self.proc._nr_key == key:
            self.proc.apply_filters()
            self._draw_wave(self.pw2, self.proc.enhanced, '#4ac0ff')
            self._msg("✅ Ready", "#3fb950")
            return

        self._msg("⏳ Removing noise... (this takes a few seconds)", "#d29922")
        self._set_busy(True)
        QApplication.processEvents()

        token = self.proc._denoise_token
        original = self.proc.original.copy()
        sr = self.proc.sr
        self._worker = DenoiseWorker(self.proc, token, original, sr, key)
        self._worker.done.connect(self._on_denoise_done)
        self._worker.error.connect(self._on_denoise_err)
        self._worker.start()

    def _on_denoise_done(self, token, audio, key):
        """Denoise finished → apply filters (fast) → update waveform."""
        self._set_busy(False)
        if token != self.proc._denoise_token:
            return
        self.proc.denoised = audio
        self.proc._nr_key = key
        self.proc.apply_filters()
        self._draw_wave(self.pw2, self.proc.enhanced, '#4ac0ff')
        self._msg("✅ Ready", "#3fb950")

    def _on_denoise_err(self, msg):
        self._set_busy(False)
        self._msg(f"❌ Error: {msg}", "#f85149")

    def _msg(self, text, color):
        self.stat.setText(text)
        self.stat.setStyleSheet(f"color:{color};")

    # ── Playback (dead simple: stop → wait → play copy) ──

    def _play_orig(self):
        self._do_play('original')

    def _play_enh(self):
        self._do_play('enhanced')

    def _do_play(self, source):
        """The ONE function that plays audio. Hard-stops everything first."""
        # 1) Hard kill ALL audio
        try:
            sd.stop()
        except Exception:
            pass
        # Wait for audio device to fully release
        time.sleep(0.15)

        # 2) Get the data
        if source == 'original':
            data = self.proc.original
            vol = self._play_vol_orig
        else:
            data = self.proc.enhanced
            vol = self._play_vol_enh
        if data is None:
            self._msg("❌ No audio data", "#f85149")
            return

        # 3) COPY and play (copy prevents corruption from processing thread)
        audio_copy = np.clip(data * vol, -1, 1).copy()
        sd.play(audio_copy, self.proc.sr)
        self._is_playing = True
        self._start_playback_cursor(0.0, len(audio_copy) / self.proc.sr)

        name = "Original" if source == 'original' else "Enhanced"
        color = "#f08060" if source == 'original' else "#4ac0ff"
        self._msg(f"🔊 Playing {name}", color)

    def _play_selection(self):
        if self.proc.enhanced is None or self._duration <= 0:
            self._msg("❌ No enhanced audio", "#f85149")
            return
        start = int(self._sel_start * self.proc.sr)
        end = int(self._sel_end * self.proc.sr)
        if end <= start:
            self._msg("❌ Selection is empty", "#f85149")
            return
        try:
            sd.stop()
        except Exception:
            pass
        time.sleep(0.15)
        audio_copy = np.clip(self.proc.enhanced[start:end] * self._play_vol_enh, -1, 1).copy()
        sd.play(audio_copy, self.proc.sr)
        self._is_playing = True
        self._start_playback_cursor(self._sel_start, len(audio_copy) / self.proc.sr)
        self._msg("🔊 Playing Selection", "#4ac0ff")

    def _on_play_vol_orig(self, v):
        self._play_vol_orig = v / 100.0
        self.pv_orig_lbl.setText(f"{v}%")

    def _on_play_vol_enh(self, v):
        self._play_vol_enh = v / 100.0
        self.pv_enh_lbl.setText(f"{v}%")

    def _stop(self):
        """Stop all playback."""
        try:
            sd.stop()
        except Exception:
            pass
        self._is_playing = False
        self._stop_playback_cursor()
        if self.proc.loaded:
            self._msg("✅ Ready", "#3fb950")

    def _start_playback_cursor(self, offset_sec, play_len_sec):
        self._play_started_at = time.monotonic()
        self._play_offset_sec = max(0.0, float(offset_sec))
        self._play_len_sec = max(0.0, float(play_len_sec))
        self.cursor1.setVisible(True)
        self.cursor2.setVisible(True)
        self._set_cursor_time(self._play_offset_sec)
        self._play_timer.start()

    def _set_cursor_time(self, t_sec):
        t = max(0.0, float(t_sec))
        self.cursor1.setValue(t)
        self.cursor2.setValue(t)
        self.play_pos_lbl.setText(self._fmt_pos(t, self._duration))

    def _update_play_cursor(self):
        if not self._is_playing:
            self._stop_playback_cursor()
            return
        elapsed = time.monotonic() - self._play_started_at
        if elapsed >= self._play_len_sec:
            self._is_playing = False
            self._stop_playback_cursor()
            if self.proc.loaded:
                self._msg("✅ Ready", "#3fb950")
            return
        self._set_cursor_time(self._play_offset_sec + elapsed)

    def _stop_playback_cursor(self):
        self._play_timer.stop()
        self.cursor1.setVisible(False)
        self.cursor2.setVisible(False)
        if self.proc.loaded:
            self.play_pos_lbl.setText(self._fmt_pos(0.0, self._duration))

    # ── Export ──

    def _export(self):
        if self.proc.enhanced is None:
            self._msg("❌ No enhanced audio to save", "#f85149")
            return
        fp, _ = QFileDialog.getSaveFileName(self, "Export Clean Audio",
            "enhanced_audio.wav", "WAV (*.wav);;All (*)")
        if fp and self.proc.save(fp):
            self._msg(f"💾 Saved: {os.path.basename(fp)}", "#3fb950")

    def _init_selection(self):
        self._sel_start = 0.0
        self._sel_end = min(5.0, self._duration) if self._duration > 0 else 0.0
        if self._duration > 0:
            self.region.setBounds([0.0, self._duration])
            self.region.setRegion([self._sel_start, self._sel_end])
            self.region.setVisible(True)
        else:
            self.region.setVisible(False)
        self._update_selection_label()

    def _on_region_changed(self):
        r = self.region.getRegion()
        self._sel_start = max(0.0, min(r[0], self._duration))
        self._sel_end = max(0.0, min(r[1], self._duration))
        if self._sel_end < self._sel_start:
            self._sel_start, self._sel_end = self._sel_end, self._sel_start
        self._update_selection_label()

    def _update_selection_label(self):
        if self._duration <= 0:
            self.sel_lbl.setText("Selection: --")
            return
        dur = max(0.0, self._sel_end - self._sel_start)
        self.sel_lbl.setText(
            f"Selection: {self._sel_start:.2f}s → {self._sel_end:.2f}s  ({dur:.2f}s)"
        )

    def _export_selection(self):
        if self.proc.enhanced is None or self._duration <= 0:
            self._msg("❌ No enhanced audio to save", "#f85149")
            return
        start = int(self._sel_start * self.proc.sr)
        end = int(self._sel_end * self.proc.sr)
        if end <= start:
            self._msg("❌ Selection is empty", "#f85149")
            return
        clip = self.proc.enhanced[start:end]
        fp, _ = QFileDialog.getSaveFileName(self, "Export Selection",
            "enhanced_selection.wav", "WAV (*.wav);;All (*)")
        if fp:
            sf.write(fp, clip, self.proc.sr)
            self._msg(f"💾 Saved: {os.path.basename(fp)}", "#3fb950")

    def _run_ai_enhance(self):
        if not self.proc.loaded or self.proc.original is None:
            self._msg("❌ No audio loaded", "#f85149")
            return
        if self._ai_worker is not None and self._ai_worker.isRunning():
            return
        if not self._loaded_path:
            self._msg("❌ Missing file path", "#f85149")
            return

        base = os.path.splitext(os.path.basename(self._loaded_path))[0]
        out_path = os.path.join(os.path.dirname(self._loaded_path), f"{base}_ai_enhanced.wav")

        self._msg("🤖 AI enhancing... (this may take a bit)", "#d29922")
        self._set_busy(True)
        QApplication.processEvents()

        self._ai_worker = AIEnhanceWorker(
            self.proc.original.copy(), self.proc.sr, out_path, self._dialogue_recovery
        )
        self._ai_worker.done.connect(self._on_ai_done)
        self._ai_worker.error.connect(self._on_ai_error)
        self._ai_worker.start()

    def _on_ai_done(self, enhanced, out_path):
        self._set_busy(False)
        self.proc.denoised = enhanced
        self._apply_filters_now()
        self._msg(f"✅ AI enhanced saved: {os.path.basename(out_path)}", "#3fb950")

    def _on_ai_error(self, msg):
        self._set_busy(False)
        short = msg.splitlines()[0] if msg else "Unknown error"
        print(msg)
        self._msg(f"❌ AI enhance failed: {short}", "#f85149")

    def _apply_traffic_preset(self):
        # Female speech + traffic horns preset (conservative, clarity-preserving)
        self.k_noise.s.setValue(50)
        self.k_traffic.s.setValue(40)
        self.k_hiss.s.setValue(35)
        self.k_clarity.s.setValue(45)
        self.k_speech.s.setValue(70)
        self.k_dialog.s.setValue(70)
        self.k_gate.s.setValue(5)
        self.k_vol.s.setValue(20)
        self._on_filter()
        self._nr_debounce.start()
        self._msg("⚡ Traffic Horns preset applied", "#d29922")

    def closeEvent(self, event):
        try:
            sd.stop()
        except Exception:
            pass
        self._stop_playback_cursor()
        super().closeEvent(event)


# ─────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(13, 17, 23))
    p.setColor(QPalette.ColorRole.WindowText, QColor(201, 209, 217))
    p.setColor(QPalette.ColorRole.Base, QColor(22, 27, 34))
    p.setColor(QPalette.ColorRole.Text, QColor(201, 209, 217))
    p.setColor(QPalette.ColorRole.Button, QColor(22, 27, 34))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(201, 209, 217))
    p.setColor(QPalette.ColorRole.Highlight, QColor(88, 166, 255))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(p)
    w = App()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
