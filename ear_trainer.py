#!/usr/bin/env python3
# ear-trainer — random pitch ear trainer, standard 12-TET
# pip3 install sounddevice numpy scipy --user

import tkinter as tk
import sounddevice as sd
import numpy as np
import math
import random
import queue
import threading
import collections
import platform
from scipy.signal import iirnotch, butter, lfilter, lfilter_zi

SR = 44100

SEMITONE_NAMES = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]

def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))

def midi_to_name(midi):
    return SEMITONE_NAMES[midi % 12] + str((midi // 12) - 1)

SCALES = {
    "Chromatic":      list(range(12)),
    "Major":          [0,2,4,5,7,9,11],
    "Natural minor":  [0,2,3,5,7,8,10],
    "Harmonic minor": [0,2,3,5,7,8,11],
    "Melodic minor":  [0,2,3,5,7,9,11],
    "Dorian":         [0,2,3,5,7,9,10],
    "Mixolydian":     [0,2,4,5,7,9,10],
    "Pentatonic maj": [0,2,4,7,9],
    "Pentatonic min": [0,3,5,7,10],
    "Blues":          [0,3,5,6,7,10],
    "Whole tone":     [0,2,4,6,8,10],
    "Diminished":     [0,2,3,5,6,8,9,11],
}

def make_tone(freq, duration, vol=0.7, crossfade=0.3, fade_in=0.3,
              low_cut_db=0.0):
    n    = int(SR * duration)
    t    = np.arange(n) / SR
    wave = np.sin(2 * math.pi * freq * t)
    env  = np.ones(n)
    hold_samples = max(1, n - int(SR * crossfade))
    atk = max(1, min(int(SR * fade_in), int(hold_samples * 0.8)))
    env[:atk] = (1.0 - np.cos(np.linspace(0.0, math.pi, atk))) * 0.5
    rel_start = hold_samples
    rel_len   = n - rel_start
    if rel_len > 0:
        env[rel_start:] = (1.0 + np.cos(np.linspace(0.0, math.pi, rel_len))) * 0.5
    ref_hz = midi_to_hz(60)
    if low_cut_db > 0.0 and freq < ref_hz:
        octaves_below = math.log2(ref_hz / max(1.0, freq))
        frac   = min(1.0, octaves_below / 3.0)
        db_cut = frac * low_cut_db
        vol    = max(0.01, vol * (10.0 ** (-db_cut / 20.0)))
    return (wave * env * vol * 0.55).astype(np.float32)

# ── Audio state ───────────────────────────────────────────────────────────────
_bpm           = [60]
_volume        = [0.5]
_midi_lo       = [48]
_midi_hi       = [72]
_scale_mode    = ["Chromatic"]
_blocked_up    = set()
_blocked_down  = set()
_block_unison  = [True]
_block_octave  = [True]   # blocks same pitch-class, any octave
_wide_bias     = [2.5]
_pitch_rolloff = [5.0]
_low_cut_db    = [6.0]
_xfade_mult    = [1.0]
_fadein_sec    = [1.5]
_hold_mode     = [False]

_total      = [0]
_next       = [0.0]
_note_n     = [0]
_tail       = [np.zeros(0, dtype=np.float32)]
_pending    = [None]
_tone_queue    = queue.Queue(maxsize=3)
_worker_active = [False]

_drone_on    = [True]
_drone_midi  = [37]
_drone_vol   = [0.4]
_drone_phase = [0.0]

_notch_on    = [True]
_notch_freq  = [450.0]
_notch_depth = [0.5]
_notch_ba    = [None]
_notch_zi    = [None]

_live_note_gain  = [1.0]
_live_drone_gain = [0.4]
_live_notch_mix  = [0.5]

_hpf_ba = [None]
_hpf_zi = [None]

_recent_notes = [collections.deque(maxlen=9)]
_history      = []
_show_note    = [True]

def build_notch(freq):
    f0 = max(20.0, min(freq, SR / 2 - 1))
    b, a = iirnotch(f0 / (SR / 2), 0.707)
    zi   = lfilter_zi(b, a) * 0
    return b, a, zi

def build_hpf(cutoff=100.0):
    b, a = butter(2, cutoff / (SR / 2), btype="high")
    zi   = lfilter_zi(b, a) * 0
    return b, a, zi

_hb, _ha, _hzi = build_hpf(100.0)
_hpf_ba[0] = (_hb, _ha)
_hpf_zi[0] = _hzi

_nb, _na, _nzi = build_notch(450.0)
_notch_ba[0] = (_nb, _na)
_notch_zi[0] = _nzi

def random_note():
    lo = _midi_lo[0]; hi = _midi_hi[0]
    if lo > hi: lo, hi = hi, lo
    scale_set  = set(SCALES.get(_scale_mode[0], list(range(12))))
    candidates = [m for m in range(lo, hi + 1) if m % 12 in scale_set]
    if not candidates:
        candidates = list(range(lo, hi + 1))

    recent = _recent_notes[0]

    # Block by recent history (exact midi)
    blocked_recent = set(list(recent)[1:9]) if len(recent) > 0 else set()
    filtered = [m for m in candidates if m not in blocked_recent]
    if filtered:
        candidates = filtered

    last = list(recent)[0] if recent else None
    if last is not None:
        last_pc = last % 12   # pitch class of last note
        allowed = []
        for m in candidates:
            diff     = m - last
            semidiff = abs(diff)
            interval = semidiff % 12

            # Unison: exactly the same midi note
            if semidiff == 0:
                if _block_unison[0]: continue
                else: allowed.append(m); continue

            # Octave: same pitch class (any octave), semidiff > 0
            # This covers unison-in-different-octave AND same note name
            if _block_octave[0] and (m % 12 == last_pc):
                continue

            if diff > 0 and interval in _blocked_up:   continue
            if diff < 0 and interval in _blocked_down: continue
            allowed.append(m)
        if allowed:
            candidates = allowed

    bias = _wide_bias[0]
    if last is None or abs(bias - 1.0) < 0.05:
        choice = random.choice(candidates)
    else:
        weights = [max(1, abs(m - last)) ** bias for m in candidates]
        total   = sum(weights)
        weights = [w / total for w in weights]
        choice  = random.choices(candidates, weights=weights, k=1)[0]

    _recent_notes[0].appendleft(choice)
    return choice

def _render_note(midi=None):
    bpm  = float(_bpm[0])
    if midi is None:
        midi = random_note()
    freq = midi_to_hz(midi)
    name = midi_to_name(midi)
    vol  = _volume[0]
    if _pitch_rolloff[0] > 0.0:
        ref_hz  = midi_to_hz(60)
        octaves = math.log2(max(1.0, freq / ref_hz))
        db_cut  = _pitch_rolloff[0] * max(0.0, octaves)
        vol     = max(0.02, vol * (10.0 ** (-db_cut / 20.0)))
    spb_sec = 60.0 / max(1.0, bpm)
    hold    = 4 * spb_sec
    xfade   = min(hold * 2.0, hold * _xfade_mult[0])
    tone    = make_tone(freq, hold + xfade, vol,
                        crossfade=xfade, fade_in=_fadein_sec[0],
                        low_cut_db=_low_cut_db[0])
    return tone, name, freq, midi

def _prerender_worker():
    while _worker_active[0]:
        try:
            tone, name, freq, midi = _render_note()
            while _worker_active[0]:
                try:
                    _tone_queue.put((tone, name, freq, midi), timeout=0.1)
                    break
                except queue.Full:
                    continue
        except Exception:
            import time; time.sleep(0.05)

def callback(outdata, frames, t, status):
    outdata[:] = 0.0
    note_buf = np.zeros(frames, dtype=np.float32)

    tail = _tail[0]
    if len(tail) > 0:
        n = min(len(tail), frames)
        note_buf[:n] += tail[:n]
        _tail[0] = tail[n:]

    frame_start = _total[0]
    frame_end   = frame_start + frames
    bpm = float(_bpm[0])

    if not _hold_mode[0]:
        while _next[0] < frame_end:
            spb    = SR * 60.0 / max(1.0, bpm)
            offset = int(_next[0]) - frame_start
            if 0 <= offset < frames:
                try:
                    tone, name, freq, midi = _tone_queue.get_nowait()
                except queue.Empty:
                    tone, name, freq, midi = _render_note()
                end_in_block = min(offset + len(tone), frames)
                n = end_in_block - offset
                note_buf[offset:end_in_block] += tone[:n]
                if len(tone) - n > 0:
                    new_tail = tone[n:].copy()
                    old_tail = _tail[0]
                    if len(old_tail) > 0:
                        ml = min(len(old_tail), len(new_tail))
                        new_tail[:ml] += old_tail[:ml]
                        if len(old_tail) > len(new_tail):
                            new_tail = np.concatenate([new_tail, old_tail[len(new_tail):]])
                    _tail[0] = new_tail
                _pending[0] = (name, freq, midi)
            _next[0]   += spb * 4
            _note_n[0] += 1

    note_buf *= _live_note_gain[0]

    drone_buf = np.zeros(frames, dtype=np.float32)
    if _drone_on[0]:
        drone_freq = midi_to_hz(_drone_midi[0])
        phase      = _drone_phase[0]
        t_arr      = (np.arange(frames) / SR) + phase / (2 * math.pi * drone_freq)
        drone_wave = np.sin(2 * math.pi * drone_freq * t_arr) * 0.35
        drone_buf += drone_wave.astype(np.float32)
        _drone_phase[0] = (phase + 2 * math.pi * drone_freq * frames / SR) % (2 * math.pi)
    drone_buf *= _live_drone_gain[0]

    outdata[:, 0] = note_buf + drone_buf

    if _hpf_ba[0] is not None:
        try:
            b, a = _hpf_ba[0]
            filtered, new_zi = lfilter(b, a, outdata[:, 0], zi=_hpf_zi[0])
            outdata[:, 0] = filtered.astype(np.float32)
            _hpf_zi[0] = new_zi
        except Exception:
            pass

    mix = _live_notch_mix[0]
    if _notch_on[0] and _notch_ba[0] is not None and mix > 0.001:
        try:
            b, a = _notch_ba[0]
            filtered, new_zi = lfilter(b, a, outdata[:, 0], zi=_notch_zi[0])
            filtered = filtered.astype(np.float32)
            _notch_zi[0] = new_zi
            outdata[:, 0] = (1.0 - mix) * outdata[:, 0] + mix * filtered
        except Exception:
            pass

    _total[0] += frames

def advance_note():
    if not stream[0]: return
    _tail[0] = np.zeros(0, dtype=np.float32)
    tone, name, freq, midi = _render_note()
    _tail[0] = tone
    _pending[0] = (name, freq, midi)

# ── Playback control ──────────────────────────────────────────────────────────
_poll_running = [False]
stream = [None]

def poll():
    p = _pending[0]
    if p is not None:
        _pending[0] = None
        name, freq, midi = p
        if _show_note[0]:
            note_btn.config(text=name, state="normal")
            octave = midi // 12
            colors = ["#884488","#4466cc","#44aacc","#44bb66","#bbaa22","#cc7722","#cc4444"]
            note_btn.config(fg=colors[min(octave, len(colors)-1)])
        else:
            note_btn.config(text="?", fg="#555")
        freq_lbl.config(text="{:.1f} Hz".format(freq) if _show_note[0] else "")
        _history.append(name)
        if len(_history) > 8: _history.pop(0)
        hist_lbl.config(text="  ".join(_history))
    if _poll_running[0]:
        root.after(16, poll)

def start():
    _total[0] = 0; _next[0] = 0.0; _note_n[0] = 0
    _tail[0]  = np.zeros(0, dtype=np.float32)
    _history.clear(); _recent_notes[0].clear()
    _worker_active[0] = False
    while not _tone_queue.empty():
        try: _tone_queue.get_nowait()
        except: pass
    _worker_active[0] = True
    threading.Thread(target=_prerender_worker, daemon=True).start()
    try:
        stream[0] = sd.OutputStream(samplerate=SR, channels=1,
                                     dtype="float32", blocksize=1024,
                                     callback=callback)
        stream[0].start()
        start_btn.config(text="Stop  [space]", bg="#800")
        note_btn.config(text="—")
        freq_lbl.config(text="")
        _poll_running[0] = True
        poll()
        if _hold_mode[0]:
            advance_note()
    except Exception as e:
        print("Audio error:", e)
        _worker_active[0] = False
        start_btn.config(text="Start  [space]", bg="#222")

def stop():
    _worker_active[0] = False
    _poll_running[0]  = False
    if stream[0]:
        stream[0].stop(); stream[0].close(); stream[0] = None
    start_btn.config(text="Start  [space]", bg="#222")
    note_btn.config(text="—", fg="#eee")
    freq_lbl.config(text="")

def toggle():
    if stream[0]: stop()
    else:         start()

def _on_space(e):
    if _hold_mode[0] and stream[0]:
        advance_note()
    else:
        toggle()
    return "break"

# ── UI ────────────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("ear-trainer")
root.geometry("420x900")
root.configure(bg="#111")
root.resizable(True, True)

bg = "#111"; fg = "#eee"

_canvas    = tk.Canvas(root, bg=bg, highlightthickness=0)
_scrollbar = tk.Scrollbar(root, orient="vertical", command=_canvas.yview)
_canvas.configure(yscrollcommand=_scrollbar.set)
_scrollbar.pack(side="right", fill="y")
_canvas.pack(side="left", fill="both", expand=True)
_inner = tk.Frame(_canvas, bg=bg)
_cwin  = _canvas.create_window((0, 0), window=_inner, anchor="nw")

def _on_frame_cfg(e):  _canvas.configure(scrollregion=_canvas.bbox("all"))
def _on_canvas_cfg(e): _canvas.itemconfig(_cwin, width=e.width)
_inner.bind("<Configure>", _on_frame_cfg)
_canvas.bind("<Configure>", _on_canvas_cfg)

def _on_mwheel(e):
    if platform.system() == "Darwin":
        _canvas.yview_scroll(-1 * int(e.delta), "units")
    else:
        _canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
root.bind_all("<MouseWheel>", _on_mwheel)
root.bind_all("<Button-4>", lambda e: _canvas.yview_scroll(-1, "units"))
root.bind_all("<Button-5>", lambda e: _canvas.yview_scroll(1, "units"))

def section(text):
    tk.Label(_inner, text=text, bg=bg, fg="#aaa",
             font=("Helvetica",11,"bold")).pack(anchor="w", padx=20, pady=(14,4))

def sep():
    tk.Frame(_inner, bg="#2a2a2a", height=1).pack(fill="x", padx=20, pady=6)

# ── Striped canvas button helper ──────────────────────────────────────────────
# When a block is ACTIVE (preventing notes), the button looks greyed-out
# with diagonal stripes drawn on a canvas widget to make it visually distinct.

def make_block_btn(parent, label_blocked, label_ok, width_px, on_click,
                   txt_blocked="#666", txt_ok="#888"):
    """
    Canvas button with two states:
    - BLOCKED (active=True):  grey + diagonal stripes, label_blocked text
    - OK      (active=False): plain dark,              label_ok text
    Returns (canvas, set_state_fn).
    """
    H = 26
    c = tk.Canvas(parent, width=width_px, height=H,
                  bg="#222", highlightthickness=1,
                  highlightbackground="#333", cursor="hand2")
    _state = [False]

    def draw(active):
        c.delete("all")
        w = width_px
        if active:
            c.config(bg="#2a2a2a", highlightbackground="#444")
            for x in range(-H, w + H, 6):
                c.create_line(x, 0, x + H, H, fill="#383838", width=1)
            c.create_text(w // 2, H // 2, text=label_blocked,
                          fill=txt_blocked, font=("Helvetica", 9))
        else:
            c.config(bg="#222", highlightbackground="#333")
            c.create_text(w // 2, H // 2, text=label_ok,
                          fill=txt_ok, font=("Helvetica", 9))

    c.bind("<Button-1>", lambda _e: (on_click(), draw(_state[0])))

    def set_state(active):
        _state[0] = active
        draw(active)

    draw(False)
    return c, set_state

# ════════════════════════════════════════════════════════
# SECTION 1: Pitch Range
# ════════════════════════════════════════════════════════
section("Pitch Range")

flo = tk.Frame(_inner, bg=bg); flo.pack(fill="x", padx=20, pady=(4,2))
tk.Label(flo, text="Low note", bg=bg, fg="#777",
         font=("Helvetica",10), width=16, anchor="w").pack(side="left")
lo_lbl = tk.Label(flo, text=midi_to_name(48), bg=bg, fg="#44aacc",
                   font=("Helvetica",13,"bold"), width=6, anchor="w")
lo_lbl.pack(side="left", padx=4)
def on_lo(v):
    _midi_lo[0] = int(float(v)); lo_lbl.config(text=midi_to_name(int(float(v))))
lo_sl = tk.Scale(_inner, from_=24, to=96, orient="horizontal", showvalue=False,
                  bg=bg, fg=fg, highlightthickness=0, troughcolor="#333",
                  length=320, command=on_lo)
lo_sl.set(48); lo_sl.pack(padx=20)

fhi = tk.Frame(_inner, bg=bg); fhi.pack(fill="x", padx=20, pady=(6,2))
tk.Label(fhi, text="High note", bg=bg, fg="#777",
         font=("Helvetica",10), width=16, anchor="w").pack(side="left")
hi_lbl = tk.Label(fhi, text=midi_to_name(72), bg=bg, fg="#cc7722",
                   font=("Helvetica",13,"bold"), width=6, anchor="w")
hi_lbl.pack(side="left", padx=4)
def on_hi(v):
    _midi_hi[0] = int(float(v)); hi_lbl.config(text=midi_to_name(int(float(v))))
hi_sl = tk.Scale(_inner, from_=24, to=96, orient="horizontal", showvalue=False,
                  bg=bg, fg=fg, highlightthickness=0, troughcolor="#333",
                  length=320, command=on_hi)
hi_sl.set(72); hi_sl.pack(padx=20)

# ════════════════════════════════════════════════════════
# SECTION 2: Scale
# ════════════════════════════════════════════════════════
section("Scale")
scale_var = tk.StringVar(value="Chromatic")
scale_names = list(SCALES.keys())
for row_start in range(0, len(scale_names), 4):
    row = tk.Frame(_inner, bg=bg); row.pack(fill="x", padx=20, pady=1)
    for sc in scale_names[row_start:row_start+4]:
        tk.Radiobutton(row, text=sc, variable=scale_var, value=sc,
                       bg=bg, fg=fg, selectcolor="#333", activebackground=bg,
                       font=("Helvetica",10),
                       command=lambda: _scale_mode.__setitem__(0, scale_var.get())
                       ).pack(side="left", padx=3)

sep()

# ════════════════════════════════════════════════════════
# SECTION 3: Tempo
# ════════════════════════════════════════════════════════
section("Tempo")
ft = tk.Frame(_inner, bg=bg); ft.pack(fill="x", padx=20, pady=(4,2))
tk.Label(ft, text="BPM", bg=bg, fg="#777",
         font=("Helvetica",10), width=16, anchor="w").pack(side="left")
bpm_lbl = tk.Label(ft, text="60", bg=bg, fg=fg,
                    font=("Helvetica",18,"bold"), width=5)
bpm_lbl.pack(side="left", padx=4)
def on_bpm(v):
    _bpm[0] = int(float(v)); bpm_lbl.config(text=str(int(float(v))))
tk.Scale(_inner, from_=10, to=200, orient="horizontal", showvalue=False,
         bg=bg, fg=fg, highlightthickness=0, troughcolor="#333",
         length=320, command=on_bpm).pack(padx=20)

sep()

# ════════════════════════════════════════════════════════
# SECTION 4: Crossfade / Fade
# ════════════════════════════════════════════════════════
section("Crossfade / Fade")

fxf = tk.Frame(_inner, bg=bg); fxf.pack(fill="x", padx=20, pady=(0,2))
tk.Label(fxf, text="Crossfade tail", bg=bg, fg="#777",
         font=("Helvetica",10), width=16, anchor="w").pack(side="left")
xf_lbl = tk.Label(fxf, text="1.0×", bg=bg, fg="#88ccaa",
                   font=("Helvetica",13,"bold"), width=6, anchor="w")
xf_lbl.pack(side="left", padx=4)
def on_xfade(v):
    val = float(v); _xfade_mult[0] = val; xf_lbl.config(text="{:.1f}×".format(val))
xfs = tk.Scale(_inner, from_=0.1, to=2.0, resolution=0.1,
               orient="horizontal", showvalue=False,
               bg=bg, fg=fg, highlightthickness=0, troughcolor="#333",
               length=320, command=on_xfade)
xfs.set(1.0); xfs.pack(padx=20)

ffi = tk.Frame(_inner, bg=bg); ffi.pack(fill="x", padx=20, pady=(8,2))
tk.Label(ffi, text="Fade-in time", bg=bg, fg="#777",
         font=("Helvetica",10), width=16, anchor="w").pack(side="left")
fi_lbl = tk.Label(ffi, text="1.50 s", bg=bg, fg="#aaccff",
                   font=("Helvetica",13,"bold"), width=6, anchor="w")
fi_lbl.pack(side="left", padx=4)
def on_fadein(v):
    val = float(v); _fadein_sec[0] = val; fi_lbl.config(text="{:.2f} s".format(val))
fis = tk.Scale(_inner, from_=0.01, to=8.0, resolution=0.05,
               orient="horizontal", showvalue=False,
               bg=bg, fg=fg, highlightthickness=0, troughcolor="#333",
               length=320, command=on_fadein)
fis.set(1.5); fis.pack(padx=20)

sep()

# ════════════════════════════════════════════════════════
# SECTION 5: Blocked intervals
# ════════════════════════════════════════════════════════
section("Block intervals")

_up_btns   = {}
_down_btns = {}

interval_defs = [
    (1,"m2"),(2,"M2"),(3,"m3"),(4,"M3"),(5,"P4"),(6,"TT"),
    (7,"P5"),(8,"m6"),(9,"M6"),(10,"m7"),(11,"M7"),(12,"Oct")
]

def toggle_up(s):
    if s in _blocked_up:
        _blocked_up.discard(s)
        _up_btns[s][1](False)
    else:
        _blocked_up.add(s)
        _up_btns[s][1](True)

def toggle_down(s):
    if s in _blocked_down:
        _blocked_down.discard(s)
        _down_btns[s][1](False)
    else:
        _blocked_down.add(s)
        _down_btns[s][1](True)

int_frame = tk.Frame(_inner, bg=bg)
int_frame.pack(fill="x", padx=20, pady=(0,4))

# No unison / no octave — striped canvas buttons
uo_row = tk.Frame(int_frame, bg=bg); uo_row.pack(fill="x", pady=(0,8))

def toggle_unison():
    _block_unison[0] = not _block_unison[0]
    unison_set(_block_unison[0])

def toggle_octave():
    _block_octave[0] = not _block_octave[0]
    octave_set(_block_octave[0])

unison_canvas, unison_set = make_block_btn(
    uo_row, "unisons not ok", "unisons ok", 100, toggle_unison)
unison_canvas.pack(side="left", padx=(0,6))
unison_set(True)   # default: blocking ON = striped

octave_canvas, octave_set = make_block_btn(
    uo_row, "same note not ok", "same note ok", 114, toggle_octave)
octave_canvas.pack(side="left")
octave_set(True)   # default: blocking ON = striped

tk.Label(uo_row, text="(any octave)", bg=bg, fg="#444",
         font=("Helvetica",9)).pack(side="left", padx=8)

# Wide interval bias
wb_hdr = tk.Frame(int_frame, bg=bg); wb_hdr.pack(fill="x", pady=(8,2))
tk.Label(wb_hdr, text="Wide interval bias", bg=bg, fg="#ccaa44",
         font=("Helvetica",12,"bold"), anchor="w").pack(side="left")
wb_lbl = tk.Label(wb_hdr, text="medium", bg=bg, fg="#ffcc44",
                   font=("Helvetica",16,"bold"), width=8, anchor="e")
wb_lbl.pack(side="right")
def on_wide_bias(v):
    val = float(v); _wide_bias[0] = val
    wb_lbl.config(text="flat"   if val < 1.1 else
                       "slight" if val < 2.0 else
                       "medium" if val < 3.0 else "strong")
wb_sl = tk.Scale(int_frame, from_=1.0, to=4.0, resolution=0.1,
                  orient="horizontal", showvalue=False,
                  bg=bg, fg=fg, highlightthickness=0,
                  troughcolor="#443300", activebackground="#ffcc44",
                  length=340, command=on_wide_bias)
wb_sl.set(2.5); wb_sl.pack(fill="x", pady=(0,6))

for row_start in [0, 6]:
    up_row = tk.Frame(int_frame, bg=bg); up_row.pack(fill="x", pady=1)
    tk.Label(up_row, text="↑", bg=bg, fg="#4af",
             font=("Helvetica",9), width=2).pack(side="left")
    for semi, iname in interval_defs[row_start:row_start+6]:
        c, set_fn = make_block_btn(up_row, iname, iname, 44,
                                   lambda s=semi: toggle_up(s),
                                   txt_blocked="#445566", txt_ok="#4af")
        c.pack(side="left", padx=2)
        _up_btns[semi] = (c, set_fn)

    dn_row = tk.Frame(int_frame, bg=bg); dn_row.pack(fill="x", pady=1)
    tk.Label(dn_row, text="↓", bg=bg, fg="#f84",
             font=("Helvetica",9), width=2).pack(side="left")
    for semi, iname in interval_defs[row_start:row_start+6]:
        c, set_fn = make_block_btn(dn_row, iname, iname, 44,
                                   lambda s=semi: toggle_down(s),
                                   txt_blocked="#664433", txt_ok="#f84")
        c.pack(side="left", padx=2)
        _down_btns[semi] = (c, set_fn)

sep()

# ════════════════════════════════════════════════════════
# SECTION 6: Display
# ════════════════════════════════════════════════════════
disp = tk.Frame(_inner, bg="#0a0a0a")
disp.pack(fill="x", padx=20, pady=(8,4))

def on_note_btn():
    if not _show_note[0] and _history:
        note_btn.config(text=_history[-1], fg="#eee")
        freq_lbl.config(fg="#888")

note_btn = tk.Button(disp, text="—", bg="#0a0a0a", fg="#eee",
                      font=("Helvetica",52,"bold"),
                      relief="flat", bd=0, cursor="hand2",
                      activebackground="#0a0a0a", activeforeground="#eee",
                      command=on_note_btn)
note_btn.pack(pady=(10,0))

freq_lbl = tk.Label(disp, text="", bg="#0a0a0a", fg="#555",
                     font=("Helvetica",12))
freq_lbl.pack(pady=(0,4))

hist_lbl = tk.Label(disp, text="", bg="#0a0a0a", fg="#444",
                     font=("Helvetica",11))
hist_lbl.pack(pady=(0,6))

vis_row = tk.Frame(disp, bg="#0a0a0a"); vis_row.pack(pady=(0,8))
vis_var = tk.BooleanVar(value=True)
def on_vis_toggle():
    _show_note[0] = vis_var.get()
    if not _show_note[0]:
        note_btn.config(text="?", fg="#555")
        freq_lbl.config(text="")
    else:
        freq_lbl.config(fg="#555")
        if _history:
            note_btn.config(text=_history[-1], fg="#eee")
tk.Checkbutton(vis_row, text="Show note name", variable=vis_var,
               command=on_vis_toggle, bg="#0a0a0a", fg="#666",
               selectcolor="#1a1a1a", activebackground="#0a0a0a",
               font=("Helvetica",10)).pack(side="left", padx=6)

sep()

# ════════════════════════════════════════════════════════
# SECTION 7: Volume
# ════════════════════════════════════════════════════════
section("Volume")

fv = tk.Frame(_inner, bg=bg); fv.pack(fill="x", padx=20, pady=(4,2))
tk.Label(fv, text="Note volume", bg=bg, fg="#777",
         font=("Helvetica",10), width=16, anchor="w").pack(side="left")
vol_lbl = tk.Label(fv, text="50%", bg=bg, fg=fg,
                    font=("Helvetica",13,"bold"), width=6)
vol_lbl.pack(side="left", padx=4)
def on_vol(v):
    val = float(v) / 100.0
    _volume[0] = val
    _live_note_gain[0] = val / 0.5
    vol_lbl.config(text="{}%".format(int(float(v))))
vs = tk.Scale(_inner, from_=0, to=100, orient="horizontal", showvalue=False,
              bg=bg, fg=fg, highlightthickness=0, troughcolor="#333",
              length=320, command=on_vol)
vs.set(50); vs.pack(padx=20)

fpr = tk.Frame(_inner, bg=bg); fpr.pack(fill="x", padx=20, pady=(8,2))
tk.Label(fpr, text="High note rolloff", bg=bg, fg="#777",
         font=("Helvetica",10), width=16, anchor="w").pack(side="left")
pr_lbl = tk.Label(fpr, text="5.0 dB/oct", bg=bg, fg="#cc88aa",
                   font=("Helvetica",13,"bold"), width=10, anchor="w")
pr_lbl.pack(side="left", padx=4)
def on_rolloff(v):
    val = float(v); _pitch_rolloff[0] = val
    pr_lbl.config(text="off" if val < 0.1 else "{:.1f} dB/oct".format(val))
prs = tk.Scale(_inner, from_=0.0, to=12.0, resolution=0.5,
               orient="horizontal", showvalue=False,
               bg=bg, fg=fg, highlightthickness=0, troughcolor="#333",
               length=320, command=on_rolloff)
prs.set(5.0); prs.pack(padx=20)

flc = tk.Frame(_inner, bg=bg); flc.pack(fill="x", padx=20, pady=(8,2))
tk.Label(flc, text="Low note cut", bg=bg, fg="#777",
         font=("Helvetica",10), width=16, anchor="w").pack(side="left")
lc_lbl = tk.Label(flc, text="6 dB", bg=bg, fg="#cc88aa",
                   font=("Helvetica",13,"bold"), width=8, anchor="w")
lc_lbl.pack(side="left", padx=4)
def on_low_cut(v):
    val = float(v); _low_cut_db[0] = val
    lc_lbl.config(text="off" if val < 0.5 else "{:.0f} dB".format(val))
lcs = tk.Scale(_inner, from_=0, to=24, resolution=1,
               orient="horizontal", showvalue=False,
               bg=bg, fg=fg, highlightthickness=0, troughcolor="#333",
               length=320, command=on_low_cut)
lcs.set(6); lcs.pack(padx=20)

sep()

# ════════════════════════════════════════════════════════
# SECTION 8: Drone
# ════════════════════════════════════════════════════════
section("Drone")
drone_row = tk.Frame(_inner, bg=bg); drone_row.pack(fill="x", padx=20, pady=(0,4))
drone_var = tk.BooleanVar(value=True)
def on_drone_toggle(): _drone_on[0] = drone_var.get()
tk.Checkbutton(drone_row, text="Enable drone", variable=drone_var,
               command=on_drone_toggle, bg=bg, fg=fg, selectcolor="#333",
               activebackground=bg, font=("Helvetica",10)).pack(side="left")

fd = tk.Frame(_inner, bg=bg); fd.pack(fill="x", padx=20, pady=(0,2))
tk.Label(fd, text="Drone note", bg=bg, fg="#777",
         font=("Helvetica",10), width=16, anchor="w").pack(side="left")
drone_lbl = tk.Label(fd, text=midi_to_name(37), bg=bg, fg="#88aaff",
                      font=("Helvetica",13,"bold"), width=5)
drone_lbl.pack(side="left", padx=4)
def on_drone_note(v):
    m = int(float(v)); _drone_midi[0] = m; drone_lbl.config(text=midi_to_name(m))
dk = tk.Scale(_inner, from_=24, to=60, orient="horizontal", showvalue=False,
              bg=bg, fg=fg, highlightthickness=0, troughcolor="#333",
              length=320, command=on_drone_note)
dk.set(37); dk.pack(padx=20)

fdv = tk.Frame(_inner, bg=bg); fdv.pack(fill="x", padx=20, pady=(6,2))
tk.Label(fdv, text="Drone volume", bg=bg, fg="#777",
         font=("Helvetica",10), width=16, anchor="w").pack(side="left")
dvol_lbl = tk.Label(fdv, text="40%", bg=bg, fg="#88aaff",
                     font=("Helvetica",13,"bold"), width=5)
dvol_lbl.pack(side="left", padx=4)
def on_drone_vol(v):
    val = float(v) / 100.0
    _drone_vol[0]       = val
    _live_drone_gain[0] = val
    dvol_lbl.config(text="{}%".format(int(float(v))))
dvs = tk.Scale(_inner, from_=0, to=100, orient="horizontal", showvalue=False,
               bg=bg, fg=fg, highlightthickness=0, troughcolor="#333",
               length=320, command=on_drone_vol)
dvs.set(40); dvs.pack(padx=20)

sep()

# ════════════════════════════════════════════════════════
# SECTION 9: Notch filter
# ════════════════════════════════════════════════════════
section("Notch Filter")
notch_row = tk.Frame(_inner, bg=bg); notch_row.pack(fill="x", padx=20, pady=(0,4))
notch_var = tk.BooleanVar(value=True)
def on_notch_toggle():
    _notch_on[0] = notch_var.get()
    if _notch_on[0]:
        b, a, zi = build_notch(_notch_freq[0])
        _notch_ba[0] = (b, a); _notch_zi[0] = zi
tk.Checkbutton(notch_row, text="Enable notch", variable=notch_var,
               command=on_notch_toggle, bg=bg, fg=fg, selectcolor="#333",
               activebackground=bg, font=("Helvetica",10)).pack(side="left")

fn = tk.Frame(_inner, bg=bg); fn.pack(fill="x", padx=20, pady=(0,2))
tk.Label(fn, text="Notch freq", bg=bg, fg="#777",
         font=("Helvetica",10), width=16, anchor="w").pack(side="left")
notch_lbl = tk.Label(fn, text="450 Hz", bg=bg, fg="#aacc88",
                      font=("Helvetica",13,"bold"), width=8)
notch_lbl.pack(side="left", padx=4)
def on_notch_freq(v):
    f = float(v); _notch_freq[0] = f
    notch_lbl.config(text="{:.0f} Hz".format(f))
    if _notch_on[0]:
        b, a, zi = build_notch(f)
        _notch_ba[0] = (b, a); _notch_zi[0] = zi
nfs = tk.Scale(_inner, from_=80, to=8000, orient="horizontal", showvalue=False,
               bg=bg, fg=fg, highlightthickness=0, troughcolor="#333",
               length=320, command=on_notch_freq)
nfs.set(450); nfs.pack(padx=20)

fnd = tk.Frame(_inner, bg=bg); fnd.pack(fill="x", padx=20, pady=(6,2))
tk.Label(fnd, text="Notch depth", bg=bg, fg="#777",
         font=("Helvetica",10), width=16, anchor="w").pack(side="left")
notch_depth_lbl = tk.Label(fnd, text="50%", bg=bg, fg="#aacc88",
                             font=("Helvetica",13,"bold"), width=6)
notch_depth_lbl.pack(side="left", padx=4)
def on_notch_depth(v):
    val = float(v) / 100.0
    _notch_depth[0]    = val
    _live_notch_mix[0] = val
    notch_depth_lbl.config(text="{}%".format(int(float(v))))
nds = tk.Scale(_inner, from_=0, to=100, orient="horizontal", showvalue=False,
               bg=bg, fg=fg, highlightthickness=0, troughcolor="#333",
               length=320, command=on_notch_depth)
nds.set(50); nds.pack(padx=20)

sep()

# ════════════════════════════════════════════════════════
# SECTION 10: Transport
# ════════════════════════════════════════════════════════
transport = tk.Frame(_inner, bg=bg); transport.pack(fill="x", padx=20, pady=(4,4))

start_btn = tk.Button(transport, text="Start  [space]", width=14,
                       font=("Helvetica",14), relief="flat",
                       bg="#222", fg=fg, cursor="hand2", command=toggle)
start_btn.pack(side="left", pady=(0,10), padx=(0,10))

hold_var = tk.BooleanVar(value=False)
def on_hold_toggle():
    _hold_mode[0] = hold_var.get()
    next_btn.config(state="normal" if _hold_mode[0] else "disabled")
    hold_toggle.config(bg="#003355" if _hold_mode[0] else "#222",
                       fg="#4af"    if _hold_mode[0] else "#888")
hold_toggle = tk.Checkbutton(transport, text="Hold", variable=hold_var,
                               command=on_hold_toggle,
                               bg="#222", fg="#888", selectcolor="#003355",
                               activebackground="#222",
                               font=("Helvetica",11), relief="flat",
                               cursor="hand2", indicatoron=False,
                               width=6, pady=6)
hold_toggle.pack(side="left", padx=(0,6))

next_btn = tk.Button(transport, text="Next  [space]", width=12,
                      font=("Helvetica",11), relief="flat",
                      bg="#113322", fg="#4af", cursor="hand2",
                      state="disabled", command=advance_note)
next_btn.pack(side="left")

tk.Label(_inner, text="", bg=bg).pack(pady=8)

root.bind("<space>", _on_space)
root.mainloop()
