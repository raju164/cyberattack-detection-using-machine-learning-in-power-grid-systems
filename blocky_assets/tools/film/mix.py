import json, wave, numpy as np
SR = 44100
TL = json.load(open('timeline.json'))
DUR = TL[-1]['end'] + 0.1
N = int(DUR * SR)
rng = np.random.default_rng(1)
t_all = np.arange(N) / SR
music = np.zeros(N); sfx = np.zeros(N); vo = np.zeros(N)

def add(buf, start, sig, gain=1.0):
    i = int(start * SR); j = min(N, i + len(sig))
    if i < N: buf[i:j] += sig[:j - i] * gain

def env(n, a=0.005, r=0.3):
    t = np.arange(n) / SR
    return np.minimum(1, t / max(a, 1e-4)) * np.exp(-t / r)

def note(f, dur, kind='pluck', a=0.005, r=0.25):
    n = int(dur * SR); t = np.arange(n) / SR
    if kind == 'pluck':
        s = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)
        return s * env(n, a, r)
    if kind == 'pad':
        s = sum(np.sin(2 * np.pi * f * d * t) for d in (1, 1.003, 0.997)) / 3
        att = np.minimum(1, t / 0.4); rel = np.minimum(1, (dur - t) / 0.5)
        return s * att * np.clip(rel, 0, 1)
    if kind == 'bell':
        s = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 2.76 * t) + 0.25 * np.sin(2 * np.pi * f * 5.4 * t)
        return s * env(n, 0.002, 0.6)

mid = lambda m: 440 * 2 ** ((m - 69) / 12)
# --- section A: playful (0 -> 19.44), mysterious dusk (19.44 -> 25.15)
bpm = 112; beat = 60 / bpm
prog = [[60, 64, 67], [55, 59, 62], [57, 60, 64], [53, 57, 60]]  # C G Am F
A_end = 19.44
t = 0.0; bar = 0
while t < A_end:
    ch = prog[bar % 4]
    for k in range(8):
        tt = t + k * beat / 2
        if tt >= A_end: break
        m = ch[[0, 1, 2, 1, 0 + 0, 2, 1, 2][k]] + 12
        add(music, tt, note(mid(m), 0.5, 'pluck', r=0.18), 0.16)
    add(music, t, note(mid(ch[0] - 12), beat * 4, 'pluck', r=0.6), 0.22)
    add(music, t, note(mid(ch[0] - 24), beat * 4, 'pad'), 0.10)
    t += beat * 4; bar += 1
# dusk mystery: slow minor pads + music-box notes
for i, (ch, tt) in enumerate(zip([[57, 60, 64], [53, 57, 62], [52, 55, 59]], [19.44, 21.3, 23.2])):
    for m in ch: add(music, tt, note(mid(m - 12), 2.1, 'pad'), 0.07)
for k, m in enumerate([76, 72, 69, 71, 74, 72, 68, 69]):
    add(music, 19.6 + k * 0.68, note(mid(m), 1.0, 'bell'), 0.06)
# --- section B: night tension 25.15 -> 39.76
B0, B1 = 25.15, 39.76
n = int((B1 - B0) * SR); tt = np.arange(n) / SR
drone = (np.sin(2 * np.pi * 55 * tt) + 0.5 * np.sin(2 * np.pi * 82.4 * tt) + 0.3 * np.sin(2 * np.pi * 110.5 * tt))
drone *= np.minimum(1, tt / 1.5) * np.clip((B1 - B0 - tt) / 1.0, 0, 1)
add(music, B0, drone, 0.09)
k = 0
while B0 + 1.4 + k * 0.62 < 37.9:  # heartbeat pulses
    for off, g in ((0, 1), (0.16, 0.6)):
        p = note(48, 0.25, 'pluck', r=0.09)
        add(music, B0 + 1.4 + k * 0.62 + off, p, 0.5 * g)
    k += 1
for tt0, m in ((32.96, 72), (33.5, 75), (34.5, 79), (35.9, 72), (36.6, 75), (37.3, 79), (38.0, 84)):
    add(music, tt0, note(mid(m), 1.2, 'bell'), 0.05)
# --- section C: warm resolve 39.76 -> end
for i, ch in enumerate([[60, 64, 67, 72], [65, 69, 72, 77], [60, 64, 67, 76]]):
    for m in ch: add(music, 39.76 + i * 1.7, note(mid(m - 12), 2.2, 'pad'), 0.06)
for k, m in enumerate([72, 76, 79, 84, 79, 84, 88]):
    add(music, 40.0 + k * 0.42, note(mid(m), 0.8, 'bell'), 0.05)

# --- replace synth score with instrumental from reference video (DfOwQkgBSiz)
import os
if os.environ.get('BGM'):
    w = wave.open(os.environ['BGM']); sr = w.getframerate(); ch = w.getnchannels(); sw = w.getsampwidth()
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16 if sw == 2 else np.int32).astype(np.float32)
    x /= 32768 if sw == 2 else 2 ** 31; w.close()
    x = x.reshape(-1, ch).mean(1)
    if sr != SR: x = np.interp(np.arange(int(len(x) * SR / sr)) * sr / SR, np.arange(len(x)), x)
    x = x[:N]; x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 0.12   # normalise to ~-18 dBFS
    # night section: muffled + quieter for tension
    B0, B1 = TL[12]['start'], TL[21]['start']
    i0, i1 = int(B0 * SR), int(B1 * SR)
    seg = x[i0:i1].copy(); lp = np.zeros_like(seg); acc = 0.0
    for i in range(len(seg)): acc += 0.06 * (seg[i] - acc); lp[i] = acc
    xf = int(0.4 * SR); wgt = np.ones(len(seg)); wgt[:xf] = np.linspace(0, 1, xf); wgt[-xf:] = np.linspace(1, 0, xf)
    x[i0:i1] = seg * (1 - wgt) + lp * 1.6 * wgt
    fi = int(0.5 * SR); x[:fi] *= np.linspace(0, 1, fi)
    music = x * 1.1 if len(x) == N else np.pad(x, (0, N - len(x))) * 1.1
    # keep the tense heartbeat pulses from the synth score under the night section
    k = 0
    while B0 + 1.4 + k * 0.62 < 37.9:
        for off, g in ((0, 1), (0.16, 0.6)): add(music, B0 + 1.4 + k * 0.62 + off, note(48, 0.25, 'pluck', r=0.09), 0.35 * g)
        k += 1

# --- SFX
def noise(d): return rng.standard_normal(int(d * SR))
def lowpass(x, a):
    y = np.zeros_like(x); acc = 0.0
    for i in range(len(x)): acc += a * (x[i] - acc); y[i] = acc
    return y
def crack(): x = noise(0.12) * env(int(0.12 * SR), 0.001, 0.03); return x
def whoosh(d=0.28):
    x = lowpass(noise(d), 0.08); n = len(x); e = np.sin(np.pi * np.arange(n) / n) ** 2; return x * e
def thump():
    n = int(0.5 * SR); t = np.arange(n) / SR
    return np.sin(2 * np.pi * (90 * np.exp(-t * 6) + 35) * t) * env(n, 0.002, 0.18)
def rumble(d):
    x = lowpass(noise(d), 0.01) * 6; n = len(x); t = np.arange(n) / SR
    return x * np.minimum(1, t / 0.3) * np.clip((d - t) / 0.4, 0, 1)
def sad_trombone():
    out = []
    for f, d in ((233, 0.32), (220, 0.32), (207, 0.32), (196, 0.9)):
        n = int(d * SR); t = np.arange(n) / SR
        vib = 1 + (0.012 * np.sin(2 * np.pi * 6 * t) if d > 0.5 else 0)
        ph = 2 * np.pi * np.cumsum(f * vib) / SR
        s = np.sign(np.sin(ph)) * 0.3 + np.sin(ph) * 0.7
        out.append(lowpass(s, 0.15) * np.minimum(1, t / 0.03) * np.clip((d - t) / 0.08, 0, 1))
    return np.concatenate(out)
def sweep_down(d=1.2):
    n = int(d * SR); t = np.arange(n) / SR
    f = 600 * np.exp(-t * 2.5) + 40; ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.clip((d - t) / d, 0, 1)
def sparkle():
    out = np.zeros(int(1.2 * SR))
    for k, m in enumerate([84, 88, 91, 96]):
        s = note(mid(m), 0.8, 'bell'); i = int(k * 0.09 * SR); out[i:i + len(s)] += s[:len(out) - i]
    return out
def boom():
    n = int(1.6 * SR); t = np.arange(n) / SR
    return (np.sin(2 * np.pi * (60 * np.exp(-t * 2) + 30) * t) + lowpass(noise(1.6), 0.02) * 3) * env(n, 0.01, 0.5)

L = {l['id']: l for l in TL}
for l in TL[1:]: add(sfx, l['start'] - 0.12, whoosh(), 0.10)
add(sfx, L['s03']['start'] + 0.5, crack(), 0.5); add(sfx, L['s03']['start'] + 0.62, sparkle(), 0.12)
add(sfx, L['s04']['start'], whoosh(0.6), 0.2)
for k in range(5): add(sfx, L['s05']['start'] + 0.1 + k * 0.2, crack() * 0.4, 0.35)
add(sfx, L['s05']['end'] - 0.15, crack(), 0.55)
add(sfx, L['s07']['start'] + 0.2, sad_trombone(), 0.22)
add(sfx, L['s13']['start'], rumble(3.2), 0.5)
add(sfx, L['s14']['start'] + 0.1, boom(), 0.35)
add(sfx, L['s15']['start'] + 1.05, thump(), 0.7); add(sfx, L['s15']['start'] + 0.2, whoosh(0.7), 0.25)
add(sfx, L['s16']['start'] + 0.8, thump(), 0.7)
add(sfx, L['s21']['start'] + 0.05, sweep_down(1.4), 0.12)
add(sfx, L['s23']['start'] + 1.6, sparkle(), 0.15)

# --- VO (22.05k -> 44.1k)
for l in TL:
    w = wave.open(f"vo/{l['id']}.wav"); sr = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768; w.close()
    xi = np.interp(np.arange(int(len(x) * SR / sr)) * sr / SR, np.arange(len(x)), x)
    add(vo, l['vo_start'], xi, 1.0)

# duck music under VO
vo_env = lowpass(np.abs(vo), 0.0008) * 8
duck = 1 - 0.45 * np.clip(vo_env, 0, 1)
mixd = vo * 0.95 + music * duck * 0.9 + sfx
mixd /= max(1.0, np.abs(mixd).max() / 0.95)
# fade end
fade = int(0.6 * SR); mixd[-fade:] *= np.linspace(1, 0, fade)
st = np.stack([mixd, mixd], 1)
w = wave.open('mix.wav', 'wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes((st * 32767).astype(np.int16).tobytes()); w.close()
print('mix ok', DUR, 'bgm' if os.environ.get('BGM') else 'synth')
