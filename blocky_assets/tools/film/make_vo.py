"""Narration + timeline: python3 make_vo.py <piper-voice.onnx>   (pip install piper-tts)
Reads lines.json, writes vo/<id>.wav and timeline.json (shot start/end follow the voice)."""
import json, subprocess, sys, wave, os
voice = sys.argv[1]
L = json.load(open('lines.json')); os.makedirs('vo', exist_ok=True)
t, gap = 0.25, 0.18
for l in L:
    out = f"vo/{l['id']}.wav"
    subprocess.run([sys.executable, '-m', 'piper', '-m', voice, '--length-scale', '0.88', '--sentence-silence', '0', '-f', out],
                   input=l['text'].replace('*', '').encode(), check=True, capture_output=True)
    w = wave.open(out); d = w.getnframes() / w.getframerate(); w.close()
    l['vo_start'], l['vo_dur'] = round(t, 3), round(d, 3)
    l['start'] = 0 if l is L[0] else round(t - 0.12, 3)
    t += d + gap + (0.35 if l.get('beat') else 0)
    l['end'] = round(t - 0.12, 3)
L[-1]['end'] = round(L[-1]['end'] + 1.6, 3)
json.dump(L, open('timeline.json', 'w'), indent=1)
print('total', L[-1]['end'], 's')
