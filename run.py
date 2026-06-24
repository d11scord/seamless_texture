"""
Прогон пайплайна по всем картинкам из examples/inputs.
Сохраняет в examples/results: *_seamless.png, *_tiled3x3.png, montage_*.png, report.json
"""
import seamless as s
import numpy as np
import cv2, glob, os, json

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, 'examples', 'inputs')
OUT = os.path.join(HERE, 'examples', 'results')
os.makedirs(OUT, exist_ok=True)

report = {}
for f in sorted(glob.glob(os.path.join(IN, '*.png'))):
    nm = os.path.splitext(os.path.basename(f))[0]
    img = s.load(f)
    out, meta = s.make_seamless(img, method='auto')
    sc_in, sc_out = s.seam_score(img), s.seam_score(out)

    s.save(os.path.join(OUT, f'{nm}_seamless.png'), out)
    s.save(os.path.join(OUT, f'{nm}_tiled3x3.png'), s.tile(out, 3))
    meta.update(seam_in=round(float(sc_in), 3), seam_out=round(float(sc_out), 3),
                out_size=[out.shape[1], out.shape[0]])
    report[nm] = meta
    print(f'{nm:14s} method={meta["method"]:7s} grid={meta["is_grid"]!s:5s}  '
          f'{out.shape[1]}x{out.shape[0]}  seam {sc_in:.2f} -> {sc_out:.2f}')

    # монтаж для README: вход | seamless | 3x3
    def small(a, w=420):
        h = int(a.shape[0] * w / a.shape[1])
        return cv2.resize((np.clip(a, 0, 1) * 255).astype(np.uint8), (w, h))
    a, b, c = small(img), small(out), small(s.tile(out, 3))
    hh = max(a.shape[0], b.shape[0], c.shape[0])
    pad = lambda x: np.pad(x, ((0, hh - x.shape[0]), (0, 10), (0, 0)), constant_values=255)
    mont = np.concatenate([pad(a), pad(b), pad(c)], axis=1)
    cv2.imwrite(os.path.join(OUT, f'montage_{nm}.png'), cv2.cvtColor(mont, cv2.COLOR_RGB2BGR))

json.dump(report, open(os.path.join(OUT, 'report.json'), 'w'), indent=2, ensure_ascii=False)
print('\nseam_out — отношение градиента поперёк стыка к среднему внутри (1.0 ≈ шва не видно)')
