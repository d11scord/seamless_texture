#!/usr/bin/env python3
"""
CLI: закинул картинку -> получил seamless-текстуру. Без ручного редактирования.

  python cli.py texture.png                 # -> texture_seamless.png
  python cli.py texture.png -o out.png       # свой путь вывода
  python cli.py texture.png --tile 3         # ещё и превью раскладки 3x3
  python cli.py texture.png --method periodic # альтернативный метод (кроп по периодам)
"""
import argparse, os
import seamless as s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('-o', '--output')
    ap.add_argument('--method', choices=['auto', 'grid', 'mincut', 'lattice'], default='auto')
    ap.add_argument('--tile', type=int, default=0, help='сохранить превью NxN')
    args = ap.parse_args()

    img = s.load(args.input)
    out, meta = s.make_seamless(img, method=args.method)

    base, ext = os.path.splitext(args.output or args.input)
    if not args.output:
        base = base + '_seamless'
    out_path = base + (ext or '.png')
    s.save(out_path, out)

    sc = s.seam_score(out)
    print(f'{args.input} -> {out_path}')
    print(f'  method={meta['method']} grid={meta['is_grid']} '
          f'size={out.shape[1]}x{out.shape[0]} seam_score={sc:.3f}')

    if args.tile > 1:
        tpath = base + f'_tiled{args.tile}x{args.tile}' + (ext or '.png')
        s.save(tpath, s.tile(out, args.tile))
        print(f'  tiled preview -> {tpath}')


if __name__ == '__main__':
    main()
