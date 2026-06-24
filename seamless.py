"""
seamless.py — автоматическое создание бесшовных тайлящихся текстур.

Три специализированных метода + авто-выбор:
  • регулярная сетка (плитка)         -> grid-crop (стык в середине плитки,
                                          размер ячеек строго одинаковый);
  • диагональная решётка (ёлочка)     -> аффинная коррекция + period-crop
                                          (планки остаются целыми прямоугольниками);
  • остальное (доски случайной ширины)-> self-overlap min-cut + Laplacian-блендинг.

Общий препроцесс: выравнивание ЯРКОСТИ и ЦВЕТА (flat-field) — убирает виньетку и
тёплый/серый перекос, главные причины видимого шва и цветных расхождений.
"""
import numpy as np
import cv2


# ----------------------------- IO ----------------------------- #
def load(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def save(path, img):
    img8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img8, cv2.COLOR_RGB2BGR))


def luminance(img):
    return img @ np.array([0.299, 0.587, 0.114], np.float32)


# ---------------------- illumination flatten ------------------- #
def flatten_illumination(img, sigma_frac=0.10, strength=1.0, chroma=0.8):
    """Убирает крупномасштабный градиент ЯРКОСТИ (виньетка) и ЦВЕТА (тёплый/серый
    перекос по полю). Локальная разнотонность плашек сохраняется."""
    h, w = img.shape[:2]
    sigma = max(h, w) * sigma_frac
    lum = luminance(img)
    blur = cv2.GaussianBlur(lum, (0, 0), sigma)
    target = float(np.mean(blur))
    gain = target / np.clip(blur, 1e-3, None)
    gain = 1.0 + strength * (gain - 1.0)
    gain = np.clip(gain, 0.5, 2.0)[..., None]
    out = np.clip(img * gain, 0, 1)
    if chroma > 0:
        lab = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        for c in (1, 2):
            ch = lab[..., c]
            low = cv2.GaussianBlur(ch, (0, 0), sigma)
            lab[..., c] = ch - chroma * (low - low.mean())
        lab[..., 1:] = np.clip(lab[..., 1:], 0, 255)
        out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return np.clip(out, 0, 1)


# ----------------------- periodicity analysis ------------------ #
def _autocorr(sig):
    sig = sig - sig.mean()
    n = len(sig)
    f = np.fft.rfft(sig, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    return ac / (ac[0] + 1e-9)


def detect_period_1d(profile, max_frac=0.3, max_repeats=25):
    n = len(profile)
    min_p = max(40, int(0.035 * n))
    ac = _autocorr(profile)
    hi = int(n * max_frac)
    best_lag, best_val = None, -1.0
    for lag in range(min_p, hi - 1):
        reps = n / lag
        if reps < 2 or reps > max_repeats:
            continue
        if ac[lag] >= ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > best_val:
            best_val, best_lag = ac[lag], lag
    return best_lag, max(best_val, 0.05)


def analyze(img, already_flat=False):
    g = luminance(img if already_flat else flatten_illumination(img))
    hp = g - cv2.GaussianBlur(g, (0, 0), 2.0)
    prof_x = np.abs(hp).mean(axis=0)
    prof_y = np.abs(hp).mean(axis=1)
    px, sx = detect_period_1d(prof_x)
    py, sy = detect_period_1d(prof_y)
    return dict(px=px, py=py, sx=sx, sy=sy)


# --------------------- min-cut + multiband (доски) ------------- #
def min_cut_vertical(err):
    H, ov = err.shape
    cost = err.copy()
    back = np.zeros((H, ov), np.int32)
    for i in range(1, H):
        for j in range(ov):
            a, b = max(0, j - 1), min(ov, j + 2)
            k = a + int(np.argmin(cost[i - 1, a:b]))
            back[i, j] = k
            cost[i, j] += cost[i - 1, k]
    seam = np.zeros(H, np.int32)
    seam[-1] = int(np.argmin(cost[-1]))
    for i in range(H - 2, -1, -1):
        seam[i] = back[i + 1, seam[i + 1]]
    return seam


def _lap_blend(A, B, mask, levels=6):
    """Laplacian-блендинг: низкие частоты (цвет/яркость) сшиваются широко, высокие
    (кромки/волокно) — узко по маске. Убирает цветной скачок без смазывания."""
    H, W = A.shape[:2]
    lv = max(1, min(levels, int(np.log2(max(2, min(H, W)))) - 1))
    GA, GB, GM = [A.copy()], [B.copy()], [mask.copy()]
    for _ in range(lv):
        GA.append(cv2.pyrDown(GA[-1])); GB.append(cv2.pyrDown(GB[-1]))
        GM.append(cv2.pyrDown(GM[-1]))
    out = GA[-1] * (1 - GM[-1][..., None]) + GB[-1] * GM[-1][..., None]
    for i in range(lv - 1, -1, -1):
        sz = (GA[i].shape[1], GA[i].shape[0])
        LA = GA[i] - cv2.pyrUp(GA[i + 1], dstsize=sz)
        LB = GB[i] - cv2.pyrUp(GB[i + 1], dstsize=sz)
        m = GM[i][..., None]
        out = cv2.pyrUp(out, dstsize=sz) + LA * (1 - m) + LB * m
    return out


def _blend_overlap_h(A, B, feather=3):
    H, ov = A.shape[:2]
    err = ((A - B) ** 2).sum(2)
    seam = min_cut_vertical(err)
    xs = np.arange(ov)[None, :]
    alpha = np.clip((xs - seam[:, None] + feather) / (2 * feather + 1e-9), 0, 1)
    return np.clip(_lap_blend(A, B, alpha.astype(np.float32)), 0, 1)


def make_tileable_overlap(img, ov_frac=0.1, feather=3):
    """Self-overlap min-cut + Laplacian: шов идёт по кромкам досок, цвет сшивается
    широко. Размер уменьшается на ov по каждой оси."""
    H, W = img.shape[:2]
    ov = max(8, int(min(H, W) * ov_frac))
    A, B = img[:, W - ov:W], img[:, 0:ov]
    out = img[:, :W - ov].copy()
    out[:, :ov] = _blend_overlap_h(A, B, feather)
    Ht = out.shape[0]
    t = np.transpose(out, (1, 0, 2))
    A, B = t[:, Ht - ov:Ht], t[:, 0:ov]
    t2 = t[:, :Ht - ov].copy()
    t2[:, :ov] = _blend_overlap_h(A, B, feather)
    return np.clip(np.transpose(t2, (1, 0, 2)), 0, 1)


# --------------- periodic lattice (паркет / ёлочка) ------------ #
def _edge_map(flat):
    g = luminance(flat)
    return cv2.GaussianBlur(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)) +
                            np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)), (0, 0), 2.0)


def _period_axis(E, axis, ds):
    """Период по оси = лаг максимума нормированной АКФ карты кромок (геометрия,
    не зависит от текстуры дерева). Возвращает (период_в_полном_разрешении, corr)."""
    n = E.shape[1] if axis == 1 else E.shape[0]
    pmin = max(15, int(n * 0.05))
    best_v, best_p = -1.0, None
    for P in range(pmin, int(n * 0.5)):
        a, b = (E[:, :-P], E[:, P:]) if axis == 1 else (E[:-P], E[P:])
        a = a - a.mean(); b = b - b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        v = float((a * b).sum() / d) if d else 0.0
        if v > best_v:
            best_v, best_p = v, P * ds
    return best_p, best_v


def _corr_axis(E, P, axis):
    a, b = (E[:, :-P], E[:, P:]) if axis == 1 else (E[:-P], E[P:])
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / d) if d else 0.0


def _refine_subpixel(E, axis, p0, win=6):
    """Уточняет период до субпикселя параболой по пику корреляции (полное разрешение)."""
    vs = {P: _corr_axis(E, P, axis) for P in range(max(8, p0 - win), p0 + win + 1)}
    p = max(vs, key=vs.get)
    a, b, c = vs.get(p - 1, vs[p]), vs[p], vs.get(p + 1, vs[p])
    den = a - 2 * b + c
    return p + (0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0)


def detect_lattice(flat, thresh=0.40, ds=None):
    """Сильная геометрическая решётка (ёлочка/паркет): крупный период с высокой
    корреляцией карты кромок на ОБЕИХ осях. Период уточняется до СУБПИКСЕЛЯ —
    иначе ошибка округления копится по периодам и планки «едут».
    ds — даунсемпл для грубого поиска (меньше = точнее/медленнее)."""
    H, W = flat.shape[:2]
    if ds is None:
        ds = max(1, int(round(max(H, W) / 600)))
    e = _edge_map(flat)
    E = cv2.resize(e, (W // ds, H // ds), interpolation=cv2.INTER_AREA)
    px, cx = _period_axis(E, 1, ds)
    py, cy = _period_axis(E, 0, ds)
    if not (px and py and cx >= thresh and cy >= thresh):
        return None
    # субпиксельное уточнение на полном разрешении
    pxf = _refine_subpixel(e, 1, int(round(px)), win=max(3, ds + 2))
    pyf = _refine_subpixel(e, 0, int(round(py)), win=max(3, ds + 2))
    return float(pxf), float(pyf)


def _period_in_window(e, axis, p0, win, lo, hi):
    seg = e[lo:hi] if axis == 0 else e[:, lo:hi]
    vs = {P: _corr_axis(seg, P, axis) for P in range(max(8, p0 - win), p0 + win)}
    p = max(vs, key=vs.get)
    a, b, c = vs.get(p - 1, vs[p]), vs[p], vs.get(p + 1, vs[p])
    den = a - 2 * b + c
    return p + (0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0)


def _rectify_axis(img, axis, p0, win, tol=0.4):
    """Выпрямляет лёгкую перспективу: если шаг планки линейно «едет» вдоль оси,
    ресемплит так, чтобы период стал постоянным. Если уже ровно — возвращает как есть."""
    e = _edge_map(flatten_illumination(img))
    H, W = img.shape[:2]
    n = H if axis == 0 else W
    L = n // 2
    cs = np.array([n * 0.25, n * 0.5, n * 0.75])
    pe = np.array([_period_in_window(e, axis, p0, win, int(c - L // 2), int(c + L // 2))
                   for c in cs])
    if pe.max() - pe.min() < tol:          # перспективы нет — не трогаем
        return img
    m, b = np.linalg.lstsq(np.vstack([cs, np.ones(3)]).T, pe, rcond=None)[0]
    y = np.arange(n)
    per = np.clip(m * y + b, p0 * 0.5, p0 * 1.5)
    Phi = np.concatenate([[0], np.cumsum(1.0 / per)])[:n]
    P = float(per.mean())
    nout = int(round(Phi[-1] * P))
    src = np.interp(np.arange(nout) / P, Phi, y).astype(np.float32)
    if axis == 0:
        mx, my = np.meshgrid(np.arange(W, dtype=np.float32), src)
    else:
        mx, my = np.meshgrid(src, np.arange(H, dtype=np.float32))
    return cv2.remap(img, mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)


def rectify_perspective(flat, px, py):
    """Выпрямляет перспективу по обеим осям (для решётчатых текстур). Без неё на
    фото с лёгким наклоном планки не стыкуются: шаг едет вдоль кадра."""
    r = _rectify_axis(flat, 0, int(round(py)), win=max(12, int(py * 0.1)))
    r = _rectify_axis(r, 1, int(round(px)), win=max(12, int(px * 0.08)))
    return r


def _overlap_blend_axis(img, ov, axis):
    """Сшивка с перекрытием ровно ov px. Для периодической картинки ov = период:
    края геометрически совпадают, поэтому min-cut+multiband примиряет ТОЛЬКО тон,
    не ломая форму планок."""
    if axis == 0:
        img = np.transpose(img, (1, 0, 2))
    W = img.shape[1]
    ov = max(8, min(ov, W // 2))
    A, B = img[:, W - ov:W], img[:, 0:ov]
    out = img[:, :W - ov].copy()
    out[:, :ov] = _blend_overlap_h(A, B)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))
    return out


def make_tileable_lattice(flat, px, py, max_periods=10):
    """Кроп до целого числа периодов + РЕСЕМПЛ до ровно целого периода, поэтому
    тайл строго периодичен и не «едет» при любом числе повторов. Затем сшивка с
    перекрытием в один период (тон примирён, форма планок цела).
    max_periods — ограничение числа периодов в тайле (меньше = меньше остаточной
    неравномерности от перспективы, но больше повтор)."""
    H, W = flat.shape[:2]
    Px, Py = int(round(px)), int(round(py))
    nx = max(2, int(W / px))
    ny = max(2, int(H / py))
    if max_periods:
        nx, ny = min(nx, max_periods), min(ny, max_periods)
    # точный субпиксельный охват N периодов -> ресемпл в ровно N*целый_период
    spanx, spany = int(round(nx * px)), int(round(ny * py))
    crop = flat[:spany, :spanx]
    if (spanx, spany) != (nx * Px, ny * Py):
        crop = cv2.resize(crop, (nx * Px, ny * Py), interpolation=cv2.INTER_LANCZOS4)
    out = _overlap_blend_axis(crop, Px, 1)
    out = _overlap_blend_axis(out, Py, 0)
    return np.clip(out, 0, 1)


# --------------- regular grid (плитка) ------------------------- #
def _grout_lines(g, axis, period):
    sob = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=5)) if axis == 1 \
        else np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5))
    prof = (sob.mean(0) if axis == 1 else sob.mean(1)).astype(np.float32)
    prof = cv2.GaussianBlur(prof[None, :], (0, 0), 5).ravel()
    from scipy.signal import find_peaks
    pk, _ = find_peaks(prof, distance=int(period * 0.6), prominence=prof.std() * 0.25)
    return pk


def detect_grid(flat, info):
    """Регулярная сетка: ровные линии затирки на ОБЕИХ осях."""
    if not info['px'] or not info['py']:
        return None
    g = luminance(flat)
    xs = _grout_lines(g, 1, info['px'])
    ys = _grout_lines(g, 0, info['py'])
    for pk in (xs, ys):
        if len(pk) < 3:
            return None
        sp = np.diff(pk)
        if sp.std() / (sp.mean() + 1e-9) > 0.08:
            return None
    return xs, ys


def grid_crop(flat, xs, ys):
    """Кроп так, чтобы стык тайлов попадал в СЕРЕДИНУ плитки (на ровную поверхность),
    а не на линию затирки -> кресты затирки внутренние и ровные, ячейки одинаковы."""
    H, W = flat.shape[:2]
    cw = int(np.median(np.diff(xs)))
    ch = int(np.median(np.diff(ys)))
    x0 = int(xs[0]) - cw // 2
    y0 = int(ys[0]) - ch // 2
    if x0 < 0:
        x0 += cw
    if y0 < 0:
        y0 += ch
    nx = max(1, (W - x0) // cw)
    ny = max(1, (H - y0) // ch)
    return np.clip(flat[y0:y0 + ny * ch, x0:x0 + nx * cw], 0, 1)


# --------------------------- dispatch -------------------------- #
def make_seamless(img, method='auto', rectify=True):
    """Авто-выбор метода. method: 'auto' | 'grid' | 'lattice' | 'mincut'.
    rectify=True выпрямляет лёгкую перспективу для решётчатых текстур (ёлочка),
    иначе планки не стыкуются: шаг едет вдоль кадра."""
    flat = flatten_illumination(img)
    info = analyze(flat, already_flat=True)
    info = {k: (int(v) if v is not None and k in ('px', 'py') else float(v))
            for k, v in info.items()}

    grid = detect_grid(flat, info) if method in ('auto', 'grid') else None
    lat = detect_lattice(flat) if (method in ('auto', 'lattice') and grid is None) else None

    if method == 'grid' or (method == 'auto' and grid is not None):
        out = grid_crop(flat, *grid) if grid else make_tileable_overlap(flat)
        used = 'grid'
    elif method == 'lattice' or (method == 'auto' and lat is not None):
        if lat is None:
            lat = detect_lattice(flat, thresh=0.0)   # форс: берём лучший период
        if rectify:
            flat = flatten_illumination(rectify_perspective(flat, *lat))
            lat = detect_lattice(flat, thresh=0.0)   # период после выпрямления
        out = make_tileable_lattice(flat, *lat)
        used = 'lattice'
    else:
        out = make_tileable_overlap(flat)
        used = 'mincut'
    return out, dict(method=used, is_grid=bool(grid is not None),
                     is_lattice=bool(lat is not None), **info)


# --------------------------- utilities ------------------------- #
def tile(img, n=3):
    return np.tile(img, (n, n, 1))


def seam_score(img):
    """Метрика бесшовности: градиент поперёк стыка / средний внутри. ~1.0 = ок."""
    t = tile(img, 2)
    H, W = img.shape[:2]
    g = luminance(t)
    gy = np.abs(np.diff(g, axis=0))
    gx = np.abs(np.diff(g, axis=1))
    interior = 0.5 * (gx.mean() + gy.mean())
    seam = 0.5 * (gx[:, W - 1].mean() + gy[H - 1, :].mean())
    return seam / (interior + 1e-9)
