import sys, os, math, random, subprocess
import imageio_ffmpeg
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile
from scipy.signal import fftconvolve

W, H, FPS, DUR = 1080, 1920, 30, 22.5
BPM = 168
BEAT = 60 / BPM
BAR = 4 * BEAT
TC = 5.36                 # inici del compte enrere
T0 = TC + 4 * BEAT        # primer temps de la nadala
TF = T0 + 7 * BAR         # darrer compas (acord final)
TEND = T0 + 8 * BAR
AA = cv2.LINE_AA
random.seed(11)
np.random.seed(11)


def hexc(h):
    h = h.lstrip('#')
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


GOLD, GOLD2, SILVER = hexc('#f2b51d'), hexc('#b8820a'), hexc('#d9dde3')
DARK, WHITE, RED = hexc('#2b2140'), hexc('#ffffff'), hexc('#d62828')
clamp01 = lambda v: max(0.0, min(1.0, v))

# ---------------------------------------------------------------- musics
M = [
    dict(k='sax', x=250, y=1170, s=.78, mir=1, jk='#2a9d8f', skin='#f3c9a5', hair='#5a3a22'),
    dict(k='heli', x=540, y=1170, s=.78, mir=1, jk='#e76f51', skin='#e0a97d', hair='#222222', beard=1),
    dict(k='sax', x=830, y=1170, s=.78, mir=-1, jk='#7b2cbf', skin='#c68a5e', hair='#111111', glasses=1),
    dict(k='tbn', x=190, y=1335, s=.88, mir=1, jk='#3a86ff', skin='#f0bf9a', hair='#c9a227', stache=1),
    dict(k='bombo', x=540, y=1335, s=.88, mir=1, jk='#ef476f', skin='#8d5a3b', hair='#111111'),
    dict(k='tbn', x=890, y=1335, s=.88, mir=-1, jk='#06a77d', skin='#f3c9a5', hair='#8a8a8a', glasses=1),
    dict(k='tpt', x=220, y=1500, s=1.0, mir=1, jk='#f77f00', skin='#e0a97d', hair='#3b2a1a'),
    dict(k='snare', x=540, y=1500, s=1.0, mir=1, jk='#4361ee', skin='#f3c9a5', hair='#a0522d', stache=1),
    dict(k='tpt', x=860, y=1500, s=1.0, mir=-1, jk='#d62f8c', skin='#c68a5e', hair='#222222'),
]
ORDER = [1, 4, 3, 0, 5, 2, 6, 8, 7]
LAND = {m: 1.4 + 0.45 * k for k, m in enumerate(ORDER)}
ENTRY = [T0 + 2 * BAR, T0 + 4 * BAR, T0 + 2 * BAR, T0 + 4 * BAR, T0, T0 + 4 * BAR, T0, TC, T0]


def rrpts(u1, v1, u2, v2, r, n=5):
    pts = []
    for (cx_, cy_, a0) in [(u2 - r, v1 + r, -90), (u2 - r, v2 - r, 0), (u1 + r, v2 - r, 90), (u1 + r, v1 + r, 180)]:
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append((cx_ + r * math.cos(a), cy_ + r * math.sin(a)))
    return pts


def acirc(img, x, y, r, col, a):
    x0, y0 = max(0, int(x - r - 2)), max(0, int(y - r - 2))
    x1, y1 = min(W, int(x + r + 3)), min(H, int(y + r + 3))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    ov = roi.copy()
    cv2.circle(ov, (int(x - x0), int(y - y0)), int(r), col, -1, AA)
    cv2.addWeighted(ov, a, roi, 1 - a, 0, roi)


def render_m(img, i, t):
    d = M[i]
    k = d['k']
    L = LAND[i]
    F = L - 1.0
    if t < F:
        return
    x, y, s, mir = d['x'], d['y'], d['s'], d['mir']
    jk, skin, hair = hexc(d['jk']), hexc(d['skin']), hexc(d['hair'])
    skind = tuple(int(c * .85) for c in skin)
    qx = qy = 1.0
    rot = 0.0
    falling = t < L
    since = t - L
    if falling:
        u = (t - F) / 1.0
        y -= 1500 * (1 - u * u)
        rot = 0.45 * math.sin(u * 8 + i) * (1 - u * .5)
    else:
        q = math.exp(-since * 7) * math.cos(since * 26)
        qy, qx = 1 - 0.28 * q, 1 + 0.2 * q
    tb = t - T0
    amp = clamp01((t - ENTRY[i]) / 0.2) if t > ENTRY[i] else 0.0
    if t > TEND:
        amp = 0.0
    counting = (i == 7 and TC <= t < T0)
    if counting:
        amp = 1.0
        ph, kk = ((t - TC) / BEAT) % 1, int((t - TC) / BEAT)
    elif tb >= 0:
        ph, kk = (tb / BEAT) % 1, int(tb // BEAT)
    else:
        ph, kk = 0.0, -1
    bounce = 0.0
    if not falling:
        if amp > 0:
            bounce = amp * 14 * abs(math.sin(math.pi * ph))
            rot += 0.07 * amp * math.sin(math.pi * (t - (TC if counting else T0)) / BEAT)
        elif t > TEND:
            bounce = 8 * abs(math.sin(t * 6 + i))
        else:
            rot += 0.02 * math.sin(t * 2 + i)
        if t >= TF and t < TF + 1.2:
            bounce += 40 * (1 - (t - TF) / 1.2) * abs(math.sin((t - TF) * 10))
    y -= bounce
    cr, sr = math.cos(rot), math.sin(rot)

    def P(u, v):
        u = u * mir
        return (int(round(x + (u * cr - v * sr) * s * qx)), int(round(y + (u * sr + v * cr) * s * qy)))

    Rr = lambda r: max(1, int(round(r * s * (qx + qy) / 2)))

    def circ(u, v, r, c):
        cv2.circle(img, P(u, v), Rr(r), c, -1, AA)

    def ell(u, v, a, b, ang, c, th=-1):
        if th > 0:
            th = max(1, int(th * s))
        cv2.ellipse(img, P(u, v), (max(1, int(a * s * qx)), max(1, int(b * s * qy))),
                    math.degrees(rot) + ang * mir, 0, 360, c, th, AA)

    def poly(pts, c):
        cv2.fillPoly(img, [np.array([P(u, v) for u, v in pts], np.int32)], c, AA)

    def line(p, q, w, c):
        a, b = P(*p), P(*q)
        th = max(1, int(w * s * (qx + qy) / 2))
        cv2.line(img, a, b, c, th, AA)
        cv2.circle(img, a, th // 2, c, -1, AA)
        cv2.circle(img, b, th // 2, c, -1, AA)

    def rr(u1, v1, u2, v2, r, c):
        poly(rrpts(u1, v1, u2, v2, r), c)

    wood = hexc('#c28a4a')
    # ---- parts that go behind the body
    hands = [(-80, -150), (80, -150)]
    off = 0
    if k == 'heli':
        poly([(55, -240), (95, -240), (160, -440), (25, -440)], GOLD)
        ell(92, -442, 70, 20, 0, GOLD)
        ell(92, -442, 56, 13, 0, GOLD2)
    if k == 'tbn':
        poly([(-40, -275), (-125, -335), (-125, -235), (-40, -250)], GOLD)
        ell(-125, -285, 12, 50, 0, GOLD2)
    # ---- legs, body
    pants = hexc('#2b2140')
    rr(-34, -72, -8, -14, 10, pants)
    rr(8, -72, 34, -14, 10, pants)
    ell(-24, -10, 30, 14, 0, (20, 15, 25))
    ell(24, -10, 30, 14, 0, (20, 15, 25))
    rr(-54, -215, 54, -62, 28, jk)
    line((0, -205), (0, -75), 9, tuple(min(255, int(c * 1.25 + 30)) for c in jk))
    circ(0, -170, 6, GOLD)
    circ(0, -130, 6, GOLD)
    # ---- head
    circ(-55, -285, 26, hair)
    circ(55, -285, 26, hair)
    circ(-60, -266, 14, skin)
    circ(60, -266, 14, skin)
    circ(0, -270, 62, skin)
    circ(-38, -250, 12, (150, 140, 235))
    circ(38, -250, 12, (150, 140, 235))
    circ(0, -258, 9, skind)
    if d.get('beard'):
        ell(0, -226, 50, 34, 0, hair)
    blink = (t * .9 + i * .37) % 3.1 < 0.12
    lx = 3 * math.sin(t * 1.3 + i)
    for sx_ in (-1, 1):
        if blink:
            line((sx_ * 24 - 12, -278), (sx_ * 24 + 12, -278), 5, DARK)
        else:
            ell(sx_ * 24, -278, 15, 17, 0, WHITE)
            circ(sx_ * 24 + lx, -276, 7, DARK)
            circ(sx_ * 24 + lx + 2, -280, 2, WHITE)
        by = -312 if falling else -304
        line((sx_ * 24 - 13, by), (sx_ * 24 + 13, by - 2 * sx_), 6, hair)
    if d.get('glasses'):
        for sx_ in (-1, 1):
            ell(sx_ * 24, -278, 28, 28, 0, DARK, 6)
        line((-8, -280), (8, -280), 5, DARK)
    if d.get('stache'):
        ell(-17, -246, 22, 10, 20, hair)
        ell(17, -246, 22, 10, -20, hair)
    wind = k in ('sax', 'heli', 'tbn', 'tpt')
    if falling:
        ell(0, -230, 16, 22, 0, DARK)
    elif wind and amp > 0:
        circ(-46, -244, 20, skin)
        circ(46, -244, 20, skin)
        circ(-46, -244, 10, (150, 140, 235))
        circ(46, -244, 10, (150, 140, 235))
    elif amp > 0:
        ell(0, -236, 24, 15, 0, DARK)
        ell(0, -230, 12, 7, 0, (90, 90, 220))
    else:
        pts = [(-20 + 40 * j / 8, -238 + 10 * math.sin(math.pi * j / 8)) for j in range(9)]
        for j in range(8):
            line(pts[j], pts[j + 1], 5, DARK)
    # ---- Santa hat
    poly([(-60, -308), (60, -308), (36, -405)], RED)
    poly([(36, -405), (50, -390), (74, -372), (80, -392)], RED)
    circ(80, -392, 17, WHITE)
    rr(-68, -324, 68, -296, 14, WHITE)
    # ---- instrument (front)
    ph_s = 2 * math.pi * (t - T0) / (2 * BEAT)
    if k == 'tpt':
        poly([(120, -262), (190, -300), (190, -196), (120, -230)], GOLD)
        ell(190, -248, 12, 52, 0, GOLD2)
        rr(18, -255, 125, -237, 8, GOLD)
        for vx in (55, 75, 95):
            rr(vx - 5, -288, vx + 5, -255, 4, SILVER)
            rr(vx - 9, -296, vx + 9, -286, 4, GOLD2)
        ell(95, -216, 46, 22, 0, GOLD, 9)
        circ(20, -247, 10, SILVER)
        hands = [(75, -262), (122, -214)]
    elif k == 'sax':
        pts = [(24, -248), (50, -205), (66, -140), (90, -96), (128, -92), (152, -125)]
        for j in range(len(pts) - 1):
            line(pts[j], pts[j + 1], 22, GOLD)
        poly([(136, -120), (188, -134), (176, -196), (130, -160)], GOLD)
        ell(160, -170, 22, 12, -60, GOLD2)
        for kx, ky in [(58, -165), (70, -135), (92, -112)]:
            circ(kx, ky, 8, SILVER)
        circ(22, -246, 9, SILVER)
        hands = [(46, -196), (64, -150)]
    elif k == 'tbn':
        off = 48 * amp * (0.5 + 0.5 * math.sin(ph_s + i))
        rr(20, -262, 218 + off, -250, 6, GOLD)
        rr(20, -232, 218 + off, -220, 6, GOLD)
        rr(206 + off, -262, 222 + off, -220, 7, GOLD)
        line((-40, -262), (40, -256), 12, GOLD)
        circ(20, -246, 10, SILVER)
        hands = [(75, -241), (206 + off, -241)]
    elif k == 'heli':
        ell(0, -150, 102, 110, 0, GOLD, 24)
        line((20, -246), (72, -246), 14, GOLD)
        circ(20, -246, 10, SILVER)
        hands = [(-96, -150), (96, -150)]
    elif k == 'bombo':
        circ(0, -140, 96, hexc('#f4f4f4'))
        ell(0, -140, 96, 96, 0, hexc('#d62828'), 16)
        line((-32, -172), (32, -108), 12, hexc('#d62828'))
        line((-32, -108), (32, -172), 12, hexc('#d62828'))
        hit = amp > 0 and kk % 2 == 0
        tipv = -140 - 120 * min(1, ph * 3) if hit else -265
        if amp == 0:
            tipv = -250
        line((84, -190), (50, tipv), 11, wood)
        circ(50, tipv, 17, hexc('#8d5a3b'))
        line((-84, -190), (-56, -250), 11, wood)
        circ(-56, -250, 17, hexc('#8d5a3b'))
        hands = [(-84, -190), (84, -190)]
    elif k == 'snare':
        rr(-68, -125, 68, -80, 10, hexc('#d62828'))
        ell(0, -125, 68, 22, 0, hexc('#f5f5f5'))
        ell(0, -125, 68, 22, 0, SILVER, 6)
        line((-42, -208), (-58, -120), 8, (35, 30, 35))
        line((42, -208), (58, -120), 8, (35, 30, 35))
        hitR = amp > 0 and (counting or kk % 2 == 1)
        hitL = amp > 0 and (not counting) and kk % 2 == 0
        tr = -130 - 65 * min(1, ph * 3) if hitR else -205
        tl = -130 - 65 * min(1, ph * 3) if hitL else -205
        line((66, -172), (32, tr), 9, wood)
        circ(32, tr, 10, WHITE)
        line((-66, -172), (-32, tl), 9, wood)
        circ(-32, tl, 10, WHITE)
        hands = [(-66, -172), (66, -172)]
    # ---- arms
    if falling:
        hands = [(-92, -312), (92, -312)]
    for (hu, hv), sh in zip(hands, (-1, 1)):
        line((sh * 46, -195), (hu, hv), 32, jk)
        circ(hu, hv, 17, skin)
    # ---- dizzy stars and dust
    if not falling and since < 1.0:
        for j in range(3):
            a = since * 9 + j * 2.09
            circ(70 * math.cos(a), -345 + 16 * math.sin(a), 9, hexc('#ffd166'))
        if since < 0.5:
            for j in range(6):
                acirc(img, x + (j - 2.5) * since * 190, y - 10, 14 + since * 60, (245, 235, 235), .5 * (1 - since / .5))


# ---------------------------------------------------------------- fons
def star_poly(cx_, cy_, ro, ri):
    pts = []
    for j in range(10):
        r = ro if j % 2 == 0 else ri
        a = -math.pi / 2 + j * math.pi / 5
        pts.append((int(cx_ + r * math.cos(a)), int(cy_ + r * math.sin(a))))
    return np.array(pts, np.int32)


TREE_X = 130
TIERS = [(700, 880, 95), (800, 1010, 130), (920, 1150, 165)]
TREE_LIGHTS = []
for (ay, by, hw) in TIERS:
    for fy in (0.4, 0.8):
        yy = ay + fy * (by - ay)
        half = hw * fy * 0.85
        for fx in (-1, -0.33, 0.33, 1):
            TREE_LIGHTS.append((TREE_X + fx * half, yy))


def garland_y(xx):
    return 200 + 55 * math.sin(math.pi * (xx % 360) / 360)


def build_bg():
    img = np.zeros((H, W, 3), np.uint8)
    top, mid, bot = np.array(hexc('#080d2e'), float), np.array(hexc('#2b1a63'), float), np.array(hexc('#b5467f'), float)
    for yy in range(0, 1010):
        u = yy / 1010
        c = top + (mid - top) * (u / .6) if u < .6 else mid + (bot - mid) * ((u - .6) / .4)
        img[yy, :] = c.astype(np.uint8)
    for yy in range(1010, H):
        u = clamp01((yy - 1010) / 900)
        img[yy, :] = (np.array(hexc('#e9f0ff'), float) * (1 - u) + np.array(hexc('#aebfec'), float) * u).astype(np.uint8)
    # mar
    for yy in range(960, 1090):
        u = (yy - 960) / 130
        img[yy, :] = (np.array(hexc('#17407a'), float) * (1 - u) + np.array(hexc('#0a1d3d'), float) * u).astype(np.uint8)
    for _ in range(120):
        cv2.circle(img, (random.randint(0, W), random.randint(20, 900)), random.choice([1, 1, 2, 3]),
                   random.choice([(255, 255, 255), (220, 240, 255), (200, 230, 255)]), -1, AA)
    # lluna
    for r, a in [(170, .10), (120, .14)]:
        acirc(img, 860, 470, r, hexc('#fff0b0'), a)
    cv2.circle(img, (860, 470), 72, hexc('#fff4c8'), -1, AA)
    for (dx, dy, r) in [(-22, -14, 15), (18, 20, 11), (24, -26, 9)]:
        cv2.circle(img, (860 + dx, 470 + dy), r, hexc('#ecdca0'), -1, AA)
    for j in range(9):
        cv2.line(img, (860 - 36 + j * 3 % 20, 985 + j * 13), (860 + 36 - j * 3 % 20, 985 + j * 13), hexc('#c9b97a'), 4, AA)
    # barques
    for (bx, by, sc) in [(210, 1000, 1.0), (430, 1025, .8), (690, 1005, .9)]:
        cv2.fillPoly(img, [np.array([(bx - 40 * sc, by), (bx + 40 * sc, by), (bx + 28 * sc, by + 16 * sc), (bx - 28 * sc, by + 16 * sc)], np.int32)], hexc('#f4e9d8'), AA)
        cv2.line(img, (bx, int(by)), (bx, int(by - 62 * sc)), hexc('#f4e9d8'), 3, AA)
        cv2.fillPoly(img, [np.array([(bx + 4, by - 60 * sc), (bx + 4, by - 6 * sc), (bx + 38 * sc, by - 8 * sc)], np.int32)], hexc('#ffd166'), AA)
        cv2.fillPoly(img, [np.array([(bx - 4, by - 56 * sc), (bx - 4, by - 8 * sc), (bx - 30 * sc, by - 10 * sc)], np.int32)], hexc('#ef476f'), AA)
    # neu del terra amb turons
    for cxp in range(-100, W + 300, 250):
        cv2.ellipse(img, (cxp, 1088), (260, 52), 0, 0, 360, hexc('#eef3ff'), -1, AA)
    # arbre
    cv2.rectangle(img, (TREE_X - 24, 1150), (TREE_X + 24, 1235), hexc('#7a4e24'), -1)
    for (ay, by, hw) in TIERS:
        cv2.fillPoly(img, [np.array([(TREE_X - hw, by), (TREE_X + hw, by), (TREE_X, ay)], np.int32)], hexc('#1f8a42'), AA)
        cv2.fillPoly(img, [np.array([(TREE_X - hw, by), (TREE_X - hw * .35, by), (TREE_X, ay)], np.int32)], hexc('#16703a'), AA)
        sy = ay + .3 * (by - ay)
        cv2.fillPoly(img, [np.array([(TREE_X, ay), (TREE_X - hw * .3, sy), (TREE_X + hw * .3, sy)], np.int32)], hexc('#f2f6ff'), AA)
    for (ox, oy, c) in [(100, 860, '#ef476f'), (160, 880, '#ffd166'), (90, 990, '#4cc9f0'), (170, 1000, '#ef476f'),
                        (70, 1120, '#ffd166'), (130, 1100, '#7b2cbf'), (200, 1130, '#4cc9f0')]:
        cv2.circle(img, (ox, oy), 12, hexc(c), -1, AA)
    cv2.fillPoly(img, [star_poly(TREE_X, 686, 38, 16)], hexc('#ffd54a'), AA)
    # ninot de neu
    sx0 = 985
    for (cy_, r) in [(1170, 78), (1052, 58), (962, 44)]:
        cv2.circle(img, (sx0, cy_), r, hexc('#f4f8ff'), -1, AA)
    cv2.line(img, (sx0 - 50, 1050), (sx0 - 110, 1000), hexc('#6b4423'), 6, AA)
    cv2.line(img, (sx0 + 50, 1050), (sx0 + 108, 1005), hexc('#6b4423'), 6, AA)
    cv2.ellipse(img, (sx0, 1010), (50, 12), 0, 0, 360, hexc('#d62828'), -1, AA)
    cv2.fillPoly(img, [np.array([(sx0, 962), (sx0 + 50, 972), (sx0, 978)], np.int32)], hexc('#f77f00'), AA)
    for ex in (-16, 16):
        cv2.circle(img, (sx0 + ex, 950), 5, (30, 20, 30), -1, AA)
    for by_ in (1040, 1075, 1110):
        cv2.circle(img, (sx0, by_), 6, (30, 20, 30), -1, AA)
    cv2.rectangle(img, (sx0 - 34, 888), (sx0 + 34, 928), (30, 20, 40), -1, AA)
    cv2.rectangle(img, (sx0 - 52, 926), (sx0 + 52, 936), (30, 20, 40), -1, AA)
    # regals
    for (gx, gy, c, rc) in [(140, 1700, '#d62828', '#ffd166'), (230, 1740, '#2a9d8f', '#ffffff'), (930, 1710, '#7b2cbf', '#ffd166'), (845, 1750, '#f77f00', '#ffffff')]:
        cv2.rectangle(img, (gx - 50, gy - 45), (gx + 50, gy + 45), hexc(c), -1, AA)
        cv2.rectangle(img, (gx - 8, gy - 45), (gx + 8, gy + 45), hexc(rc), -1, AA)
        cv2.rectangle(img, (gx - 50, gy - 8), (gx + 50, gy + 8), hexc(rc), -1, AA)
        cv2.circle(img, (gx - 14, gy - 58), 14, hexc(rc), 5, AA)
        cv2.circle(img, (gx + 14, gy - 58), 14, hexc(rc), 5, AA)
    # garlanda
    pts = np.array([(xx, garland_y(xx)) for xx in range(0, W + 6, 6)], np.int32)
    cv2.polylines(img, [pts], False, hexc('#0f5a2c'), 7, AA)
    return img


BG = build_bg()
BULBS = [(30 + 60 * j, garland_y(30 + 60 * j) + 16) for j in range(18)]
BULB_COL = [hexc(c) for c in ('#ef476f', '#ffd166', '#4cc9f0', '#7bf1a8')]
SNOW = [(random.random() * W, random.random() * (H + 40), 3 + random.random() * 5, 80 + random.random() * 140, random.random() * 6.28) for _ in range(150)]
CONF = [(random.random() * W, -random.random() * 700, 9 + random.random() * 9, 6 + random.random() * 6, 5 + random.random() * 8, random.random() * 6.28, random.random() * 6) for _ in range(150)]
CONF_COL = [hexc(c) for c in ('#ef476f', '#ffd166', '#4cc9f0', '#7bf1a8', '#ffffff', '#c77dff')]

# ---------------------------------------------------------------- text
FONT = None
for f_path in [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", # Linux
    "C:/Windows/Fonts/arialbd.ttf",                         # Windows
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",    # Mac
    "/Library/Fonts/Arial Bold.ttf"                         # Mac (alternativa)
]:
    if os.path.exists(f_path):
        FONT = f_path
        break


def text_rgba(txt, size, fill, stroke, sw):
    f = ImageFont.truetype(FONT, size) if FONT else ImageFont.load_default()
    bb = ImageDraw.Draw(Image.new("RGBA", (10, 10))).textbbox((0, 0), txt, font=f, stroke_width=sw)
    w_, h_ = bb[2] - bb[0] + 2 * sw + 8, bb[3] - bb[1] + 2 * sw + 8
    im = Image.new("RGBA", (w_, h_), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((sw + 4 - bb[0], sw + 4 - bb[1]), txt, font=f, fill=fill, stroke_width=sw, stroke_fill=stroke)
    return np.array(im)


def blit(img, a, cx_, cy_, alpha=1.0, scale=1.0):
    if alpha <= 0:
        return
    if scale != 1.0:
        a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h_, w_ = a.shape[:2]
    x0, y0 = int(cx_ - w_ / 2), int(cy_ - h_ / 2)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w_), min(H, y0 + h_)
    if xb <= xa or yb <= ya:
        return
    src = a[ya - y0:yb - y0, xa - x0:xb - x0]
    al = src[:, :, 3:4].astype(np.float32) / 255 * alpha
    roi = img[ya:yb, xa:xb]
    roi[:] = (src[:, :, [2, 1, 0]] * al + roi * (1 - al)).astype(np.uint8)


STROKE = (58, 26, 110)
T_X1 = text_rgba("XARANGA", 118, (255, 209, 102), STROKE, 11)
T_X2 = text_rgba("NADALENCA", 118, (255, 255, 255), STROKE, 11)
T_NUM = [text_rgba(str(n), 300, (255, 209, 102), (122, 31, 162), 14) for n in (1, 2, 3, 4)]
T_BON = text_rgba("BON NADAL!", 120, (255, 255, 255), (214, 40, 40), 12)
T_ANY = text_rgba("i feliç any nou!", 74, (255, 209, 102), STROKE, 8)


def render_frame(t):
    img = BG.copy()
    # flocs de neu
    for (x0, y0, r, v, ph) in SNOW:
        yy = (y0 + v * t) % (H + 40) - 20
        xx = (x0 + 40 * math.sin(t * 0.8 + ph)) % W
        cv2.circle(img, (int(xx), int(yy)), int(r), (255, 255, 255), -1, AA)
    # llums arbre
    on_glow = clamp01((t - TF) / 1.0)
    for j, (lx, ly) in enumerate(TREE_LIGHTS):
        c = BULB_COL[(int(t * 3) + j) % 4]
        cv2.circle(img, (int(lx), int(ly)), 9, c, -1, AA)
        if (int(t * 3) + j) % 3 == 0:
            acirc(img, lx, ly, 24, c, .35)
    # llums garlanda
    for j, (bx, by) in enumerate(BULBS):
        c = BULB_COL[(int(t * 2.5) + j) % 4]
        cv2.circle(img, (int(bx), int(by)), 13, c, -1, AA)
        acirc(img, bx, by, 30, c, .25)
    # títol
    a1 = clamp01((t - 0.3) / 0.8)
    blit(img, T_X1, W / 2, 335 - 30 * (1 - a1), a1)
    blit(img, T_X2, W / 2, 450 - 30 * (1 - a1), a1)
    # compte enrere
    for j in range(4):
        s0 = TC + j * BEAT
        if s0 <= t < s0 + BEAT * .9:
            u = (t - s0) / (BEAT * .9)
            blit(img, T_NUM[j], W / 2, 700, 1 - u * u * .5, 1.35 - .35 * min(1, u * 4))
    # músics (de darrere cap al davant)
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        render_m(img, i, t)
    # text final
    if t >= TF:
        u = (t - TF) / 0.35
        sc = 1.0 if u >= 1 else 1.6 - .6 * u + (.08 * math.sin(u * 3.14) if u < 1 else 0)
        blit(img, T_BON, W / 2, 610, clamp01(u * 1.5), sc)
        blit(img, T_ANY, W / 2, 715, clamp01((t - TF - .4) / .5))
        for (x0, y0, w_, h_, v, a0, rs) in CONF:
            tt = t - TF
            if tt < 0:
                continue
            yy = y0 + 250 * tt * (v / 8) + 90 * tt * tt
            xx = x0 + 40 * math.sin(tt * 2 + a0)
            if yy < -20 or yy > H + 20:
                continue
            ang = a0 + tt * rs
            c, sn = math.cos(ang), math.sin(ang)
            pts = np.array([(xx + c * w_ / 2 - sn * h_ / 2, yy + sn * w_ / 2 + c * h_ / 2),
                            (xx - c * w_ / 2 - sn * h_ / 2, yy - sn * w_ / 2 + c * h_ / 2),
                            (xx - c * w_ / 2 + sn * h_ / 2, yy - sn * w_ / 2 - c * h_ / 2),
                            (xx + c * w_ / 2 + sn * h_ / 2, yy + sn * w_ / 2 - c * h_ / 2)], np.int32)
            cv2.fillPoly(img, [pts], CONF_COL[int(a0 * 10) % 6])
    return img


# ---------------------------------------------------------------- audio
SR = 44100
NS = int(SR * DUR)
mix = np.zeros(NS)


def put(sig, t, g=1.0):
    i = int(t * SR)
    if i < 0 or i >= NS:
        return
    n = min(len(sig), NS - i)
    mix[i:i + n] += sig[:n] * g


def tone(f, dur, kind):
    n = int(SR * (dur + .1))
    tt = np.arange(n) / SR
    vib = 1 + 0.005 * np.sin(2 * np.pi * 5.5 * tt) * np.minimum(1, tt / .15)
    ph = 2 * np.pi * np.cumsum(f * vib) / SR
    H_ = {'tpt': [1, .9, .8, .65, .5, .38, .26, .16, .1], 'sax': [1, .45, .75, .22, .4, .12, .18],
          'tbn': [1, .7, .5, .35, .22, .12], 'heli': [1, .75, .5, .35, .22, .14]}[kind]
    sig = sum(a * np.sin((j + 1) * ph) for j, a in enumerate(H_))
    if kind == 'sax':
        sig = sig + 0.15 * np.random.randn(n) * np.exp(-tt * 3)
    a, r = (.02, .06)
    e = np.ones(n)
    e[:int(a * SR)] = np.linspace(0, 1, int(a * SR))
    d0 = int(dur * SR)
    e[d0:] = np.maximum(0, 1 - (np.arange(n - d0) / SR) / r)
    e[int(a * SR):d0] *= np.linspace(1, .8, max(1, d0 - int(a * SR)))
    return sig * e / sum(H_) * 1.6


def kick():
    n = int(.4 * SR)
    tt = np.arange(n) / SR
    f = 110 * np.exp(-tt * 16) + 48
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 8)
    s[:200] += np.random.randn(200) * .5
    return s


def snare():
    n = int(.22 * SR)
    tt = np.arange(n) / SR
    nz = np.random.randn(n)
    nz = nz - np.roll(nz, 1)
    return (nz * .5 * np.exp(-tt * 26) + .5 * np.sin(2 * np.pi * 190 * tt) * np.exp(-tt * 40))


def click():
    n = int(.05 * SR)
    tt = np.arange(n) / SR
    return np.random.randn(n) * np.exp(-tt * 120) + np.sin(2 * np.pi * 1600 * tt) * np.exp(-tt * 90)


def crash():
    n = int(1.6 * SR)
    tt = np.arange(n) / SR
    nz = np.random.randn(n)
    nz = nz - np.roll(nz, 1)
    return nz * np.exp(-tt * 3.2) * .5


NOTE = dict(C4=261.63, D4=293.66, E4=329.63, F4=349.23, G4=392.0, C5=523.25)
MEL = [  # (compas, temps, nota, durada en temps)
    (0, 0, 'E4', 1), (0, 1, 'E4', 1), (0, 2, 'E4', 2),
    (1, 0, 'E4', 1), (1, 1, 'E4', 1), (1, 2, 'E4', 2),
    (2, 0, 'E4', 1), (2, 1, 'G4', 1), (2, 2, 'C4', 1.5), (2, 3.5, 'D4', .5),
    (3, 0, 'E4', 3.7),
    (4, 0, 'F4', 1), (4, 1, 'F4', 1), (4, 2, 'F4', 1.5), (4, 3.5, 'F4', .5),
    (5, 0, 'F4', 1), (5, 1, 'E4', 1), (5, 2, 'E4', 1), (5, 3, 'E4', .5), (5, 3.5, 'E4', .5),
    (6, 0, 'G4', 1), (6, 1, 'G4', 1), (6, 2, 'F4', 1), (6, 3, 'D4', 1),
    (7, 0, 'C4', 3.8),
]
CHORD = ['C', 'C', 'CG', 'C', 'F', 'FC', 'G', 'C']          # arrel per mig compas
BASS = dict(C=130.81, G=98.0, F=87.31)
AFTER = dict(C=(164.81, 196.0), F=(174.61, 220.0), G=(196.0, 246.94))
tb_ = lambda bar, beat: T0 + (bar * 4 + beat) * BEAT

for (b, bt, n, du) in MEL:
    f = NOTE[n]
    put(tone(f * 2, du * BEAT * .92, 'tpt'), tb_(b, bt), .30)
    put(tone(f * 1.0, du * BEAT * .92, 'tpt'), tb_(b, bt), .0)
    if b >= 2:
        put(tone(f, du * BEAT * .92, 'sax'), tb_(b, bt), .30)
for b in range(8):
    halves = [CHORD[b][0], CHORD[b][-1]]
    for hlf in range(2):
        r = halves[hlf]
        if b >= 4:
            put(tone(BASS[r], BEAT * .75, 'heli'), tb_(b, hlf * 2), .36)
            lo, hi = AFTER[r]
            for beat_ in (hlf * 2 + 1,):
                put(tone(lo, BEAT * .55, 'tbn'), tb_(b, beat_), .22)
                put(tone(hi, BEAT * .55, 'tbn'), tb_(b, beat_), .20)
    # percussió
    if b < 7:
        for beat_ in (0, 2):
            put(kick(), tb_(b, beat_), .85)
        for beat_ in (1, 3):
            put(snare(), tb_(b, beat_), .5)
        put(snare(), tb_(b, 3.5), .18)
put(kick(), TF, 1.0)
put(snare(), TF, .6)
put(crash(), TF, .5)
for hz in (C4 := 130.81, 164.81, 196.0):
    put(tone(hz, 3 * BEAT, 'tbn'), TF, .18)
put(tone(BASS['C'], 3 * BEAT, 'heli'), TF, .36)
for j in range(4):
    put(click(), TC + j * BEAT, .8)
for k_, m in enumerate(ORDER):
    L = LAND[m]
    n = int(.9 * SR)
    tt = np.arange(n) / SR
    put(np.sin(2 * np.pi * np.cumsum(1500 - 1000 * tt / .9) / SR) * .5 * np.minimum(1, tt * 8), L - 1.0, .05)
    n2 = int(.25 * SR)
    t2 = np.arange(n2) / SR
    fr = 260 + 280 * np.sin(np.pi * t2 / .25)
    put(np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t2 * 9), L, .22)
    put(kick() * .6, L, .35)
# cascavells de trineu al final
for j in range(10):
    tt = np.arange(int(.25 * SR)) / SR
    put(sum(np.sin(2 * np.pi * f_ * tt) for f_ in (3100, 4200, 5300)) * np.exp(-tt * 14), TF + .1 + j * .17, .05)
# reverb suau i normalitzacio
ir = np.random.randn(int(.35 * SR)) * np.exp(-np.arange(int(.35 * SR)) / SR * 9)
wet = fftconvolve(mix, ir)[:NS] * .035
out = np.tanh((mix + wet) * 1.3)
out = out / np.max(np.abs(out)) * .9
fade = int(.3 * SR)
out[-fade:] *= np.linspace(1, 0, fade)

wavfile.write('audio.wav', SR, (out * 32767).astype(np.int16))

# ---------------------------------------------------------------- sortida Instagram MP4
if len(sys.argv) > 1 and sys.argv[1] == 'test':
    times = [float(v) for v in sys.argv[2:]]
    tiles = [cv2.resize(render_frame(tt), (360, 640), interpolation=cv2.INTER_AREA) for tt in times]
    cv2.imwrite('test.jpg', np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print('ok')
else:
    sortida_mp4 = 'xaranga-nadal-ig.mp4'
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
        '-i', 'audio.wav',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '19',
        '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-level', '4.1',
        '-c:a', 'aac', '-b:a', '192k', '-ar', '44100',
        '-shortest', '-movflags', '+faststart',
        sortida_mp4
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr_ in range(int(DUR * FPS)):
        p.stdin.write(render_frame(fr_ / FPS).tobytes())
        if fr_ % 60 == 0:
            print(f'Renderitzant fotograma {fr_} de {int(DUR * FPS)}...', flush=True)
    p.stdin.close()
    p.wait()

    if os.path.exists('audio.wav'):
        os.remove('audio.wav')
    print(f'Vídeo llest per a Instagram: {sortida_mp4}')
