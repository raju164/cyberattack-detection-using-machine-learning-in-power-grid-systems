"""Procedural textures: blocky-avatar faces, outfit patterns, signs, calendars."""
import math
import random

from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT_BOLD = "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"
FONT_SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
INK = (25, 20, 20, 255)
WHITE = (255, 255, 255, 255)
BLUSH = (255, 130, 150, 170)


def font(size, path=FONT_BOLD):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


# --------------------------------------------------------------------------- faces
# 512x512 RGBA decal laid on the head's front face. Drawn at 2x and downsampled.

FACE_EXPRESSIONS = [
    "neutral", "smile", "happy", "laugh", "sleepy_blush", "sad", "crying", "scared",
    "shocked", "angry", "worried", "smug", "wink", "unimpressed", "love", "determined",
    "yelling", "nervous_sweat", "dizzy", "sleeping",
]


def face(expr, style="classic", lash=False):
    S = 1024
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    lw = 26
    ex, ey, er = 300, 440, 62  # eye centre x offset from middle, y, radius
    cx = S // 2
    mouth_y = 690

    def eye_dot(x, y, r=er, white=False, pupil_dy=0):
        if white:
            d.ellipse([x - r * 1.25, y - r * 1.45, x + r * 1.25, y + r * 1.45], fill=WHITE, outline=INK, width=16)
            d.ellipse([x - r * 0.6, y - r * 0.6 + pupil_dy, x + r * 0.6, y + r * 0.6 + pupil_dy], fill=INK)
        else:
            d.ellipse([x - r * 0.75, y - r, x + r * 0.75, y + r], fill=INK)
            d.ellipse([x - r * 0.35, y - r * 0.75, x - r * 0.02, y - r * 0.3], fill=WHITE)
        if lash:
            d.line([x - r, y - r * 1.1, x - r * 1.5, y - r * 1.6], fill=INK, width=14)
            d.line([x + r, y - r * 1.1, x + r * 1.5, y - r * 1.6], fill=INK, width=14)

    def arc_eye(x, y, up=True, w=90):
        box = [x - w, y - 60, x + w, y + 60]
        d.arc(box, 200 if up else 20, 340 if up else 160, fill=INK, width=lw)

    def blush():
        for sx in (-1, 1):
            d.ellipse([cx + sx * 330 - 85, 560, cx + sx * 330 + 85, 625], fill=BLUSH)

    def brows(kind):
        for sx in (-1, 1):
            x = cx + sx * ex
            if kind == "angry":
                d.line([x - sx * 90, 300, x + sx * 70, 350], fill=INK, width=lw + 6)
            elif kind == "worried":
                d.line([x - sx * 90, 345, x + sx * 70, 300], fill=INK, width=lw)
            elif kind == "raised":
                d.arc([x - 90, 290, x + 90, 370], 200, 340, fill=INK, width=lw)
            elif kind == "flat":
                d.line([x - 80, 330, x + 80, 330], fill=INK, width=lw)

    L, R = cx - ex, cx + ex
    if expr == "neutral":
        eye_dot(L, ey); eye_dot(R, ey)
        d.line([cx - 70, mouth_y, cx + 70, mouth_y], fill=INK, width=lw)
    elif expr == "smile":
        eye_dot(L, ey); eye_dot(R, ey)
        d.arc([cx - 150, mouth_y - 130, cx + 150, mouth_y + 60], 20, 160, fill=INK, width=lw)
    elif expr == "happy":
        eye_dot(L, ey); eye_dot(R, ey)
        d.chord([cx - 160, mouth_y - 110, cx + 160, mouth_y + 110], 0, 180, fill=INK)
        d.chord([cx - 90, mouth_y + 30, cx + 90, mouth_y + 105], 0, 180, fill=(230, 80, 90, 255))
    elif expr == "laugh":
        for sx in (-1, 1):  # >  <
            x = cx + sx * ex
            d.line([x - sx * 70, ey - 60, x + sx * 50, ey], fill=INK, width=lw + 4)
            d.line([x + sx * 50, ey, x - sx * 70, ey + 60], fill=INK, width=lw + 4)
        d.chord([cx - 170, mouth_y - 120, cx + 170, mouth_y + 140], 0, 180, fill=INK)
        d.chord([cx - 100, mouth_y + 40, cx + 100, mouth_y + 135], 0, 180, fill=(230, 80, 90, 255))
        blush()
    elif expr == "sleepy_blush":
        arc_eye(L, ey + 20, up=False); arc_eye(R, ey + 20, up=False)
        d.arc([cx - 120, mouth_y - 110, cx + 120, mouth_y + 50], 20, 160, fill=INK, width=lw)
        blush()
    elif expr == "sad":
        brows("worried"); eye_dot(L, ey); eye_dot(R, ey)
        d.arc([cx - 120, mouth_y - 10, cx + 120, mouth_y + 140], 200, 340, fill=INK, width=lw)
    elif expr == "crying":
        brows("worried"); arc_eye(L, ey, up=False); arc_eye(R, ey, up=False)
        for x in (L, R):
            d.rounded_rectangle([x - 30, ey + 50, x + 30, ey + 300], 30, fill=(110, 190, 255, 230))
        d.arc([cx - 130, mouth_y - 10, cx + 130, mouth_y + 150], 200, 340, fill=INK, width=lw)
    elif expr == "scared":
        brows("worried"); eye_dot(L, ey, white=True, pupil_dy=10); eye_dot(R, ey, white=True, pupil_dy=10)
        d.ellipse([cx - 45, mouth_y - 10, cx + 45, mouth_y + 90], fill=INK)
    elif expr == "shocked":
        brows("raised"); eye_dot(L, ey, white=True); eye_dot(R, ey, white=True)
        d.ellipse([cx - 80, mouth_y - 50, cx + 80, mouth_y + 140], fill=INK)
        d.ellipse([cx - 45, mouth_y + 60, cx + 45, mouth_y + 125], fill=(230, 80, 90, 255))
    elif expr == "angry":
        brows("angry"); eye_dot(L, ey + 15, r=50); eye_dot(R, ey + 15, r=50)
        d.arc([cx - 120, mouth_y + 10, cx + 120, mouth_y + 140], 200, 340, fill=INK, width=lw)
    elif expr == "worried":
        brows("worried"); eye_dot(L, ey); eye_dot(R, ey)
        pts = [(cx - 130 + i * 52, mouth_y + (20 if i % 2 else -10)) for i in range(6)]
        d.line(pts, fill=INK, width=lw - 4, joint="curve")
    elif expr == "smug":
        brows("flat")
        for x in (L, R):
            d.chord([x - 75, ey - 60, x + 75, ey + 70], 0, 180, fill=INK)
        d.arc([cx - 40, mouth_y - 90, cx + 170, mouth_y + 40], 20, 120, fill=INK, width=lw)
    elif expr == "wink":
        eye_dot(L, ey); arc_eye(R, ey + 10, up=True)
        d.arc([cx - 150, mouth_y - 130, cx + 150, mouth_y + 60], 20, 160, fill=INK, width=lw)
        blush()
    elif expr == "unimpressed":
        for x in (L, R):
            d.chord([x - 75, ey - 60, x + 75, ey + 70], 0, 180, fill=INK)
            d.line([x - 85, ey - 5, x + 85, ey - 5], fill=INK, width=lw)
        d.line([cx - 60, mouth_y + 20, cx + 90, mouth_y + 10], fill=INK, width=lw)
    elif expr == "love":
        for x in (L, R):
            d.polygon([(x, ey + 75), (x - 85, ey - 5), (x - 45, ey - 65), (x, ey - 30),
                       (x + 45, ey - 65), (x + 85, ey - 5)], fill=(230, 40, 70, 255))
        d.arc([cx - 150, mouth_y - 130, cx + 150, mouth_y + 60], 20, 160, fill=INK, width=lw)
        blush()
    elif expr == "determined":
        brows("angry"); eye_dot(L, ey); eye_dot(R, ey)
        d.line([cx - 100, mouth_y + 20, cx + 100, mouth_y + 20], fill=INK, width=lw + 4)
    elif expr == "yelling":
        brows("angry"); eye_dot(L, ey, white=True); eye_dot(R, ey, white=True)
        d.rounded_rectangle([cx - 140, mouth_y - 60, cx + 140, mouth_y + 170], 70, fill=INK)
        d.rectangle([cx - 110, mouth_y - 50, cx + 110, mouth_y - 15], fill=WHITE)
    elif expr == "nervous_sweat":
        brows("worried"); eye_dot(L, ey); eye_dot(R, ey)
        d.line([cx - 110, mouth_y + 20, cx + 110, mouth_y + 20], fill=INK, width=lw)
        d.polygon([(870, 200), (830, 300), (910, 300)], fill=(110, 190, 255, 255))
        d.ellipse([828, 260, 912, 340], fill=(110, 190, 255, 255))
    elif expr == "dizzy":
        for x in (L, R):
            for k in range(3):
                r = 30 + k * 28
                d.arc([x - r, ey - r, x + r, ey + r], k * 90, k * 90 + 270, fill=INK, width=14)
        pts = [(cx - 120 + i * 48, mouth_y + (25 if i % 2 else 0)) for i in range(6)]
        d.line(pts, fill=INK, width=lw - 4)
    elif expr == "sleeping":
        arc_eye(L, ey + 20, up=False); arc_eye(R, ey + 20, up=False)
        d.ellipse([cx - 30, mouth_y, cx + 30, mouth_y + 50], outline=INK, width=16)
        f = font(150)
        d.text((800, 120), "z", font=f, fill=INK)
        d.text((880, 30), "z", font=font(100), fill=INK)
    else:
        raise ValueError(expr)
    return im.resize((512, 512), Image.LANCZOS)


# --------------------------------------------------------------------------- outfits

def solid(color, size=64):
    return Image.new("RGB", (size, size), color)


def stripes(c1, c2, n=8, size=256, vertical=False):
    im = Image.new("RGB", (size, size), c1)
    d = ImageDraw.Draw(im)
    h = size / (n * 2)
    for i in range(n):
        y0 = int(i * 2 * h + h)
        if vertical:
            d.rectangle([y0, 0, int(y0 + h) - 1, size], fill=c2)
        else:
            d.rectangle([0, y0, size, int(y0 + h) - 1], fill=c2)
    return im


def plaid(base, line, size=256):
    im = Image.new("RGB", (size, size), base)
    d = ImageDraw.Draw(im, "RGBA")
    for i in range(0, size, 42):
        d.rectangle([i, 0, i + 8, size], fill=line + (160,))
        d.rectangle([0, i, size, i + 8], fill=line + (160,))
        d.rectangle([i + 20, 0, i + 22, size], fill=(255, 255, 255, 90))
    return im


def hoodie_front(color, string=(240, 240, 240), size=256):
    im = Image.new("RGB", (size, size), color)
    d = ImageDraw.Draw(im)
    dark = tuple(max(0, c - 40) for c in color)
    d.rounded_rectangle([50, 165, 206, 240], 12, fill=dark)  # pocket
    d.line([95, 0, 85, 110], fill=string, width=9)
    d.line([161, 0, 171, 110], fill=string, width=9)
    d.polygon([(70, 0), (186, 0), (128, 40)], fill=dark)  # collar dip
    return im


def suit_front(jacket, shirt=(245, 245, 245), tie=(25, 25, 25), size=256, bowtie=False, sparkle=False):
    im = Image.new("RGB", (size, size), jacket)
    d = ImageDraw.Draw(im)
    if sparkle:
        rnd = random.Random(4)
        for _ in range(900):
            x, y = rnd.randrange(size), rnd.randrange(size)
            c = rnd.choice([(255, 215, 90), (200, 150, 40), (255, 240, 170), (150, 100, 20)])
            d.rectangle([x, y, x + 3, y + 3], fill=c)
    d.polygon([(85, 0), (171, 0), (128, 150)], fill=shirt)
    if bowtie:
        d.polygon([(100, 15), (128, 30), (100, 45)], fill=tie)
        d.polygon([(156, 15), (128, 30), (156, 45)], fill=tie)
    else:
        d.polygon([(120, 10), (136, 10), (142, 120), (128, 140), (114, 120)], fill=tie)
    d.line([(85, 0), (128, 150)], fill=(0, 0, 0), width=4)
    d.line([(171, 0), (128, 150)], fill=(0, 0, 0), width=4)
    for y in (175, 215):
        d.ellipse([122, y, 134, y + 12], fill=(20, 20, 20))
    return im


def jersey(color, number="10", trim=(255, 255, 255), size=256):
    im = Image.new("RGB", (size, size), color)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, size, 14], fill=trim)
    d.text((size / 2, size / 2 + 10), number, font=font(120), fill=trim, anchor="mm")
    return im


def overalls(denim=(70, 100, 170), shirt=(220, 80, 50), size=256):
    im = Image.new("RGB", (size, size), shirt)
    d = ImageDraw.Draw(im)
    d.rectangle([60, 110, 196, size], fill=denim)
    d.rectangle([60, 0, 85, 115], fill=denim)
    d.rectangle([171, 0, 196, 115], fill=denim)
    for x in (72, 184):
        d.ellipse([x - 9, 100, x + 9, 118], fill=(230, 200, 80))
    return im


def pajama(color=(250, 170, 200), dot=(255, 255, 255), size=256):
    im = Image.new("RGB", (size, size), color)
    d = ImageDraw.Draw(im)
    for y in range(10, size, 48):
        for x in range(10 + (y // 48 % 2) * 24, size, 48):
            d.ellipse([x, y, x + 12, y + 12], fill=dot)
    return im


def button_shirt(color=(240, 120, 40), btn=(250, 210, 60), size=256):
    im = Image.new("RGB", (size, size), color)
    d = ImageDraw.Draw(im)
    light = tuple(min(255, c + 60) for c in color)
    d.polygon([(90, 0), (166, 0), (128, 60)], fill=light)
    for y in (90, 150, 210):
        d.ellipse([116, y, 140, y + 24], fill=btn)
    return im


def school_uniform(jacket=(30, 30, 35), shirt=(245, 245, 245), tie=(20, 20, 20), size=256):
    return suit_front(jacket, shirt, tie, size)


def dress_pink(size=256):
    im = Image.new("RGB", (size, size), (255, 170, 210))
    d = ImageDraw.Draw(im)
    for y in range(0, size, 40):
        d.line([(0, y + 20), (size, y + 20)], fill=(255, 220, 235), width=6)
    d.polygon([(100, 20), (128, 45), (100, 70)], fill=(230, 70, 140))
    d.polygon([(156, 20), (128, 45), (156, 70)], fill=(230, 70, 140))
    return im


def wood(base=(150, 95, 55), size=256, seed=1, planks=4):
    rnd = random.Random(seed)
    im = Image.new("RGB", (size, size), base)
    d = ImageDraw.Draw(im)
    for _ in range(140):
        y = rnd.randrange(size)
        c = tuple(max(0, min(255, v + rnd.randint(-22, 14))) for v in base)
        d.line([(0, y), (size, y + rnd.randint(-4, 4))], fill=c, width=rnd.randint(1, 3))
    for i in range(1, planks):
        y = i * size // planks
        d.line([(0, y), (size, y)], fill=tuple(int(v * 0.6) for v in base), width=3)
    return im


def brick(base=(170, 80, 60), mortar=(215, 205, 190), size=256):
    im = Image.new("RGB", (size, size), mortar)
    d = ImageDraw.Draw(im)
    rnd = random.Random(3)
    bh, bw = 32, 64
    for r in range(size // bh):
        off = (r % 2) * bw // 2
        for c in range(-1, size // bw + 1):
            x = c * bw + off
            col = tuple(max(0, min(255, v + rnd.randint(-15, 15))) for v in base)
            d.rectangle([x + 3, r * bh + 3, x + bw - 3, r * bh + bh - 3], fill=col)
    return im


def siding(base=(225, 225, 230), size=256, rows=12):
    im = Image.new("RGB", (size, size), base)
    d = ImageDraw.Draw(im)
    for i in range(rows):
        y = i * size // rows
        d.line([(0, y), (size, y)], fill=tuple(int(v * 0.82) for v in base), width=3)
    return im


def window_grid(frame=(255, 255, 255), glass=(140, 190, 230), cols=3, rows=2, lit=False, size=256):
    im = Image.new("RGB", (size, size), frame)
    d = ImageDraw.Draw(im)
    g = (255, 225, 140) if lit else glass
    m = 14
    cw = (size - m * (cols + 1)) / cols
    ch = (size - m * (rows + 1)) / rows
    for r in range(rows):
        for c in range(cols):
            x = m + c * (cw + m); y = m + r * (ch + m)
            d.rectangle([x, y, x + cw, y + ch], fill=g)
            if not lit:
                d.line([(x + cw * 0.2, y + ch * 0.8), (x + cw * 0.6, y + ch * 0.2)], fill=(210, 235, 255), width=6)
    return im


def road(size=512):
    im = Image.new("RGB", (size, size), (70, 72, 78))
    d = ImageDraw.Draw(im)
    rnd = random.Random(7)
    for _ in range(1500):
        x, y = rnd.randrange(size), rnd.randrange(size)
        v = rnd.randint(60, 90)
        d.point((x, y), fill=(v, v, v + 4))
    for y in range(0, size, 128):
        d.rectangle([size // 2 - 8, y + 20, size // 2 + 8, y + 90], fill=(250, 205, 40))
    d.rectangle([12, 0, 22, size], fill=(235, 235, 235))
    d.rectangle([size - 22, 0, size - 12, size], fill=(235, 235, 235))
    return im


def tiles(c1=(200, 200, 205), c2=(175, 175, 182), n=4, size=256):
    im = Image.new("RGB", (size, size), c1)
    d = ImageDraw.Draw(im)
    s = size // n
    for i in range(n):
        for j in range(n):
            if (i + j) % 2:
                d.rectangle([i * s, j * s, i * s + s, j * s + s], fill=c2)
    return im


def cobble(base=(205, 190, 150), size=256):
    rnd = random.Random(11)
    im = Image.new("RGB", (size, size), tuple(int(v * 0.75) for v in base))
    d = ImageDraw.Draw(im)
    for y in range(0, size, 32):
        for x in range(-(y // 32 % 2) * 16, size, 32):
            col = tuple(max(0, min(255, v + rnd.randint(-18, 18))) for v in base)
            d.rounded_rectangle([x + 2, y + 2, x + 30, y + 30], 7, fill=col)
    return im


def grass(size=256):
    rnd = random.Random(5)
    im = Image.new("RGB", (size, size), (95, 170, 70))
    d = ImageDraw.Draw(im)
    for _ in range(900):
        x, y = rnd.randrange(size), rnd.randrange(size)
        g = rnd.randint(140, 195)
        d.line([(x, y), (x + rnd.randint(-2, 2), y - rnd.randint(3, 7))], fill=(80, g, 60))
    return im


# --------------------------------------------------------------------------- signs / graphics

def sign(text, bg=(30, 60, 160), fg=(255, 255, 255), w=1024, h=256, border=None, fsize=None, serif=False):
    im = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(im)
    if border:
        d.rectangle([8, 8, w - 9, h - 9], outline=border, width=12)
    fs = fsize or int(h * 0.42)
    f = font(fs, FONT_SERIF if serif else FONT_BOLD)
    while d.textlength(text, font=f) > w * 0.88 and fs > 10:
        fs -= 4; f = font(fs, FONT_SERIF if serif else FONT_BOLD)
    d.text((w / 2, h / 2), text, font=f, fill=fg, anchor="mm")
    return im


def calendar_page(month="NOVEMBER", big="DAY 1", size=512):
    im = Image.new("RGB", (size, size), (250, 248, 240))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, size, 140], fill=(205, 40, 45))
    d.text((size / 2, 85), month, font=font(66), fill=WHITE, anchor="mm")
    for x in (130, size - 130):
        d.rounded_rectangle([x - 12, 0, x + 12, 50], 6, fill=(40, 40, 40))
    fs = 210
    f = font(fs)
    while d.textlength(big, font=f) > size * 0.86:
        fs -= 8; f = font(fs)
    d.text((size / 2, 330), big, font=f, fill=(20, 20, 20), anchor="mm")
    return im


def newspaper(headline="TOWN NEWS", sub="SPECIAL EDITION", size=(512, 700)):
    w, h = size
    im = Image.new("RGB", size, (232, 230, 222))
    d = ImageDraw.Draw(im)
    d.text((w / 2, 50), "THE DAILY TOWN", font=font(46, FONT_SERIF), fill=(20, 20, 20), anchor="mm")
    d.line([(20, 85), (w - 20, 85)], fill=(20, 20, 20), width=3)
    d.text((w / 2, 102), sub, font=font(18, FONT_SERIF), fill=(60, 60, 60), anchor="mm")
    y = 150
    for line in headline.split("\n"):
        fs = 70
        f = font(fs)
        while d.textlength(line, font=f) > w - 50:
            fs -= 4; f = font(fs)
        d.text((w / 2, y), line, font=f, fill=(200, 25, 30), anchor="mm")
        y += fs + 8
    d.rectangle([30, y + 10, w // 2 - 10, h - 60], fill=(160, 160, 160))
    for ly in range(y + 15, h - 60, 22):
        d.rectangle([w // 2 + 10, ly, w - 30, ly + 10], fill=(40, 40, 40))
    return im


def mood_dial(size=512):
    """Sun-to-storm gauge face (half yellow, half purple)."""
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = size / 2
    d.ellipse([10, 10, size - 10, size - 10], fill=(235, 175, 50))
    for i in range(90):
        t = i / 89
        col = (int(250 * (1 - t) + 90 * t), int(190 * (1 - t) + 60 * t), int(60 * (1 - t) + 120 * t))
        a0 = -90 + i * 2
        d.pieslice([40, 40, size - 40, size - 40], a0 - 90, a0 - 88, fill=col)
    for i in range(90):
        t = i / 89
        col = (int(250 * (1 - t) + 90 * t), int(190 * (1 - t) + 60 * t), int(60 * (1 - t) + 120 * t))
        d.pieslice([40, 40, size - 40, size - 40], -180 + i * 4, -176 + i * 4, fill=col)
    d.ellipse([140, 220, 200, 280], fill=(255, 235, 120))
    d.ellipse([320, 230, 380, 270], fill=(230, 230, 240))
    d.polygon([(350, 270), (335, 310), (352, 305), (340, 340), (370, 290), (352, 294)], fill=(255, 220, 40))
    d.ellipse([c - 22, c - 22, c + 22, c + 22], fill=(50, 30, 25))
    return im


def speech_bubble(text="GOOD MORNING!", w=512, h=256):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([10, 10, w - 10, h - 60], 90, fill=WHITE, outline=INK, width=8)
    d.polygon([(w / 2 - 30, h - 64), (w / 2 + 30, h - 64), (w / 2, h - 8)], fill=WHITE)
    d.line([(w / 2 - 30, h - 62), (w / 2, h - 8), (w / 2 + 30, h - 62)], fill=INK, width=8)
    f = font(56)
    while d.textlength(text, font=f) > w - 80:
        f = font(f.size - 4)
    d.text((w / 2, (h - 50) / 2), text, font=f, fill=INK, anchor="mm")
    return im


def chalkboard(lines, size=(1024, 512)):
    im = Image.new("RGB", size, (40, 85, 55))
    d = ImageDraw.Draw(im)
    rnd = random.Random(2)
    for _ in range(200):
        x, y = rnd.randrange(size[0]), rnd.randrange(size[1])
        d.point((x, y), fill=(70, 115, 85))
    y = 50
    for i, ln in enumerate(lines):
        d.text((50, y), ln, font=font(64 if i == 0 else 40), fill=(240, 240, 230))
        y += 90 if i == 0 else 60
    return im


def poster(title="STARLIGHT", sub="PRINCESS", bg=(250, 150, 200), size=(256, 384)):
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    w, h = size
    d.rectangle([10, 10, w - 10, h - 10], outline=(255, 255, 255), width=6)
    d.text((w / 2, 45), title, font=font(30), fill=(255, 255, 255), anchor="mm")
    d.ellipse([w / 2 - 50, 90, w / 2 + 50, 190], fill=(255, 220, 200))
    d.polygon([(w / 2 - 80, h - 70), (w / 2 + 80, h - 70), (w / 2, 180)], fill=(220, 60, 140))
    d.text((w / 2, h - 40), sub, font=font(30), fill=(255, 255, 255), anchor="mm")
    return im


def sky_gradient(top, bottom, w=64, h=512):
    im = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(im)
    for y in range(h):
        t = y / (h - 1)
        d.line([(0, y), (w, y)], fill=tuple(int(a * (1 - t) + b * t) for a, b in zip(top, bottom)))
    return im


def rain_streaks(w=512, h=512, n=260, seed=9):
    rnd = random.Random(seed)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for _ in range(n):
        x, y = rnd.randrange(w), rnd.randrange(h)
        l = rnd.randint(20, 55)
        d.line([(x, y), (x - l * 0.12, y + l)], fill=(220, 230, 255, rnd.randint(110, 200)), width=2)
    return im


def star_sky(w=1024, h=512, seed=3):
    rnd = random.Random(seed)
    im = sky_gradient((8, 10, 40), (35, 45, 110), w, h)
    d = ImageDraw.Draw(im)
    for _ in range(300):
        x, y = rnd.randrange(w), rnd.randrange(h)
        b = rnd.randint(150, 255)
        d.point((x, y), fill=(b, b, b))
    return im


def soft_glow(color=(255, 230, 150), size=256):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for r in range(size // 2, 0, -2):
        a = int(255 * (1 - r / (size / 2)) ** 2)
        d.ellipse([size / 2 - r, size / 2 - r, size / 2 + r, size / 2 + r], fill=color + (a,))
    return im


def caption_card(word="TOWN", highlight=True, w=1024, h=256):
    """Example of the bold outlined caption style used in the references."""
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = font(150)
    fill = (60, 230, 60, 255) if highlight else WHITE
    d.text((w / 2, h / 2), word, font=f, fill=fill, anchor="mm", stroke_width=10, stroke_fill=(0, 0, 0, 255))
    return im
