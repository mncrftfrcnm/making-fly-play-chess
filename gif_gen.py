from PIL import Image, ImageDraw, ImageFont
import re

MOVES = """ 1. g4  d5
 2. Bh3 a5
 3. b4  Be6
 4. Ba3 Qd7
 5. Nc3 c6
 6. Ne4 dxe4
 7. b5  Qd4
 8. Qb1 cxb5
 9. Nf3 Bxa2
10. Nxd4     Nf6
11. Nb3 e6
12. Qxa2     Bxa3
13. Qb1 Ke7
14. Qa2 a4
15. Qb1 axb3
16. Qa2 bxa2
17. Rc1 a1=R
18. Rxa1     g6
19. Ra2 b6
20. Ra1 h5
21. Ra2 Nh7
22. d4  hxg4
23. Ra1 gxh3
24. Ra2 f6
25. Ra1 f5
26. Ra2 f4
27. Ra1 f3
28. Ra2 fxe2
29. Ra1 e3
30. Ra2 exf2+
31. Kxf2     Nf8
32. Ke1 Nh7
33. Ra1 e5
34. Rf1 Ke6
35. Rf3 exd4
36. Ra2 g5
37. Ra1 g4
38. Ra2 gxf3
39. Ra1 f2+
40. Kxf2     Nf8
41. Ke1 Rh7
42. Rb1 Rg7
43. Ra1 b4
44. Rb1 b3
45. Ra1 bxc2
46. Rb1 cxb1=Q+
47. Kxe2 b5
48. Kf3 Nc6
49. Ke2 Ne7
50. Kf3 b4""".split()

FILES, RANKS = "abcdefgh", "12345678"
PIECE = {
    "K":"♔","Q":"♕","R":"♖","B":"♗","N":"♘","P":"♙",
    "k":"♚","q":"♛","r":"♜","b":"♝","n":"♞","p":"♟",
}

def square(s): return (FILES.index(s[0]), RANKS.index(s[1]))
def sqname(p): return FILES[p[0]] + RANKS[p[1]]
def side_of(p): return "w" if p.isupper() else "b"

def start_board():
    b = {(f,1):"P" for f in range(8)}
    b.update({(0,0):"R",(1,0):"N",(2,0):"B",(3,0):"Q",
    (4,0):"K",(5,0):"B",(6,0):"N",(7,0):"R"})
    b.update({(f,6):"p" for f in range(8)})
    b.update({(0,7):"r",(1,7):"n",(2,7):"b",(3,7):"q",
    (4,7):"k",(5,7):"b",(6,7):"n",(7,7):"r"})
    return b

def clear_path(a, z, b):
    dx, dy = z[0]-a[0], z[1]-a[1]
    sx = 0 if dx == 0 else (1 if dx > 0 else -1)
    sy = 0 if dy == 0 else (1 if dy > 0 else -1)
    p = (a[0]+sx, a[1]+sy)
    while p != z:
   if p in b: return False
   p = (p[0]+sx, p[1]+sy)
    return True

def can_move(piece, a, z, b, capture):
    dx, dy = z[0]-a[0], z[1]-a[1]
    t = piece.upper()
    if t == "N": return sorted((abs(dx), abs(dy))) == [1, 2]
    if t == "B": return dx and abs(dx) == abs(dy) and clear_path(a,z,b)
    if t == "R": return ((dx == 0) ^ (dy == 0)) and clear_path(a,z,b)
    if t == "Q":
   return (((dx == 0) ^ (dy == 0)) or (dx and abs(dx) == abs(dy))) and clear_path(a,z,b)
    if t == "K": return max(abs(dx), abs(dy)) == 1
    if t == "P":
   step = 1 if piece.isupper() else -1
   home = 1 if piece.isupper() else 6
   if capture: return dy == step and abs(dx) == 1
   if dx: return False
   if dy == step and z not in b: return True
   mid = (a[0], a[1] + step)
   return a[1] == home and dy == 2*step and z not in b and mid not in b
    return False

def apply_san(san, side, b):
    s = san.replace("+","").replace("#","")
    m = re.search(r"([a-h][1-8])(?:=[QRBN])?$", s)
    if not m: raise ValueError("Unsupported SAN: " + san)
    z = square(m.group(1))
    pre = s[:m.start()]
    capture = "x" in pre

    if pre and pre[0] in "KQRBN":
   kind, hint = pre[0], pre[1:].replace("x","")
    else:
   kind, hint = "P", pre.replace("x","")

    wanted = kind if side == "w" else kind.lower()
    src = []
    for a, p in b.items():
   if p != wanted: continue
   if any(ch in FILES and a[0] != FILES.index(ch) for ch in hint): continue
   if any(ch in RANKS and a[1] != RANKS.index(ch) for ch in hint): continue
   if z in b and side_of(b[z]) == side: continue
   if capture != (z in b): continue   # enough for this move list; no en-passant here
   if can_move(p, a, z, b, capture): src.append(a)

    if len(src) != 1:
   raise ValueError(f"{san}: expected 1 source, got {[sqname(x) for x in src]}")
    a = src[0]
    p = b.pop(a)
    b.pop(z, None)
    b[z] = p
    return a, z

def font(size, bold=False):
    paths = [
   "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
   "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for p in paths:
   try: return ImageFont.truetype(p, size)
   except OSError: pass
    return ImageFont.load_default()

def frame(board, label, last=None):
    S, top = 80, 86
    im = Image.new("RGB", (8*S, top + 8*S + 34), "#181818")
    d = ImageDraw.Draw(im)
    light, dark = "#f0d9b5", "#b58863"
    hi1, hi2 = "#f6f669", "#d6d640"

    d.text((18, 18), label, font=font(34, True), fill="white")
    for r in range(8):
   for f in range(8):
  x, y = f*S, top + (7-r)*S
  c = light if (f+r) % 2 else dark
  if last and (f,r) in last:
      c = hi1 if c == light else hi2
  d.rectangle((x,y,x+S,y+S), fill=c)

    pf = font(58)
    for (f,r), p in board.items():
   x, y = f*S, top + (7-r)*S
   glyph = PIECE[p]
   box = d.textbbox((0,0), glyph, font=pf)
   w, h = box[2]-box[0], box[3]-box[1]
   fill = "#fafafa" if p.isupper() else "#111111"
   stroke = "#111111" if p.isupper() else "#f5f5f5"
   d.text((x+(S-w)/2, y+(S-h)/2-4), glyph, font=pf, fill=fill,
     stroke_width=1, stroke_fill=stroke)

    cf = font(18, True)
    for f in range(8):
   d.text((f*S+5, top+8*S+5), FILES[f], font=cf, fill="#dddddd")
    return im

def main():
    b = start_board()
    frames = [frame(b, "Start position")]
    durations = [1000]
    side = "w"

    for ply, san in enumerate(MOVES, 1):
   a, z = apply_san(san, side, b)
   move_no = (ply + 1) // 2
   who = "White" if side == "w" else "Black"
   frames.append(frame(b, f"{move_no}. {who}: {san}", (a,z)))
   durations.append(480)
   side = "b" if side == "w" else "w"

    durations[-1] = 2200
    frames[0].save(
   "chess_selfplay.gif",
   save_all=True,
   append_images=frames[1:],
   duration=durations,
   loop=0,
   disposal=2,
    )
    print("Wrote chess_selfplay.gif")

if __name__ == "__main__":
    main()