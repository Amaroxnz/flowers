import sys
import time
import shutil
import argparse
import cv2
import numpy as np


class Terminal:
    def write(self, text: str):
        sys.stdout.write(text)
        sys.stdout.flush()

    def clear(self):
        self.write("\x1b[2J\x1b[H")

    def hide_cursor(self):
        self.write("\x1b[?25l")

    def show_cursor(self):
        self.write("\x1b[?25h")


class Renderer:
    def __init__(
        self,
        cols: int,
        rows: int,
        term: Terminal,
        ramp: str = " .:-=+*#%@",
        color: bool = True,
        edges: float = 0.35,
        saturation: float = 1.2,
        bg_tint: float = 0.15,
        use_blocks: bool = False,
        x0: int = 1,
        y0: int = 1,
    ):
        self.cols = cols
        self.rows = rows
        self.term = term
        self.ramp = np.array(list(ramp), dtype="<U1")
        self.color = color
        self.edges = edges
        self.saturation = saturation
        self.bg_tint = bg_tint
        self.mode = "blocks" if use_blocks else "ascii"
        self.x0 = x0
        self.y0 = y0

        self.lut = np.arange(256, dtype=np.uint8)
        self.prev = None
        self.cur_fg = None
        self.cur_bg = None

    def _pack_rgb(self, rgb: np.ndarray) -> np.ndarray:
        r = rgb[..., 0].astype(np.int32)
        g = rgb[..., 1].astype(np.int32)
        b = rgb[..., 2].astype(np.int32)
        return (r << 16) | (g << 8) | b

    def _sgr(self, kind: str, packed: int) -> str:
        if packed < 0:
            return "\x1b[39m" if kind == "f" else "\x1b[49m"
        r = (packed >> 16) & 255
        g = (packed >> 8) & 255
        b = packed & 255
        code = 38 if kind == "f" else 48
        return f"\x1b[{code};2;{r};{g};{b}m"

    def convert(self, frame: np.ndarray):
        s, cols, rows = self, self.cols, self.rows
        use_blocks = s.mode == "blocks" and s.color
        img = cv2.resize(
            frame,
            (cols, rows * 2 if use_blocks else rows),
            interpolation=cv2.INTER_AREA,
        )
        img = cv2.LUT(img, self.lut)

        if s.saturation != 1.0 and s.color:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * s.saturation, 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        rgb = img[..., ::-1].astype(np.float32)

        if use_blocks:
            glyph = np.full((rows, cols), "▀", dtype="<U1")
            return glyph, self._pack_rgb(rgb[0::2]), self._pack_rgb(rgb[1::2])

        lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0
        n = len(self.ramp)
        idx = np.clip((lum * (n - 1) + 0.5).astype(np.int32), 0, n - 1)
        idx[lum < 0.06] = 0
        glyph = self.ramp[idx]

        if s.edges > 0:
            gray = lum.astype(np.float32)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.hypot(gx, gy) / 4.0
            ang = (np.degrees(np.arctan2(gy, gx)) + 90.0) % 180.0
            sel = np.array(["-", "\\", "|", "/"])[((ang + 22.5) % 180.0 // 45).astype(np.int32)]
            mask = (mag > (0.8 - 0.6 * s.edges)) & (lum > 0.12)
            glyph = np.where(mask, sel, glyph)

        if not s.color:
            neg = np.full(glyph.shape, -1, dtype=np.int32)
            return glyph, neg, neg

        v = np.maximum(rgb.max(-1, keepdims=True), 1.0)
        fg_rgb = rgb * 0.4 + rgb * (255.0 / v) * 0.6
        fg = self._pack_rgb(fg_rgb)
        bg = self._pack_rgb(rgb * s.bg_tint)
        blank = glyph == " "
        fg = np.where(blank, 0, fg)
        return glyph, fg, bg

    def draw(self, grid) -> int:
        glyph, fg, bg = grid
        out: list[str] = []
        pg = self.prev

        for y in range(self.rows):
            if pg is None:
                idxs = np.arange(self.cols)
            else:
                ch = (glyph[y] != pg[0][y]) | (fg[y] != pg[1][y]) | (bg[y] != pg[2][y])
                idxs = np.flatnonzero(ch)
                if idxs.size == 0:
                    continue

            seg_start = prev_i = int(idxs[0])
            segs = []
            for i in idxs[1:]:
                i = int(i)
                if i - prev_i > 6:
                    segs.append((seg_start, prev_i))
                    seg_start = i
                prev_i = i
            segs.append((seg_start, prev_i))

            for a, b in segs:
                out.append(f"\x1b[{self.y0 + y};{self.x0 + a}H")
                for x in range(a, b + 1):
                    f, k = int(fg[y, x]), int(bg[y, x])
                    if k != self.cur_bg:
                        out.append(self._sgr("b", k))
                        self.cur_bg = k
                    if f != self.cur_fg:
                        out.append(self._sgr("f", f))
                        self.cur_fg = f
                    out.append(glyph[y, x])

        self.prev = (glyph.copy(), fg.copy(), bg.copy())
        text = "".join(out)
        if text:
            self.term.write(text)
        return len(text)


def play(video_path: str, use_blocks: bool = False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = 1.0 / fps
    vid_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    vid_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    term_size = shutil.get_terminal_size((80, 24))
    max_cols = term_size.columns - 2
    max_rows = term_size.lines - 2

    # Ajusta proporção considerando que caracteres no terminal são ~2x mais altos que largos
    aspect = (vid_w / vid_h) * 2.0
    cols = max_cols
    rows = int(cols / aspect)
    if rows > max_rows:
        rows = max_rows
        cols = int(rows * aspect)

    term = Terminal()
    renderer = Renderer(cols=cols, rows=rows, term=term, use_blocks=use_blocks)

    term.hide_cursor()
    term.clear()

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            grid = renderer.convert(frame)
            renderer.draw(grid)

            elapsed = time.perf_counter() - t0
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass
    finally:
        term.write("\x1b[0m")
        term.show_cursor()
        cap.release()
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reprodutor de vídeo ASCII no terminal")
    parser.add_argument("video", help="Caminho para o arquivo de vídeo (.mp4, .mov, etc.)")
    parser.add_argument("--blocks", action="store_true", help="Usar blocos Unicode (▀) em vez de ASCII")
    args = parser.parse_args()

    play(args.video, use_blocks=args.blocks)