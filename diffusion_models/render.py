# The denoising steps as an animated GIF (v1's terminal-style frames, without the chat layout).

import os

import numpy as np


def _font(size=20):
    from PIL import ImageFont
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                 "/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def wrap(text, width=90):
    out = []
    for para in text.split("\n"):
        para = para.rstrip()
        if not para:
            out.append("")
        while len(para) > width:
            out.append(para[:width])
            para = para[width:]
        if para:
            out.append(para)
    return out


def frame(lines, width=1200, height=700, font_size=20, margin=20, spacing=6):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (width, height), (10, 10, 10))
    draw, font, y = ImageDraw.Draw(img), _font(font_size), margin
    for line in lines:
        draw.text((margin, y), line, font=font, fill=(230, 230, 230))
        y += font_size + spacing
        if y > height - margin:
            break
    return img


def make_gif(frames, path, prompt="", total=None, duration=0.08):
    """frames: [(t, decoded text with █ for [MASK]), ...] from diffusion_generate."""
    import imageio.v2 as imageio
    images = []
    for t, text in frames:
        lines = [f"masked diffusion · denoising step {total - t + 1 if total else t}/{total or ''}  (t = {t})", ""]
        lines += wrap(text)
        images.append(np.array(frame(lines)))
    images += [images[-1]] * 15           # hold the final text
    imageio.mimsave(path, images, duration=duration)
    return path
