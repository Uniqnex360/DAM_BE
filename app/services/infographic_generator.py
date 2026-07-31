import io
import base64
import logging
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageColor
import numpy as np

try:
    from .embedded_fonts import BOLD_TTF_B64, REGULAR_TTF_B64
except ImportError:
    from embedded_fonts import BOLD_TTF_B64, REGULAR_TTF_B64

logger = logging.getLogger(__name__)


class InfographicGenerator:

    CARD_WIDTH = 1200
    CARD_HEIGHT = 1200
    PADDING = 60
    ACCENT = "#F9A800"  # yellow accent, swap via analysis dominant_colors if wanted

    def __init__(self):
        self._analysis_cache = {}

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    async def generate(self, image_bytes: bytes, product_name: str, options: dict = {}) -> bytes:
        analysis = await self._analyze_product(image_bytes, product_name)
        card = self._create_card(image_bytes, analysis, options)
        output = io.BytesIO()
        card.save(output, format="PNG", quality=95)
        return output.getvalue()

    # ------------------------------------------------------------------ #
    # Analysis (unchanged behavior)
    # ------------------------------------------------------------------ #
    async def _analyze_product(self, image_bytes: bytes, fallback_name: str) -> dict:
        try:
            return await self._vision_ai_analysis(image_bytes)
        except Exception as e:
            logger.warning(f"Vision AI failed, using basic analysis: {e}")
            return self._basic_analysis(image_bytes, fallback_name)

    def _pick_contrast_background(self, analysis: dict, options: dict) -> str:
        """
        Pick a background color that contrasts with the product's dominant color.
        Uses complementary color logic with brightness adjustment.
        """
        # If user explicitly passed a color, use it
        user_color = (options or {}).get("bg_color")
        if user_color:
            return user_color

        # Get the primary dominant color from analysis
        dominant_colors = analysis.get("dominant_colors", [])
        if not dominant_colors:
            return "#111111"

        primary = dominant_colors[0].lstrip("#")

        # Convert hex to RGB
        r, g, b = int(primary[0:2], 16), int(primary[2:4], 16), int(primary[4:6], 16)

        # Calculate brightness (perceived luminance)
        brightness = 0.299 * r + 0.587 * g + 0.114 * b

        # Calculate complementary color
        comp_r = 255 - r
        comp_g = 255 - g
        comp_b = 255 - b

        # If product is bright, use dark background; if dark, use light background
        if brightness > 128:
            # Product is bright → dark background
            # Darken the complementary color
            bg_r = int(comp_r * 0.25)
            bg_g = int(comp_g * 0.25)
            bg_b = int(comp_b * 0.25)
        else:
            # Product is dark → light background with dark text area
            # Darken for contrast
            bg_r = int(r * 0.15) if r < 128 else int(comp_r * 0.3)
            bg_g = int(g * 0.15) if g < 128 else int(comp_g * 0.3)
            bg_b = int(b * 0.15) if b < 128 else int(comp_b * 0.3)

        # Ensure it's dark enough for white text readability
        bg_brightness = 0.299 * bg_r + 0.587 * bg_g + 0.114 * bg_b
        if bg_brightness > 80:
            # Too light, darken
            bg_r = int(bg_r * 0.4)
            bg_g = int(bg_g * 0.4)
            bg_b = int(bg_b * 0.4)

        return f"#{bg_r:02x}{bg_g:02x}{bg_b:02x}"

    async def _vision_ai_analysis(self, image_bytes: bytes) -> dict:
        import os
        import httpx
        import json

        image_b64 = base64.b64encode(image_bytes).decode()
        prompt = """Analyze this product image and return ONLY valid JSON:
    {
        "brand": "e.g., DEWALT, Nike, Apple",
        "product_type": "e.g., Cordless Jig Saw, Running Shoe",
        "model": "e.g., DCS331B, Air Max 270",
        "tagline": "e.g., Powerful. Precise. Portable.",
        "subtext": "e.g., Take on any cut with confidence. Built for performance and control.",
        "badges": ["BARE TOOL"],
        "specifications": ["UP TO 3,000 SPM", "4-POSITION ORBITAL", "TOOL-FREE BLADE CHANGE"],
        "dominant_colors": ["#HEX1", "#HEX2", "#HEX3"],
        "material": "e.g., Metal/Plastic, Leather, Cotton",
        "category": "e.g., Power Tools, Footwear, Accessories",
        "key_features": [
            {"title": "Powerful Performance", "desc": "High-performance motor delivers up to 3,000 SPM for fast, efficient cutting."},
            {"title": "Precise Control", "desc": "Variable speed trigger and 4-position orbital action for smooth, accurate cuts."},
            {"title": "Compact & Lightweight", "desc": "Ergonomic design for comfortable handling and ease of use in tight spaces."}
        ],
        "disclaimer": "e.g., *Maximum initial battery voltage (measured without a workload) is 20 volts. Nominal voltage is 18."
    }"""

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "gpt-4o",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]
                    }],
                    "max_tokens": 700,
                }
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
            return json.loads(content)

    def _basic_analysis(self, image_bytes: bytes, product_name: str) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        try:
            from sklearn.cluster import KMeans
            image_small = image.resize((150, 150))
            pixels = np.array(image_small).reshape(-1, 3)
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            dominant_colors = kmeans.cluster_centers_.astype(int)
            color_hexes = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in dominant_colors[:3]]
        except Exception:
            image_small = image.resize((50, 50))
            pixels = np.array(image_small).reshape(-1, 3)
            from collections import Counter
            color_counts = Counter()
            for pixel in pixels[::5]:
                quantized = tuple((p // 32) * 32 for p in pixel)
                color_counts[quantized] += 1
            dominant = color_counts.most_common(3)
            color_hexes = [f"#{r:02x}{g:02x}{b:02x}" for (r, g, b), _ in dominant]

        is_furniture = any(k in product_name.lower() for k in ("chair", "table", "sofa"))

        return {
            "product_type": product_name.rsplit(".", 1)[0].replace("_", " ").title(),
            "dominant_colors": color_hexes,
            "material": "Wood / Fabric" if is_furniture else "See details",
            "category": "Furniture" if is_furniture else "General",
            "brand": "UNKNOWN",
            "model": "",
            "tagline": "Comfort and style in one.",
            "subtext": "Built for everyday performance and lasting quality.",
            "badges": [],
            "specifications": ["ERGONOMIC DESIGN", "REMOVABLE COVER", "DURABLE FRAME"],
            "key_features": [
                {"title": "Comfort First", "desc": "Cushioned seat built for all-day comfort."},
                {"title": "Modern Design", "desc": "Clean lines that fit any room."},
                {"title": "Built to Last", "desc": "Durable frame and construction."},
            ],
            "disclaimer": "",
        }

    # ------------------------------------------------------------------ #
    # Card rendering
    # ------------------------------------------------------------------ #
    def _load_fonts(self):
        # Embedded fonts guarantee identical, readable rendering on every host
        # regardless of which OS font packages happen to be installed there.
        bold_bytes = base64.b64decode(BOLD_TTF_B64)
        reg_bytes = base64.b64decode(REGULAR_TTF_B64)

        def load(data, size):
            return ImageFont.truetype(io.BytesIO(data), size)

        return {
            "title": load(bold_bytes, 80),
            "subtitle": load(bold_bytes, 52),
            "tagline": load(bold_bytes, 34),
            "body": load(reg_bytes, 26),
            "small": load(reg_bytes, 20),
            "badge": load(bold_bytes, 23),
            "feat_title": load(bold_bytes, 29),
            "stat_value": load(bold_bytes, 52),
            "stat_label": load(bold_bytes, 19),
            "brand": load(bold_bytes, 38),
            "_bold_bytes": bold_bytes,
            "_reg_bytes": reg_bytes,
        }

    def _lighten_color(self, rgb, factor=0.55):
        """Blend a color toward white by `factor` (0 = unchanged, 1 = white).
        Used to get a lighter tint of the poster's bg_color for the photo
        area, so it reads as coordinated with the poster but isn't the exact
        same dark shade as the text panel."""
        r, g, b = rgb
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        return (r, g, b)

    def _remove_background(self, img: Image.Image, target_rgb) -> Image.Image:
        """Cut the product out from its background and place it on target_rgb.
        Tries rembg (real ML segmentation) first since it handles busy/
        lifestyle photos correctly; falls back to corner-sampled flat-color
        keying (for simple studio shots) if rembg isn't installed or fails."""
        target = ImageColor.getrgb(target_rgb) if isinstance(target_rgb, str) else target_rgb

        try:
            from rembg import remove
            cutout = remove(img)  # RGBA, subject opaque, bg transparent
            cutout = cutout.convert("RGBA")
            canvas = Image.new("RGB", cutout.size, target)
            canvas.paste(cutout, (0, 0), cutout)
            return canvas
        except Exception as e:
            logger.warning(f"rembg unavailable/failed, falling back to flat-bg keying: {e}")
            return self._key_out_background(img, target)

    def _key_out_background(self, img: Image.Image, target_rgb, threshold=42, feather=28):
        """Fallback for when rembg isn't available: if the photo has a flat,
        uniform background (typical studio product shot), replace it with
        target_rgb. Lifestyle photos (corners disagree) are left untouched,
        since there's no single background color to key out safely."""
        arr = np.array(img.convert("RGB")).astype(np.float64)
        h, w = arr.shape[:2]
        patch = max(4, min(h, w) // 25)

        corners = np.array([
            arr[0:patch, 0:patch].reshape(-1, 3).mean(axis=0),
            arr[0:patch, w - patch:w].reshape(-1, 3).mean(axis=0),
            arr[h - patch:h, 0:patch].reshape(-1, 3).mean(axis=0),
            arr[h - patch:h, w - patch:w].reshape(-1, 3).mean(axis=0),
        ])
        max_corner_spread = np.max(np.linalg.norm(corners - corners.mean(axis=0), axis=1))
        if max_corner_spread > 20:
            return img

        key_color = corners.mean(axis=0)
        dist = np.linalg.norm(arr - key_color, axis=2)
        alpha = np.clip((dist - threshold) / feather, 0, 1)
        target = np.array(
            ImageColor.getrgb(target_rgb) if isinstance(target_rgb, str) else target_rgb,
            dtype=np.float64,
        )
        out = arr * alpha[..., None] + target * (1 - alpha[..., None])
        return Image.fromarray(out.astype(np.uint8))

    def _sample_image_bg_color(self, img: Image.Image, fallback=(245, 245, 245)):
        """Sample the product photo's own corners to find its natural
        background color, so any letterbox padding blends with the photo
        itself rather than looking like a foreign color slapped behind it."""
        arr = np.array(img.convert("RGB")).astype(np.float64)
        h, w = arr.shape[:2]
        patch = max(4, min(h, w) // 25)
        corners = np.array([
            arr[0:patch, 0:patch].reshape(-1, 3).mean(axis=0),
            arr[0:patch, w - patch:w].reshape(-1, 3).mean(axis=0),
            arr[h - patch:h, 0:patch].reshape(-1, 3).mean(axis=0),
            arr[h - patch:h, w - patch:w].reshape(-1, 3).mean(axis=0),
        ])
        spread = np.max(np.linalg.norm(corners - corners.mean(axis=0), axis=1))
        if spread > 30:
            # busy/lifestyle photo, corners disagree — no single "bg color"
            return fallback
        avg = corners.mean(axis=0)
        return tuple(int(c) for c in avg)

    def _contain_fit(self, img: Image.Image, w: int, h: int, bg_color=(17, 17, 17)) -> Image.Image:
        """Scale image to fit fully within w x h with no cropping (letterbox),
        centered on a solid background so the whole product stays visible."""
        src_ratio = img.width / img.height
        dst_ratio = w / h
        if src_ratio > dst_ratio:
            new_w = w
            new_h = int(w / src_ratio)
        else:
            new_h = h
            new_w = int(h * src_ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (w, h), bg_color)
        canvas.paste(img, ((w - new_w) // 2, (h - new_h) // 2))
        return canvas

    def _cover_fit(self, img: Image.Image, w: int, h: int) -> Image.Image:
        """Crop/scale image to fully cover w x h (like CSS background-size: cover)."""
        src_ratio = img.width / img.height
        dst_ratio = w / h
        if src_ratio > dst_ratio:
            new_h = h
            new_w = int(h * src_ratio)
        else:
            new_w = w
            new_h = int(w / src_ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        return img.crop((left, top, left + w, top + h))

    def _fit_wrapped(self, draw, text, font_bytes, start_size, max_width, min_size=16, max_lines=None):
        """Wrap text to max_width, shrinking font size if any line (including
        unbreakable single words) would still overflow. Guarantees nothing
        ever crosses max_width, however long the word or however small the panel."""
        size = start_size
        while size >= min_size:
            font = ImageFont.truetype(io.BytesIO(font_bytes), size)
            lines = self._wrap_text(draw, text, font, max_width)
            if max_lines:
                lines = lines[:max_lines]
            widest = max((draw.textlength(l, font=font) for l in lines), default=0)
            if widest <= max_width or size == min_size:
                return lines, font
            size -= 4
        return lines, font

    def _wrap_text(self, draw, text, font, max_width):
        words = text.split()
        lines, cur = [], ""
        for word in words:
            trial = (cur + " " + word).strip()
            if draw.textlength(trial, font=font) <= max_width:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines

    def _draw_icon(self, draw, cx, cy, r, kind, color):
        """Simple generic line-icon inside a circle outline: gauge / target / feature dot."""
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=3)
        if kind == 0:  # gauge / speedometer style
            draw.arc([cx - r + 8, cy - r + 8, cx + r - 8, cy + r - 8], 200, 340, fill=color, width=3)
            draw.line([cx, cy, cx + r * 0.45, cy - r * 0.35], fill=color, width=3)
        elif kind == 1:  # target
            draw.ellipse([cx - r * 0.55, cy - r * 0.55, cx + r * 0.55, cy + r * 0.55], outline=color, width=2)
            draw.ellipse([cx - r * 0.18, cy - r * 0.18, cx + r * 0.18, cy + r * 0.18], fill=color)
        else:  # compact / spark
            draw.line([cx - r * 0.5, cy, cx + r * 0.5, cy], fill=color, width=3)
            draw.line([cx, cy - r * 0.5, cx, cy + r * 0.5], fill=color, width=3)
            draw.line([cx - r * 0.35, cy - r * 0.35, cx + r * 0.35, cy + r * 0.35], fill=color, width=2)
            draw.line([cx - r * 0.35, cy + r * 0.35, cx + r * 0.35, cy - r * 0.35], fill=color, width=2)

    def _sanitize_analysis(self, analysis: dict) -> dict:
        """Vision AI can return explicit JSON nulls for any field (key present,
        value None), which bypasses dict.get(key, default). Coerce everything
        to safe types/defaults up front so rendering never sees None."""
        analysis = analysis or {}

        def s(key, default=""):
            v = analysis.get(key)
            return v if isinstance(v, str) else default

        def lst(key):
            v = analysis.get(key)
            return v if isinstance(v, list) else []

        return {
            "brand": s("brand"),
            "product_type": s("product_type", "Product"),
            "model": s("model"),
            "tagline": s("tagline"),
            "subtext": s("subtext"),
            "badges": [b for b in lst("badges") if isinstance(b, str)],
            "category": s("category"),
            "material": s("material"),
            "specifications": [x for x in lst("specifications") if isinstance(x, str)],
            "key_features": [f for f in lst("key_features") if isinstance(f, (dict, str))],
            "disclaimer": s("disclaimer"),
            "dominant_colors": [c for c in lst("dominant_colors") if isinstance(c, str)] or [self.ACCENT],
        }

    def _create_card(self, image_bytes: bytes, analysis: dict, options: dict) -> Image.Image:
        analysis = self._sanitize_analysis(analysis)
        W, H = self.CARD_WIDTH, self.CARD_HEIGHT
        F = self._load_fonts()
        accent = analysis.get("dominant_colors", [self.ACCENT])
        accent_color = accent[0] if accent else self.ACCENT

        bar_h = 160  # bottom stat bar, reserved from total height
        panel_w = int(W * 0.42)  # solid text panel width; rest is pure photo
        bg_color = self._pick_contrast_background(analysis, options)
        bg_rgb = ImageColor.getrgb(bg_color) if isinstance(bg_color, str) else bg_color

        card = Image.new("RGB", (W, H), bg_color)

        # ---- 1. Right side: product photo, own background preserved ----
        photo_w = W - panel_w
        photo_h = H - bar_h

        product_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Actually remove the product's own background (not just color-match
        # the letterbox) and composite it onto a LIGHTER TINT of the poster's
        # bg_color — coordinated with the poster, but not the same dark shade
        # as the text panel, so the photo area still reads distinctly.
        photo_bg_rgb = self._lighten_color(bg_rgb, factor=0.55)
        product_cutout = self._remove_background(product_img, photo_bg_rgb)
        photo = self._contain_fit(product_cutout, photo_w, photo_h, bg_color=photo_bg_rgb)
        card.paste(photo, (panel_w, 0))

        # ---- 2. Left side: solid panel for all text (image never touched) ----
        draw = ImageDraw.Draw(card)
        draw.rectangle([0, 0, panel_w, H], fill=bg_color)
        # thin divider between panel and photo
        draw.line([panel_w, 0, panel_w, photo_h], fill="#333333", width=2)

        # ---- 2b. Grey frame lines around the photo to set it off cleanly ----
        frame_margin = 14
        draw.rectangle(
            [panel_w + frame_margin, frame_margin, W - frame_margin, photo_h - frame_margin],
            outline="#8a8a8a",
            width=2,
        )

        content_w = panel_w - self.PADDING - 30  # extra right margin, keeps text off divider
        x0 = self.PADDING
        y = self.PADDING

        # ---- 3. Brand badge ----
        brand = analysis.get("brand", "").upper()
        if brand and brand.strip().lower() not in {"unknown", "n/a", "none", "tbd"}:
            bw = draw.textlength(brand, font=F["brand"]) + 44
            draw.rounded_rectangle([x0, y, x0 + bw, y + 68], radius=6, fill=accent_color)
            draw.text((x0 + 22, y + 14), brand, fill="#111111", font=F["brand"])
            y += 68 + 34

        # ---- 4. Product type headline (white, big) ----
        title = analysis.get("product_type", "Product").upper()
        title_lines, title_font = self._fit_wrapped(draw, title, F["_bold_bytes"], 80, content_w, min_size=40)
        for line in title_lines:
            draw.text((x0, y), line, fill="#FFFFFF", font=title_font)
            y += int(title_font.size * 1.12)

        y += 6

        # ---- 5. Small badges row: model + extra badges ----
        badges = []
        model = analysis.get("model", "")
        placeholder = {"", "unknown", "n/a", "none", "tbd"}
        if model.strip().lower() not in placeholder and model.strip().lower() != brand.strip().lower():
            badges.append((model, accent_color, "#111111"))
        for b in analysis.get("badges", []):
            if b.strip().lower() not in placeholder:
                badges.append((b.upper(), "#2a2a2a", "#FFFFFF"))
        if badges:
            bx = x0
            for text, bg_c, fg_c in badges:
                bw = draw.textlength(text, font=F["badge"]) + 34
                draw.rounded_rectangle([bx, y, bx + bw, y + 48], radius=5, fill=bg_c)
                draw.text((bx + 17, y + 11), text, fill=fg_c, font=F["badge"])
                bx += bw + 18
            y += 48 + 30

        # ---- 6. Tagline (bold white) ----
        tagline = analysis.get("tagline", "")
        if tagline:
            tag_lines, tag_font = self._fit_wrapped(draw, tagline, F["_bold_bytes"], 34, content_w, min_size=20)
            for line in tag_lines:
                draw.text((x0, y), line, fill="#FFFFFF", font=tag_font)
                y += int(tag_font.size * 1.25)

        # ---- 7. Subtext (smaller gray/white, wrapped) ----
        subtext = analysis.get("subtext", "")
        if subtext:
            for line in self._wrap_text(draw, subtext, F["body"], content_w):
                draw.text((x0, y), line, fill="#CCCCCC", font=F["body"])
                y += 36
            y += 12

        # ---- 8. Divider ----
        draw.line([x0, y, x0 + content_w, y], fill="#555555", width=1)
        y += 35

        # ---- 9. Feature rows with icon circles ----
        features = analysis.get("key_features", [])
        for i, feat in enumerate(features[:3]):
            if isinstance(feat, dict):
                f_title = str(feat.get("title") or "").upper()
                f_desc = str(feat.get("desc") or "")
            else:
                f_title = str(feat or "").upper()
                f_desc = ""
            r = 30
            cx, cy = x0 + r, y + r
            self._draw_icon(draw, cx, cy, r, i % 3, accent_color)

            tx = x0 + r * 2 + 24
            feat_col_w = content_w - r * 2 - 24
            title_lines, title_font2 = self._fit_wrapped(
                draw, f_title, F["_bold_bytes"], 29, feat_col_w, min_size=18, max_lines=2
            )
            ty = y
            for line in title_lines:
                draw.text((tx, ty), line, fill="#FFFFFF", font=title_font2)
                ty += int(title_font2.size * 1.2)
            if f_desc:
                for line in self._wrap_text(draw, f_desc, F["small"], feat_col_w):
                    draw.text((tx, ty), line, fill="#AAAAAA", font=F["small"])
                    ty += 27
            y = max(ty, y + r * 2) + 28

        # ---- 10. Bottom stat bar (full width, 3 columns w/ dividers) ----
        specs = analysis.get("specifications", [])[:3]
        bar_y0 = H - bar_h
        draw.rectangle([0, bar_y0, W, H], fill=bg_color)

        disclaimer = analysis.get("disclaimer", "")
        reserved_right = 320 if disclaimer else 0
        bar_content_w = W - reserved_right

        if specs:
            col_w = bar_content_w / len(specs)
            for i, spec in enumerate(specs):
                spec = str(spec)
                # split into "VALUE" + "label" if there's a natural split point
                parts = spec.split(" ", 1)
                value = parts[0]
                label = parts[1] if len(parts) > 1 else ""
                cx0 = int(i * col_w) + 40
                vy = bar_y0 + 34
                draw.text((cx0, vy), value, fill=accent_color, font=F["stat_value"])
                if label:
                    draw.text((cx0, vy + 60), label.upper(), fill="#EEEEEE", font=F["stat_label"])
                if i > 0:
                    draw.line([int(i * col_w), bar_y0 + 24, int(i * col_w), H - 24], fill="#444444", width=1)

        # ---- 11. Disclaimer / footer text, reserved bottom-right slot ----
        if disclaimer:
            lines = self._wrap_text(draw, disclaimer, F["small"], reserved_right - 40)[:3]
            start_y = H - 24 - 24 * len(lines)
            for j, line in enumerate(lines):
                tw = draw.textlength(line, font=F["small"])
                draw.text((W - self.PADDING - tw, start_y + j * 24), line, fill="#888888", font=F["small"])

        return card.convert("RGB")