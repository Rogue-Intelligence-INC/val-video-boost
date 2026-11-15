import json
from pathlib import Path

from PIL import Image, ImageDraw

R = Path(r"C:\Ai\rogue-products\val-video-boost\reports\fate_long_chain")
rep = json.loads((R / "report.json").read_text(encoding="utf-8"))
S, cols, rows_n, LH = 192, 6, 4, 16
sheet = Image.new("RGB", (S * cols, (S + LH) * rows_n), (18, 18, 18))
d = ImageDraw.Draw(sheet)
for i, r in enumerate(rep["rows"]):
    x, y = (i % cols) * S, (i // cols) * (S + LH)
    d.text((x + 4, y + 3), f"{r['segment']:02d} {r['scene']} {r['chosen_cos']}", fill=(255, 255, 0))
    sheet.paste(Image.open(r["chosen"]).convert("RGB").resize((S, S)), (x, y + LH))
sheet.save(R / "contact_long_chain.png")
print("ok")
