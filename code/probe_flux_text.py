import sys
from pathlib import Path

sys.path.insert(0, r"C:\Ai\Valhalla-main\tools\media_pipeline")
import module_patch  # noqa: E402
from test_flux_fast import gen, load_fast  # noqa: E402

triad = module_patch.load_export_patch(
    Path(r"C:\Ai\Valhalla-main\reports\media_pipeline\transformer_aligned\patches\triad.json")
)
pipe = load_fast()
out = Path(r"C:\Ai\Valhalla-main\reports\media_pipeline\flux")
prompt = 'a ceramic sign on a wall with the text "QUIET ROOM" written on it'
print("baseline", gen(pipe, prompt, out / "text_baseline.png", 7))
print("adaptive", gen(pipe, prompt, out / "text_adaptive.png", 7, patch=triad, strength=0.0166))
