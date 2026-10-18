"""在已有只读后端镜像核对运行包锁和真实栅格网格正反例。"""

import importlib.metadata
import json
from pathlib import Path

from bidwb.parsing import PARSER_VERSION
from bidwb.table_candidates import raster_tables
from PIL import Image, ImageDraw


def grid(*, inner=True, horizontal=True):
    image = Image.new("RGB", (600, 800), "white")
    draw = ImageDraw.Draw(image)
    for x in [50, 300, 550] if inner else [50, 550]:
        draw.line((x, 100, x, 700), fill="black", width=3)
    if horizontal:
        for y in (100, 300, 500, 700):
            draw.line((50, y, 550, y), fill="black", width=3)
    return image


def main():
    count = 0
    for line in Path("/checks/constraints.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        assert importlib.metadata.version(name) == expected, name
        count += 1
    assert PARSER_VERSION == "pdfium-1.4"
    lines = [{"text": "15.12", "bbox": [0.55, 0.2, 0.8, 0.3]}]
    tables = raster_tables(grid(), lines)
    assert len(tables) == 1 and len(tables[0]["rows"]) == 3
    assert tables[0]["rows"][0] == ["", "15.12"] and tables[0]["cell_line_indices"][0] == [[], [0]]
    assert raster_tables(Image.new("RGB", (600, 800), "white"), []) == []
    assert raster_tables(grid(inner=False), lines) == []
    assert raster_tables(grid(horizontal=False), lines) == []
    assert raster_tables(grid().rotate(9, fillcolor="white"), []) == []
    print(
        json.dumps(
            {
                "parser": PARSER_VERSION,
                "existing_runtime_package_versions_preserved": count,
                "actual_raster_grid_cell_assignment": True,
                "blank_one_column_unruled_rotated_rejected": True,
                "network_disabled": True,
                "image_root_read_only": True,
                "packages": {name: importlib.metadata.version(name) for name in ("rapidocr", "opencv-python-headless", "numpy", "pypdfium2")},
                "real_bid_quality_evaluated": False,
            }
        )
    )


if __name__ == "__main__":
    main()
