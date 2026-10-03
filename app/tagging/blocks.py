"""偵測圖片中遮擋用的白色色塊（例如為了只留一個人，用純白矩形蓋掉其他人）。

把圖縮成粗網格，幾乎全白的格子算白，再找最大的全白矩形。夠大的矩形才算色塊，
自然的白色區域（白牆、過曝的窗戶）通常有紋理或不夠純白，不會被當成色塊。
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

GRID_W, GRID_H = 96, 54  # 粗網格大小（與圖片比例無關，矩形以 0–1 的比例回傳）
CELL = 4  # 每格取樣 4×4 像素
WHITE_MIN = 248  # RGB 三個通道都 ≥ 這個值才算白
CELL_WHITE = 0.97  # 格子裡白色像素的比例
MIN_AREA = 0.04  # 矩形至少佔整張圖的比例
MIN_W, MIN_H = 0.08, 0.15  # 矩形最小寬高（比例）
MAX_RECTS = 4


def _largest_rect(grid: np.ndarray) -> tuple[int, int, int, int, int]:
    """二值矩陣中最大的全 True 矩形：(面積, x, y, w, h)。"""
    h, w = grid.shape
    heights = np.zeros(w, dtype=int)
    best = (0, 0, 0, 0, 0)
    for y in range(h):
        heights = np.where(grid[y], heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for x in range(w + 1):
            cur = int(heights[x]) if x < w else 0
            start = x
            while stack and stack[-1][1] >= cur:
                start, hh = stack.pop()
                area = hh * (x - start)
                if area > best[0]:
                    best = (area, start, y - hh + 1, x - start, hh)
            stack.append((start, cur))
    return best


def detect_white_blocks(im: Image.Image) -> dict[str, Any]:
    """回傳 {"rects": [[x, y, w, h], …]（0–1 比例）, "area": 色塊總面積比例}。"""
    small = im.convert("RGB").resize((GRID_W * CELL, GRID_H * CELL), Image.BOX)
    a = np.asarray(small, dtype=np.uint8)
    white = a.min(axis=2) >= WHITE_MIN
    grid = white.reshape(GRID_H, CELL, GRID_W, CELL).mean(axis=(1, 3)) >= CELL_WHITE
    rects: list[list[float]] = []
    total = GRID_W * GRID_H
    area_sum = 0
    for _ in range(MAX_RECTS):
        area, x, y, w, h = _largest_rect(grid)
        if area / total < MIN_AREA or w < GRID_W * MIN_W or h < GRID_H * MIN_H:
            break
        rects.append([round(x / GRID_W, 3), round(y / GRID_H, 3), round(w / GRID_W, 3), round(h / GRID_H, 3)])
        area_sum += area
        grid[y:y + h, x:x + w] = False
    return {"rects": rects, "area": round(area_sum / total, 3)}


def has_blocks(blocks: dict[str, Any] | None) -> bool:
    """手動標記（override）優先，否則看偵測結果。"""
    if not blocks:
        return False
    if blocks.get("override") is not None:
        return bool(blocks["override"])
    return bool(blocks.get("rects"))
