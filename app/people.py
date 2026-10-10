"""多人圖：找出畫面裡的每個人，只留下目標角色（裁出目標，或用白色方塊蓋住其他人）。

用 deepghs 的動漫人物偵測與頭部偵測模型（YOLOv8 ONNX，MIT 授權）找出每個人和每個頭；
每個人裁下來用 CCIP 和參考圖比對，最像的那個是目標（差異要在門檻內，也不能更像排除參考圖）。
- 裁切（crop）：目標的框加一點邊，往內縮到不含別人的身體（不切到目標）和別人的頭（不切到目標的頭）。
  仍含別人的頭、比例太極端，或短邊小於設定的最小短邊就跳過。
- 白色方塊（mask）：別人的身體塗白（留下目標的框），別人的頭一律塗白（留下目標的頭）。
  和專案的「白色色塊」相同，匯入專案後會自動加上白色色塊的關鍵字。別人的頭和目標的頭重疊太多就跳過。
跳過的圖片保持原樣，交給使用者處理。規則用兩個篩選的 90 張多人截圖調過：約 7 成能自動處理。
找回：整張圖和參考圖不夠像（多人圖常這樣），但逐人比對時有一個人在門檻內，也算目標。
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

PERSON_MODEL = ("deepghs/anime_person_detection", "person_detect_v1.1_m/model.onnx", 0.35)  # 103 MB
HEAD_MODEL = ("deepghs/anime_head_detection", "head_detect_v2.0_s/model.onnx", 0.41)  # 44 MB
VERSION = 1  # 偵測或判斷方式改了就加一，舊的結果會重算
MODES = ("crop", "mask")
MARGIN = 0.06  # 裁切時目標的框往外多留的比例
HEAD_LIMIT = 0.15  # 別人的頭最多可以有這麼多在結果裡（面積比例）
BODY_LIMIT = 0.5  # 別人的身體最多可以有這麼多在裁切裡
MAX_RATIO = 3.0  # 裁切的長寬比上限
MIN_SIDE = 384  # 裁切後短邊的預設下限（像素；和專案放大的「太小」相同，低於這個放大也救不回細節）
AMBIGUOUS = 0.02  # 第二像的人也在門檻內、而且差不到這麼多時，不確定哪個是目標
SKIPPED = ("overlap", "small", "no_target", "ambiguous")  # 跳過的原因
TAGS = {**{k: f"people_{k}" for k in SKIPPED}, "recovered": "people_recovered"}  # 給 tag 篩選用的 tag


# ------------------------------------------------------------------ 偵測
def _nms(b: np.ndarray, s: np.ndarray, iou: float = 0.5) -> list[int]:
    order, keep = s.argsort()[::-1], []
    while len(order):
        i, rest = order[0], order[1:]
        keep.append(int(i))
        w = np.clip(np.minimum(b[i, 2], b[rest, 2]) - np.maximum(b[i, 0], b[rest, 0]), 0, None)
        h = np.clip(np.minimum(b[i, 3], b[rest, 3]) - np.maximum(b[i, 1], b[rest, 1]), 0, None)
        area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])  # noqa: E731
        order = rest[w * h / (area(b[i]) + area(b[rest]) - w * h + 1e-9) < iou]
    return keep


def detect(sess: Any, im: Image.Image, conf: float, size: int = 640) -> list[list[float]]:
    """YOLOv8：長邊縮到 size、長寬補到 32 的倍數，回傳 [x0, y0, x1, y1, 信心]（原圖座標）。"""
    w, h = im.size
    r = min(1.0, size / max(w, h))
    nw, nh = int(np.ceil(w * r / 32) * 32), int(np.ceil(h * r / 32) * 32)
    x = np.asarray(im.resize((nw, nh), Image.BILINEAR), np.float32).transpose(2, 0, 1)[None] / 255
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    if out.shape[0] > out.shape[1]:
        out = out.T
    score = out[4:].max(0)
    keep = score >= conf
    cx, cy, bw, bh = out[:4, keep]
    boxes, score = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1), score[keep]
    if not len(score):
        return []
    k = _nms(boxes, score)
    boxes = boxes[k] * np.array([w / nw, h / nh, w / nw, h / nh])
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return [[round(float(v), 1) for v in b] + [round(float(c), 3)] for b, c in zip(boxes, score[k])]


def analyze(im: Image.Image, person: Any, head: Any,
            score: Callable[[list[Image.Image]], list[tuple[float, float | None]]]) -> dict[str, Any]:
    """找出每個人，每個人和參考圖比對（score：裁下來的圖 → [(差異, 和排除參考圖的差異)]）；兩個人以上時再找每個頭。"""
    persons = detect(person, im, PERSON_MODEL[2])
    info: dict[str, Any] = {"v": VERSION, "w": im.width, "h": im.height, "persons": persons, "heads": [], "d": []}
    if persons:
        crops = [im.crop(tuple(int(round(v)) for v in p[:4])) for p in persons]
        info["d"] = [[round(d, 4), None if n is None else round(n, 4)] for d, n in score(crops)]
    if len(persons) >= 2:
        info["heads"] = detect(head, im, HEAD_MODEL[2])
    return info


# ------------------------------------------------------------------ 判斷
def _area(b: Any) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inside(a: Any, b: Any) -> float:
    """a 有多少（面積比例）在 b 裡面。"""
    inter = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return _area(inter) / max(_area(a), 1e-9)


def _expand(b: Any, w: int, h: int) -> list[float]:
    """加一點邊，長寬比限制在 1:2 – 2:1（往短的那邊補），不超出畫面。"""
    x0, y0, x1, y1 = b[:4]
    bw, bh = x1 - x0, y1 - y0
    x0, x1, y0, y1 = x0 - bw * MARGIN, x1 + bw * MARGIN, y0 - bh * MARGIN, y1 + bh * MARGIN
    bw, bh = x1 - x0, y1 - y0
    if bh > 2 * bw:
        x0, x1 = x0 - (bh / 2 - bw) / 2, x1 + (bh / 2 - bw) / 2
    elif bw > 2 * bh:
        y0, y1 = y0 - (bw / 2 - bh) / 2, y1 + (bw / 2 - bh) / 2
    return [max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)]


def _trim(c: list[float], keep: Any, other: Any) -> list[float]:
    """把 c 的一邊往內縮到不含 other（看 other 在 keep 的哪一邊），但不縮進 keep。"""
    x0, y0, x1, y1 = c
    kx, ky, ox, oy = (keep[0] + keep[2]) / 2, (keep[1] + keep[3]) / 2, (other[0] + other[2]) / 2, (other[1] + other[3]) / 2
    dx = abs(ox - kx) / max(1.0, (keep[2] - keep[0] + other[2] - other[0]) / 2)
    dy = abs(oy - ky) / max(1.0, (keep[3] - keep[1] + other[3] - other[1]) / 2)
    if dx >= dy:
        if ox > kx:
            x1 = max(keep[2], min(x1, other[0]))
        else:
            x0 = min(keep[0], max(x0, other[2]))
    elif oy > ky:
        y1 = max(keep[3], min(y1, other[1]))
    else:
        y0 = min(keep[1], max(y0, other[3]))
    return [x0, y0, x1, y1]


def _short(box: Any) -> int:
    """裁切框的短邊（像素，四捨五入；網頁用同一個值判斷「太小」）。"""
    return round(min(box[2] - box[0], box[3] - box[1]))


def target(info: dict[str, Any]) -> int | None:
    """最像參考圖的那個人。是不是真的是目標由 judge() 依門檻判斷。"""
    d = info.get("d") or []
    return int(np.argmin([x[0] for x in d])) if d else None


def judge(info: dict[str, Any], threshold: float) -> str:
    """逐人比對：found（最像的人在門檻內）/ no_target（沒有人夠像，或更像排除參考圖）/ ambiguous（兩個人都很像）。"""
    d = sorted(info.get("d") or [], key=lambda x: x[0])
    if not d or d[0][0] > threshold or (d[0][1] is not None and d[0][1] <= d[0][0]):
        return "no_target"
    if len(d) > 1 and d[1][0] <= threshold and d[1][0] - d[0][0] < AMBIGUOUS:
        return "ambiguous"
    return "found"


def _parts(info: dict[str, Any], t: int) -> tuple[list[float], list[float], list[list[float]], list[list[float]]]:
    """目標的框、目標的頭、別人的身體、別人的頭。每個頭屬於最包住它的那個人（一樣時給比較小的框）。"""
    persons = [p[:4] for p in info["persons"]]
    body = persons[t]
    mine, others = [], []
    for hd in (x[:4] for x in info["heads"]):
        owner = max(range(len(persons)), key=lambda i: (round(_inside(hd, persons[i]), 2), -_area(persons[i])))
        (mine if owner == t and _inside(hd, persons[owner]) >= 0.5 else others).append(hd)
    face = max(mine, key=_area) if mine else body
    others += [hd for hd in mine if hd is not face]  # 目標框裡的第二個頭是別人的
    return body, face, [p for i, p in enumerate(persons) if i != t], others


def geometry(info: dict[str, Any], mode: str) -> tuple[str, list[float] | None]:
    """以最像的人為目標時，這個模式能不能處理：ok（裁切框）/ overlap。不看門檻和最小短邊。"""
    t = target(info)
    if t is None or len(info["persons"]) < 2:
        return "single", None
    body, face, bodies, heads = _parts(info, t)
    if mode == "mask":
        return ("overlap", None) if any(_inside(hd, face) > HEAD_LIMIT for hd in heads) else ("ok", body)
    c = _expand(body, info["w"], info["h"])
    for o in bodies:  # 先避開別人的身體（不切到目標）
        c = _trim(c, body, o)
    for o in heads:  # 再避開別人的頭（可以切到目標的身體，但不切到目標的頭）
        if _inside(o, c) > 0:
            c = _trim(c, face, o)
    w, h = c[2] - c[0], c[3] - c[1]
    if w <= 0 or h <= 0 or any(_inside(o, c) > HEAD_LIMIT for o in heads) \
            or any(_inside(o, c) > BODY_LIMIT for o in bodies) or max(w / h, h / w) > MAX_RATIO:
        return "overlap", c
    return "ok", c


def summary(info: dict[str, Any]) -> dict[str, Any]:
    """給網頁的精簡結果：人數、每個人的差異，以及兩種模式的處理結果和裁切後的短邊（網頁依門檻自己判斷狀態）。"""
    out: dict[str, Any] = {"n": len(info.get("persons") or [])}
    if info.get("d"):
        out["d"] = info["d"]
    if out["n"] >= 2:
        for mode in MODES:
            out[mode], box = geometry(info, mode)
            if mode == "crop" and out[mode] == "ok":
                out["side"] = _short(box)
    return out


def state(info: dict[str, Any] | None, mode: str, threshold: float, min_side: float = MIN_SIDE) -> str | None:
    """目前門檻下的狀態：None（還沒偵測）/ single / no_target / ambiguous / overlap / small / ok。"""
    if not info:
        return None
    if len(info["persons"]) < 2:
        return "single"
    found = judge(info, threshold)
    if found != "found":
        return found
    status, box = geometry(info, mode)
    if status == "ok" and mode == "crop" and _short(box) < min_side:
        return "small"
    return status


# ------------------------------------------------------------------ 輸出
def render(im: Image.Image, info: dict[str, Any], mode: str) -> Image.Image:
    """處理後的圖（呼叫前先確認 geometry 是 ok）。"""
    status, box = geometry(info, mode)
    if status != "ok" or box is None:
        return im
    if mode == "crop":
        return im.crop(tuple(int(round(v)) for v in box))
    body, face, bodies, heads = _parts(info, target(info))
    out = im.copy()
    draw = ImageDraw.Draw(out)
    keep = tuple(int(round(v)) for v in body)
    for b in bodies:
        draw.rectangle([int(round(v)) for v in b], fill="white")
    out.paste(im.crop(keep), keep[:2])  # 目標的框不塗
    face_box = tuple(int(round(v)) for v in face)
    for hd in heads:
        draw.rectangle([int(round(v)) for v in hd], fill="white")
    out.paste(im.crop(face_box), face_box[:2])  # 目標的頭不塗
    return out
