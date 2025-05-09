# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict, List

import cv2
import numpy as np
import torch

from utils.geo import BoundaryBox
from .data import MapArea, MapLine, MapNode
from .parser import Groups


class Canvas:
    def __init__(self, bbox: BoundaryBox, ppm: float):
        self.bbox = bbox
        self.ppm = ppm
        self.scaling = bbox.size * ppm

        # My Hack
        # Need to round because sometimes we get 1280.0000000000005 and we want to round this
        # to 1280. since the 0.0000000000005 is a numerical instability issue not an actual number
        # So round to 8 decimals of precision and call it a day
        # Also for local maps sometimes we get 128.00006 which we want to round to 128
        # so safest thing to do is just round 3 decimals
        # These should be whole numbers anyways so fractions are not really an issue and are more of an
        # artifact of floating point math
        self.scaling = np.round(self.scaling, decimals=3)

        
        self.w, self.h = np.ceil(self.scaling).astype(int)

        self.clear()


    def get_data(self):

        bbox_data = self.bbox.get_data()

        ppm_data = np.zeros(bbox_data.shape[:-1])
        ppm_data = np.expand_dims(ppm_data, -1)
        ppm_data[...] = self.ppm

        return np.concatenate([bbox_data, ppm_data], axis=-1)

    @staticmethod
    def create_from_data(data):

        bbox = data[..., :-1]
        bbox = BoundaryBox.create_from_data(bbox)

        
        ppm = data[..., -1]
        for i in range(len(ppm.shape)):
            ppm = ppm[0]

        return Canvas(bbox, ppm)



    def clear(self):
        self.raster = np.zeros((self.h, self.w), np.uint8)

    def to_uv(self, xy: np.ndarray):
        xy = self.bbox.normalize(xy)
        xy[..., 1] = 1 - xy[..., 1]
        s = self.scaling
        if isinstance(xy, torch.Tensor):
            s = torch.from_numpy(s).to(xy)
        return xy * s - 0.5

    def to_xy(self, uv: np.ndarray):
        s = self.scaling
        if isinstance(uv, torch.Tensor):
            s = torch.from_numpy(s).to(uv)
        xy = (uv + 0.5) / s
        xy[..., 1] = 1 - xy[..., 1]
        return self.bbox.unnormalize(xy)

    def draw_polygon(self, xy: np.ndarray):
        uv = self.to_uv(xy)
        cv2.fillPoly(self.raster, uv[None].astype(np.int32), 255)

    def draw_multipolygon(self, xys: List[np.ndarray]):
        uvs = [self.to_uv(xy).round().astype(np.int32) for xy in xys]
        cv2.fillPoly(self.raster, uvs, 255)

    def draw_line(self, xy: np.ndarray, width: float = 1):
        uv = self.to_uv(xy)
        cv2.polylines(
            self.raster, uv[None].round().astype(np.int32), False, 255, thickness=width
        )

    def draw_cell(self, xy: np.ndarray):
        if not self.bbox.contains(xy):
            return
        uv = self.to_uv(xy)
        self.raster[tuple(uv.round().astype(int).T[::-1])] = 255


def render_raster_masks(
    nodes: List[MapNode],
    lines: List[MapLine],
    areas: List[MapArea],
    canvas: Canvas,
) -> Dict[str, np.ndarray]:
    all_groups = Groups.areas + Groups.ways + Groups.nodes
    masks = {k: np.zeros((canvas.h, canvas.w), np.uint8) for k in all_groups}

    for area in areas:
        canvas.raster = masks[area.group]
        outlines = area.outers + area.inners
        canvas.draw_multipolygon(outlines)
        if area.group == "building":
            canvas.raster = masks["building_outline"]
            for line in outlines:
                canvas.draw_line(line)

    for line in lines:
        canvas.raster = masks[line.group]
        canvas.draw_line(line.xy)

    for node in nodes:
        canvas.raster = masks[node.group]
        canvas.draw_cell(node.xy)

    return masks


def mask_to_idx(group2mask: Dict[str, np.ndarray], groups: List[str]) -> np.ndarray:
    masks = np.stack([group2mask[k] for k in groups]) > 0
    void = ~np.any(masks, 0)
    idx = np.argmax(masks, 0)
    idx = np.where(void, np.zeros_like(idx), idx + 1)  # add background
    return idx


def render_raster_map(masks: Dict[str, np.ndarray]) -> np.ndarray:
    areas = mask_to_idx(masks, Groups.areas)
    ways = mask_to_idx(masks, Groups.ways)
    nodes = mask_to_idx(masks, Groups.nodes)
    return np.stack([areas, ways, nodes])
