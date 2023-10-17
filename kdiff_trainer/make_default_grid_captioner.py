import math
from PIL import ImageFont

from .make_captioned_grid import GridCaptioner, BBox, Typesetting, make_grid_captioner, make_typesetting
from .dimensions import Dimensions

def make_default_grid_captioner(
    font_path: str,
    sample_n: int,
    sample_size: Dimensions,
) -> GridCaptioner:
    pad = 8
    cell_pad = BBox(top=pad, left=pad, bottom=pad, right=pad)
    # abusing bottom padding to simulate a margin-bottom
    title_pad = BBox(top=pad, left=pad, bottom=pad*3, right=pad)

    cell_font: ImageFont = ImageFont.truetype(font_path, 25) if font_path else ImageFont.load_default()
    title_font: ImageFont = ImageFont.truetype(font_path, 50) if font_path else ImageFont.load_default()

    cols: int = math.ceil(sample_n ** .5)
    samp_w, samp_h = sample_size
    cell_type: Typesetting = make_typesetting(
        x_wrap_px=samp_w,
        font=cell_font,
        padding=cell_pad,
    )
    title_type: Typesetting = make_typesetting(
        x_wrap_px=samp_w*cols,
        font=title_font,
        padding=title_pad,
    )

    captioner: GridCaptioner = make_grid_captioner(
        cell_type=cell_type,
        cols=cols,
        samp_w=samp_w,
        samp_h=samp_h,
        title_type=title_type,
    )
    return captioner