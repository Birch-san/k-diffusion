from PIL import Image, ImageFont, ImageDraw
from PIL.ImageFont import FreeTypeFont
from typing import List, NamedTuple, Protocol, Optional
from dataclasses import dataclass
from textwrap import TextWrapper
from functools import partial
import math
import numpy as np
from numpy.typing import NDArray

from .iteration.batched import batched

@dataclass
class FontMetrics:
  chartop: int
  charleft: int
  charw: int
  charh: int
  line_spacing: int

class BBox(NamedTuple):
  top: int
  left: int
  bottom: int
  right: int

@dataclass
class Typesetting:
  wrapper: TextWrapper
  font: FreeTypeFont
  font_metrics: FontMetrics
  padding: BBox

def get_font_metrics(font: ImageFont):
  tmp = Image.new("RGB", (100, 100))
  draw = ImageDraw.Draw(tmp)
  bbox = draw.textbbox((0, 0), "M", font=font)
  left, top, right, bottom = bbox
  charw = right-left
  charh = bottom-top

  bbox2 = draw.textbbox((0, 0), "M\nM", font=font)
  _, top_, _, bottom_ = bbox2
  line_spacing = (bottom_-top_)-2*charh

  return FontMetrics(
    chartop=top,
    charleft=left,
    charw=charw,
    charh=charh,
    line_spacing=line_spacing,
  )

def make_captioned_grid(
  cell_type: Typesetting,
  cols: int,
  samp_w: int,
  samp_h: int,
  imgs: List[Image.Image],
  captions: List[str],
  title_type: Optional[Typesetting] = None,
  title: Optional[str] = None,
) -> Image.Image:
  """
  Args:
    font `FreeTypeFont` for example: ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSansMono.ttf", 30)
    imgs `List[Image.Image]` images (which we will tabulate into rows & columns)
  Return:
    PIL `Image`; a grid, with captions
  """
  assert len(imgs) == len(captions)
  assert len(imgs) > 0
  if title is not None:
    assert title_type is not None

  rows: int = math.ceil(len(imgs)/cols)

  # compute all wrapped captions first, so that we can know max line counts in advance of allocating canvas
  wrappeds: List[List[str]] = []
  text_heights: List[int] = []
  for captions_ in batched(captions, cols):
    lines: List[List[str]] = [cell_type.wrapper.wrap(caption) for caption in captions_]
    line_counts: List[int] = [len(lines_) for lines_ in lines]
    max_line_count: int = max(line_counts)
    text_height: int = max_line_count*cell_type.font_metrics.charh+(max_line_count-1)* cell_type.font_metrics.line_spacing
    text_heights.append(text_height)

    wrappeds_: List[str] = ["\n".join(lines_) for lines_ in lines]
    wrappeds.append(wrappeds_)

  text_heights_np: NDArray = np.array(text_heights)
  text_heights_cumsum = np.roll(text_heights_np.cumsum(), 1)
  text_heights_cumsum[0] = 0

  if title is None:
    title_height = 0
  else:
    title_lines: List[str] = title_type.wrapper.wrap(title)
    title_line_count = len(title_lines)
    title_wrapped: str = "\n".join(title_lines)
    title_height: int = title_type.padding.top + title_type.padding.bottom + (title_line_count-1)*title_type.font_metrics.line_spacing + title_line_count * title_type.font_metrics.charh
  rows_height: int = text_heights_np.sum()+rows*(cell_type.padding.top+cell_type.padding.bottom+samp_h)
  img_width: int = samp_w*cols
  img_height: int = rows_height+title_height
  out = Image.new("RGB", (img_width, img_height), (255, 255, 255))
  d = ImageDraw.Draw(out)
  title_x_offset: int = title_type.padding.left - title_type.font_metrics.charleft
  title_y_offset: int = title_type.padding.top - title_type.font_metrics.chartop
  d.rectangle((0, 0, img_width, title_height), fill=(235, 235, 235))
  d.multiline_text((title_x_offset, title_y_offset), title_wrapped, font=title_type.font, fill=(0, 0, 0))

  cell_text_x_offset: int = cell_type.padding.left - cell_type.font_metrics.charleft
  cell_text_y_offset: int = cell_type.padding.top - cell_type.font_metrics.chartop
  for row_ix, (imgs_, wrappeds_, text_heights_cumsum_, current_text_height) in enumerate(zip(batched(imgs, cols), wrappeds, text_heights_cumsum, text_heights_np)):
    row_y: int = title_height + text_heights_cumsum_ + row_ix * (cell_type.padding.top + cell_type.padding.bottom + samp_h)
    cell_text_y: int = row_y + cell_text_y_offset
    img_y: int = row_y + cell_type.padding.top + current_text_height + cell_type.padding.bottom
    for col_ix, (img, wrapped) in enumerate(zip(imgs_, wrappeds_)):
      col_x: int = col_ix * samp_w
      cell_text_x: int = col_x + cell_text_x_offset
      d.multiline_text((cell_text_x, cell_text_y), wrapped, font=cell_type.font, fill=(0, 0, 0))
      out.paste(img, box=(col_x, img_y))

  return out

class GridCaptioner(Protocol):
  @staticmethod
  def __call__(
    imgs: List[Image.Image],
    captions: List[str],
    title: Optional[str] = None,
  ) -> Image.Image: ...

class TextWrapperFactory(Protocol):
  @staticmethod
  def __call__(
    width: int,
  ) -> TextWrapper: ...

def make_typesetting(
  font: FreeTypeFont,
  x_wrap_px: int,
  padding: BBox = BBox(0, 0, 0, 0),
  font_metrics: Optional[FontMetrics] = None,
  wrapper_factory: Optional[TextWrapper] = TextWrapper,
) -> Typesetting:
  if font_metrics is None:
    font_metrics: FontMetrics = get_font_metrics(font)
  textw = x_wrap_px - (padding.left + padding.right)
  wrap_at = textw//font_metrics.charw
  textwr: TextWrapper = wrapper_factory(width=wrap_at)
  cell_type = Typesetting(
    wrapper=textwr,
    font=font,
    font_metrics=font_metrics,
    padding=padding,
  )
  return cell_type

def make_grid_captioner(
  cell_type: Typesetting,
  cols: int,
  samp_w: int,
  samp_h: int,
  title_type: Optional[Typesetting] = None,
) -> GridCaptioner:
  return partial(
    make_captioned_grid,
    cell_type=cell_type,
    cols=cols,
    samp_w=samp_w,
    samp_h=samp_h,
    title_type=title_type,
  )