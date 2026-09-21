"""Stateless, original-based image preparation shared by Colour and B&W web UI.

Positive angles mean clockwise. Shrink once from the supplied original, then
rotate onto an expanded white canvas; no plot/legend coordinates or job files
are silently modified. PNG encoding preserves the resulting raster losslessly.
"""
from __future__ import annotations

import io
import math
import warnings

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


MAX_PIXELS = 100_000_000
MAX_SIDE = 32_768
MAX_UPLOAD_BYTES = 100 * 1024**2


class ImagePreparationError(ValueError):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


def _number(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ImagePreparationError(f'{name} must be a finite number') from error
    if not math.isfinite(value):
        raise ImagePreparationError(f'{name} must be a finite number')
    return value


def _check_dimensions(width, height, stage):
    if width < 1 or height < 1:
        raise ImagePreparationError(f'{stage} image must have positive dimensions')
    if width > MAX_SIDE or height > MAX_SIDE or width * height > MAX_PIXELS:
        raise ImagePreparationError(
            f'{stage} image exceeds the {MAX_PIXELS:,}-pixel or {MAX_SIDE:,}-pixel side limit', 413)


def _flatten_alpha_white(image):
    if image.ndim != 3 or image.shape[2] != 4:
        return image
    maximum = int(np.iinfo(image.dtype).max)
    alpha = image[:, :, 3].astype(np.uint32)
    output = np.empty((*image.shape[:2], 3), dtype=image.dtype)
    # Integer half-up compositing avoids floating ambiguity and dark halos at
    # transparent edges. uint32 safely contains the products for 16-bit PNGs.
    for channel in range(3):
        foreground = image[:, :, channel].astype(np.uint32)
        combined = foreground * alpha + (maximum - alpha) * maximum
        output[:, :, channel] = ((combined + maximum // 2) // maximum).astype(image.dtype)
    return output


def _normalize_exif_orientation(image, orientation):
    """Match the browser's displayed original without interpolation.

    IMREAD_UNCHANGED preserves alpha/16-bit depth but intentionally ignores
    EXIF orientation, so apply the eight orientation transforms explicitly.
    """
    if orientation == 2:
        return cv2.flip(image, 1)
    if orientation == 3:
        return cv2.rotate(image, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(image, 0)
    if orientation == 5:
        return cv2.transpose(image)
    if orientation == 6:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.transpose(image), -1)
    if orientation == 8:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _transformed_size(width, height, percent, rotation):
    scaled_width = max(1, int(round(width * percent / 100.)))
    scaled_height = max(1, int(round(height * percent / 100.)))
    _check_dimensions(scaled_width, scaled_height, 'Resized')
    if rotation in (90., 270.):
        result_width, result_height = scaled_height, scaled_width
    elif rotation in (0., 180.):
        result_width, result_height = scaled_width, scaled_height
    else:
        radians = math.radians(rotation)
        sine, cosine = abs(math.sin(radians)), abs(math.cos(radians))
        result_width = int(math.ceil(scaled_width * cosine + scaled_height * sine))
        result_height = int(math.ceil(scaled_height * cosine + scaled_width * sine))
    _check_dimensions(result_width, result_height, 'Rotated')
    return scaled_width, scaled_height, result_width, result_height


def prepare_image_bytes(payload, resolution_percent=100., rotation_degrees=0.):
    """Return (PNG bytes, width, height) without writing or modifying inputs."""
    percent = _number(resolution_percent, 'resolution_percent')
    if not 1. <= percent <= 100.:
        raise ImagePreparationError('resolution_percent must be between 1 and 100')
    rotation = _number(rotation_degrees, 'rotation_degrees') % 360.
    if not isinstance(payload, (bytes, bytearray)) or not payload:
        raise ImagePreparationError('An image file is required')
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ImagePreparationError('Image upload exceeds the 100 MiB limit', 413)
    # Inspect headers before pixel allocation to reject decompression bombs.
    # Pillow is metadata-only here; OpenCV performs all raster operations.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as header:
                width, height = header.size
                orientation = header.getexif().get(274, 1)
    except Image.DecompressionBombError as error:
        raise ImagePreparationError('Source image exceeds the safe decode size', 413) from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ImagePreparationError('The uploaded file is not a readable image') from error
    _check_dimensions(width, height, 'Source')
    display_width, display_height = ((height, width) if orientation in (5, 6, 7, 8) else (width, height))
    new_width, new_height, output_width, output_height = _transformed_size(display_width, display_height, percent, rotation)
    try:
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as error:
        raise ImagePreparationError('The uploaded image could not be decoded') from error
    if image is None:
        raise ImagePreparationError('The uploaded image could not be decoded')
    if image.shape[:2] != (height, width):
        raise ImagePreparationError('Decoded image dimensions differ from the image header')
    if image.dtype not in (np.uint8, np.uint16) or image.ndim not in (2, 3) or (
            image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ImagePreparationError('Only 8/16-bit grayscale, RGB or RGBA images are supported')
    image = _flatten_alpha_white(_normalize_exif_orientation(image, orientation))
    if (new_width, new_height) != (display_width, display_height):
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    if rotation == 90.:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180.:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif rotation == 270.:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation != 0.:
        # Pixel centres, not integer image corners, stay centred in the expanded
        # canvas. OpenCV's positive angle is CCW, hence the sign conversion.
        centre = (.5 * (new_width - 1), .5 * (new_height - 1))
        transform = cv2.getRotationMatrix2D(centre, -rotation, 1.)
        transform[0, 2] += .5 * (output_width - 1) - centre[0]
        transform[1, 2] += .5 * (output_height - 1) - centre[1]
        white = int(np.iinfo(image.dtype).max)
        image = cv2.warpAffine(image, transform, (output_width, output_height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(white, white, white))
    try:
        ok, encoded = cv2.imencode('.png', image)
    except cv2.error as error:
        raise ImagePreparationError('The prepared image could not be encoded as PNG') from error
    if not ok:
        raise ImagePreparationError('The prepared image could not be encoded as PNG')
    return encoded.tobytes(), output_width, output_height
