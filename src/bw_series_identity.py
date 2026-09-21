"""Small B&W-only identity helpers shared by the GUI and JSON adapter."""
import hashlib

PALETTE_RGB=((31,119,180),(255,127,14),(44,160,44),(214,39,40),
             (148,103,189),(140,86,75),(227,119,194),(127,127,127),
             (188,189,34),(23,190,207))
IDENTITY_FIELDS=('swatch_id','shape_hint','class_idx','shape_idx','series_label')


def series_key(point):
    return point.get('swatch_id') or point['class_name']


def series_color(point, legacy_colors):
    swatch_id=point.get('swatch_id')
    if not swatch_id:return legacy_colors.get(point['class_name'],(255,0,0))
    number=swatch_id[1:] if swatch_id.startswith('S') else ''
    index=int(number)-1 if number.isdigit() else int(hashlib.sha256(swatch_id.encode()).hexdigest()[:8],16)
    return PALETTE_RGB[index%len(PALETTE_RGB)]


def identity_fields(point):
    return {key:point[key] for key in IDENTITY_FIELDS if key in point}


def swatch_curve(metadata, labels=None):
    """Point-free identity, usable even when a legend series has no detections."""
    sid=metadata['swatch_id']
    cls=metadata.get('class_name') or metadata.get('shape_hint') or 'unknown_marker'
    label=(labels or {}).get(sid) or metadata.get('series_label') or metadata.get('label')
    row=dict(name=sid,swatch_id=sid,label=label or f'{sid}: {cls.replace("_"," ")}',
             rgb=list(series_color(dict(metadata,class_name=cls),{})),points=[],
             class_name=cls,shape_hint=metadata.get('shape_hint') or cls,
             class_idx=metadata.get('class_idx',metadata.get('series_index')))
    if metadata.get('extraction_status')=='failed':
        row.update(class_name='unknown_marker',shape_hint='unknown_marker',
                   legend_status='unresolved',legend_error=metadata.get('extraction_error'))
    return row


def legend_series_catalog(reports, labels=None):
    """Preserve retained legend IDs; class-filtered entries stay excluded."""
    return [swatch_curve(row,labels) for row in reports
            if row.get('swatch_id') and not row.get('excluded_by_class_filter')]
