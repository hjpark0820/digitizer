"""Source-connected border evidence, not unrelated ink in an adjacent scanline."""
import cv2
import numpy as np


def _box_distance(x, y, box):
    """Chebyshev pixel distance to a half-open box; zero inside it."""
    a,b,c,d=map(int,box)
    return np.maximum.reduce([a-x,x-(c-1),b-y,y-(d-1),np.zeros_like(x)])


def continues(image, box, edge, body_support, *, frame_context=None, entries=None):
    """Test 8-connected body continuation and retain coherent pale crossings.

    Independently measured long frame bands are nuisance evidence, not proof
    that the marker continues. Source pixels are never edited by this test.
    ``entries`` optionally gives the caller's original swatch box ('entry')
    and the other supplied legend boxes ('peers').
    """
    a,b,c,d=map(int,box)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if edge=='left':
        if a==0:return False
        local=body_support[:,0];inside=gray[b:d,a];outer=gray[b:d,a-1]
        deep=gray[b:d,min(a+1,c-1)];far=gray[b:d,max(a-2,0)]
        xx=np.full(d-b,a-1);yy=np.arange(b,d);fx=np.full(d-b,max(a-2,0));fy=yy
    elif edge=='right':
        if c==gray.shape[1]:return False
        local=body_support[:,-1];inside=gray[b:d,c-1];outer=gray[b:d,c]
        deep=gray[b:d,max(a,c-2)];far=gray[b:d,min(c+1,gray.shape[1]-1)]
        xx=np.full(d-b,c);yy=np.arange(b,d);fx=np.full(d-b,min(c+1,gray.shape[1]-1));fy=yy
    elif edge=='top':
        if b==0:return False
        local=body_support[0];inside=gray[b,a:c];outer=gray[b-1,a:c]
        deep=gray[min(b+1,d-1),a:c];far=gray[max(b-2,0),a:c]
        xx=np.arange(a,c);yy=np.full(c-a,b-1);fx=xx;fy=np.full(c-a,max(b-2,0))
    elif edge=='bottom':
        if d==gray.shape[0]:return False
        local=body_support[-1];inside=gray[d-1,a:c];outer=gray[d,a:c]
        deep=gray[max(b,d-2),a:c];far=gray[min(d+1,gray.shape[0]-1),a:c]
        xx=np.arange(a,c);yy=np.full(c-a,d);fx=xx;fy=np.full(c-a,min(d+1,gray.shape[0]-1))
    else:raise ValueError(f'Unknown border edge: {edge}')
    strong=local & (inside<235)
    nearby=cv2.dilate(strong.astype(np.uint8)[None,:],np.ones((1,3),np.uint8))[0].astype(bool)
    linked=nearby & (outer<235)
    # Do not silently accept a genuinely clipped pale stroke. A faint bridge
    # must have observed support on both sides of the edge, at the same pixel
    # position; a detached pale fringe next to a distant dark line is not it.
    pale=local & (inside<248) & (outer<248) & (deep<235) & (far<235)
    if entries is not None and pale.any():
        # Legend rows can be a single pale pixel apart; the dark ink 2 px out
        # is then the NEXT entry's marker, not this marker continuing. Ink
        # strictly nearer another supplied entry than this one is not a bridge.
        own=_box_distance(fx,fy,entries['entry'])
        for peer in entries.get('peers',()):
            pale &= ~(_box_distance(fx,fy,peer)<own)
    linked |= pale
    for band in (frame_context or {}).get('bands',[]):
        pos=xx if band['side'] in ('left','right') else yy
        linked &= ~((pos>=band['start']) & (pos<band['stop']))
    return bool(linked.any())
