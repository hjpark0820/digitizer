"""Original RGB plus non-destructive own/other/paper roles for grouped BW.

Role confidences are engineering weights, not calibrated probabilities. They
never paint ink into source pixels. Ambiguous colour remains weak evidence;
only a confidently different observed ink can excuse missing target ink.
"""
import cv2
import numpy as np

VERSION = 'source_rgb_colour_roles_v2_paper_mixture'


class ColourObservation:
    def __init__(self, source, own, other, paper, report=None):
        if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
            raise ValueError('Source observation requires native uint8 BGR')
        self.source = source  # never mutate; original geometry and pixel values
        for name, plane in [('own',own),('other',other),('paper',paper)]:
            plane=np.asarray(plane,np.float32)
            if plane.shape!=source.shape[:2] or not np.isfinite(plane).all() or np.any((plane<0)|(plane>1)):
                raise ValueError('Colour roles must be finite, image-aligned weights in [0,1]')
            setattr(self,name,plane)
        self.gray=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.report=dict(version=VERSION,source_pixels_unchanged=True,
                         hard_palette_argmax=False,**(report or {}))
        self._members={}

    def validate(self, image):
        if image is not self.source and not np.array_equal(image,self.source):
            raise ValueError('Colour observation must be used with its original source image')

    def membership(self, model):
        key=(float(model.paper_gray),float(model.core_gray))
        if key not in self._members:
            # Tone comes only from the observed source, not the palette fit.
            tone=np.clip((key[0]-self.gray)/max(key[0]-key[1],24.),0,1)
            self._members[key]=(tone*self.own).astype(np.float32)
        return self._members[key]

    def grid_fields(self, model):
        from partial_swatch_detector import _edge_orientation
        tone=np.clip((float(model.paper_gray)-self.gray)/max(float(model.paper_gray-model.core_gray),24.),0,1).astype(np.float32)
        # Find physical source edges FIRST; ownership never creates an edge.
        edge=cv2.Canny(np.uint8(tone>=.45)*255,40,100)>0
        near_own=cv2.dilate(np.uint8((self.own>=.5)&(tone>=.45)),np.ones((3,3),np.uint8))>0
        edge &= near_own
        return self.membership(model).copy(),edge,_edge_orientation(tone)

    def reference_gray(self):
        """Compatibility export for the legacy Step-5 renderer, not detection."""
        return np.rint(255-(255-self.gray)*self.own).astype(np.uint8)


def build_observations(image, models, group_indices, templates):
    """Classify native pixels; narrowly recognise thin-key paper mixtures.

    Distinct orange/brown or light/dark inks remain separate. A weak, hollow
    key may borrow a stronger chromatic ray only when its measured colour is
    almost collinear with that ink's white-paper mixture. This is shared
    observation support, never a series-ID merge or a marker-shape assignment.
    """
    from color_palette_identity_v46 import paper_mixture_evidence
    n=len(models);aliases=np.eye(n,dtype=bool);reasons=[]
    for i,model in enumerate(models):
        if model['achromatic']:continue
        hollow=all(templates[k].marker_kind=='open' for k in group_indices[i])
        if not hollow:continue
        background=np.asarray(model.get('paper_bgr',[255.,255.,255.]),float)
        a=background-np.asarray(model['bgr'],float);na=np.linalg.norm(a)
        for j,rival in enumerate(models):
            if i==j or rival['achromatic']:continue
            rival_background=np.asarray(rival.get('paper_bgr',[255.,255.,255.]),float)
            if np.linalg.norm(background-rival_background)>8.:continue
            b=rival_background-np.asarray(rival['bgr'],float);nb=np.linalg.norm(b)
            alpha=float(a@b/max(nb*nb,1.));error=float(np.linalg.norm(a-alpha*b))
            cosine=float(a@b/max(na*nb,1.))
            if .25<=alpha<=.75 and cosine>=.995 and error<=18.:
                aliases[i,j]=aliases[j,i]=True
                reasons.append(dict(groups=[i,j],reason='thin_hollow_key_paper_mixture',alpha=alpha,residual=error,cosine=cosine))
    if not n:return []
    pixels=image.astype(np.float32)
    background=np.median([m.get('paper_bgr',[255.,255.,255.]) for m in models],axis=0).astype(np.float32)
    contrast=np.linalg.norm(background-pixels,axis=2)
    ink=np.clip((contrast-7.)/18.,0,1)
    paper=np.clip((16.-contrast)/10.,0,1).astype(np.float32)
    # A pale antialiased edge follows the same ink/paper ray. Absolute chroma
    # is NOT an identity test; it falls with alpha even for a perfect match.
    fits=np.asarray([paper_mixture_evidence(image,m)['fit'] for m in models],np.float32)
    neutral=paper_mixture_evidence(image,dict(bgr=[0.,0.,0.],paper_bgr=background))['fit']
    # Low-contrast near-neutral pixels are unknown, not occluding black ink.
    # Dark observed neutral ink still has to explain the pixel better than
    # the target ink. The neutral hypothesis does not add marker evidence.
    neutral_strength=np.clip((np.mean(background-pixels,axis=2)-32.)/40.,0,1)
    achromatic=np.asarray([m['achromatic'] for m in models],bool)
    expanded=np.asarray([fits[aliases[i]].max(axis=0) for i in range(n)])
    output=[]
    for i,m in enumerate(models):
        ownfit=expanded[i];rivals=fits[~aliases[i]]
        rival=rivals.max(axis=0) if len(rivals) else np.zeros_like(ownfit)
        if not m['achromatic']:rival=np.maximum(rival,neutral)
        # Near ties cannot become strong ownership solely from palette order.
        # A one-pixel, NON-recursive extension from certain source ink can
        # retain its antialias fringe. It cannot cross a better rival or add
        # positive ink to white pixels (membership still uses source tone).
        confidence=np.clip((ownfit-.15)/.55,0,1)
        gate=1.-np.clip((rival-ownfit-.10)/.35,0,1)
        separation=np.clip((ownfit-rival-.05)/.30,0,1)
        anchors=(ownfit>=.85)&(ownfit-rival>=.20)&(contrast>=25.)
        adjacent=cv2.dilate(anchors.astype(np.uint8),np.ones((3,3),np.uint8))>0
        continuity=adjacent&(ownfit>=.70)&(rival<=ownfit+.10)
        identity=np.maximum(.30+.70*separation,continuity.astype(np.float32))
        own=ink*confidence*gate*identity
        eligible=np.array(fits[~aliases[i]],copy=True)
        if len(eligible):
            eligible[achromatic[~aliases[i]]]*=neutral_strength
            otherfit=eligible.max(axis=0)
        else:otherfit=np.zeros_like(ownfit)
        if not m['achromatic']:
            otherfit=np.maximum(otherfit,neutral*neutral_strength)
        other=ink*np.clip((otherfit-.35)/.4,0,1)*np.clip((otherfit-ownfit-.10)/.35,0,1)
        output.append(ColourObservation(image,own,other,paper,dict(
            shared_observation_groups=np.flatnonzero(aliases[i]).tolist(),
            tint_evidence=[r for r in reasons if i in r['groups']],
            series_identities_merged=False,paper_bgr=background.tolist(),
            colour_policy='bounded_paper_mixture_with_local_evidence',
            low_chroma_is_not_other=True,continuity_radius=1,
            continuity_is_recursive=False,ink_strength_from_source=True)))
    return output
