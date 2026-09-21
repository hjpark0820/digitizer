"""One production profile for CLI, saved-only GUI correction and API callers.

Defaults apply to fresh corrections. Saved objectives are resumed verbatim;
changing weights midway is rejected by the existing evidence/policy fingerprint.
"""
from copy import deepcopy
import json
from pathlib import Path


def production_options():
    return dict(metric='chamfer',model='pchip',reference_policy='estimated_path',inferred_weight=.25,
                limited_model='linear',limited_completeness_weight=.5,limited_image_weight=.5)


def cli_options(args, supplied, *, supported, previous=None):
    flags={'--path-metric':('metric','path_metric'), '--path-model':('model','path_model'),
           '--path-reference':('reference_policy','path_reference'),
           '--path-inferred-weight':('inferred_weight','path_inferred_weight'),
           '--limited-path-model':('limited_model','limited_path_model'),
           '--limited-completeness-weight':('limited_completeness_weight','limited_completeness_weight'),
           '--limited-image-weight':('limited_image_weight','limited_image_weight')}
    saved=(previous or {}).get('path_options')
    if previous is not None and not isinstance(saved,dict):
        raise ValueError('Saved colour correction has no bound objective settings; load its initial detection version for the new defaults')
    core=dict(metric=args.path_metric,model=args.path_model,
              reference_policy=args.path_reference,inferred_weight=args.path_inferred_weight)
    options=deepcopy(saved) if saved is not None else production_options() if supported else core
    if saved is None and args.limited_image_policy=='hard':
        options=core
    for flag,(key,attribute) in flags.items():
        if flag in supplied:
            options[key]=getattr(args,attribute)
    # An explicit global model retains the historical meaning for experiments.
    # Omitted global model keeps connected=PCHIP, supported limited=linear.
    if '--path-model' in supplied and '--limited-path-model' not in supplied:
        options['limited_model']=args.path_model
    if args.limited_image_policy=='hard':
        if '--limited-image-weight' in supplied:
            raise ValueError('Hard image policy cannot also specify an image weight')
        options.pop('limited_image_weight',None)
    elif args.limited_image_policy=='weighted' and options.get('limited_image_weight') is None:
        # Explicitly requesting weighted mode must not leave a saved hard policy
        # untouched. The existing fingerprint then requires a fresh correction.
        options['limited_image_weight']=production_options()['limited_image_weight']
    if not supported and (args.limited_image_policy=='weighted' or any(k.startswith('limited_') for k in options)):
        raise ValueError('Limited objectives require bound v46 marker/path evidence')
    return options


def settings_summary(folder):
    """Read-only GUI notice; show the ACTUAL next-run settings, including old ones."""
    folder=Path(folder)
    if (not (folder/'step5_inputs.json').exists() or (folder/'color_group_state_v46.json').exists()
            or (folder/'type3_detection_v46.json').exists()):
        return None
    state_path=folder/'color_correction_state.json'
    try:
        state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
        options=state.get('path_options') if state is not None else production_options()
        if not isinstance(options,dict):
            return dict(source='saved',policy='unknown',note='Load the initial detection version to use current defaults.')
        soft=options.get('limited_image_weight') is not None
        return dict(source='saved' if state is not None else 'new_defaults',
            policy='weighted_cost' if soft else 'hard_admission',
            image_weight=options.get('limited_image_weight'),
            completeness_weight=options.get('limited_completeness_weight',0.),
            model=options.get('limited_model',options.get('model','pchip')),
            scope='supported fitted/uncertain colour paths only',
            note=('Saved objective settings will be preserved. Load the initial detection version to use current defaults.'
                  if state is not None else 'New correction uses the production weighted-image profile.'))
    except (OSError,ValueError,AttributeError):
        return dict(source='saved',policy='unknown',note='Saved settings could not be read; correction will validate the state.')
