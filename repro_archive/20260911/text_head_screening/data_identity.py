"""Explicit mapping between R2 mask/workbook keys and returned image names."""


def build_bindings(mask_names, workbook):
    bindings={}
    for mask_name in mask_names:
        image_name=mask_name.replace('mask_','')  # Exact frozen R2 __getitem__ contract.
        if image_name in bindings:
            raise ValueError('Ambiguous mask-to-image mapping: '+image_name)
        if mask_name not in workbook:
            raise KeyError('Missing workbook mask key: '+mask_name)
        bindings[image_name]=dict(mask_name=mask_name,text=workbook[mask_name])
    return bindings
