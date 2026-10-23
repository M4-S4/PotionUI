def attach_nag(cond: dict, uncond: dict | None, config: dict, attention_mask: bool = False) -> dict:
    nag_scale = float(config.get("nag_scale", 1.0))
    if nag_scale <= 1.0 or uncond is None:
        return cond
    attached = {
        **cond,
        "nag_context": uncond["context"],
        "nag": {
            "scale": nag_scale,
            "tau": float(config.get("nag_tau", 3.5)),
            "alpha": float(config.get("nag_alpha", 0.5)),
        },
    }
    if attention_mask:
        attached["nag_attention_mask"] = uncond.get("attention_mask")
    return attached
