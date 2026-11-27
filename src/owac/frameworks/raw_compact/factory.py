"""Select the checkpoint's declared policy while keeping CoRP the default."""


def make_policy(cfg, *, initialize=False):
    kind = cfg["model"].get("kind", "corp")
    if kind == "pouring_feedback_regression":
        from owac.frameworks.raw_compact.pouring_feedback import PouringFeedbackPolicy

        return PouringFeedbackPolicy(cfg, initialize=initialize)
    if kind == "corp_temporal_spatial_regression":
        from owac.frameworks.raw_compact.temporal_spatial_regression import (
            TemporalSpatialRegression,
        )

        return TemporalSpatialRegression(cfg, initialize=initialize)
    if kind == "corp_spatial_regression":
        from owac.frameworks.raw_compact.spatial_regression import SpatialActionRegression

        return SpatialActionRegression(cfg, initialize=initialize)
    if kind == "fruit_placement_regression":
        from owac.frameworks.raw_compact.fruit_placement import FruitPlacementPolicy

        return FruitPlacementPolicy(cfg)
    if kind == "corp_regression" or (
        kind == "corp" and cfg["model"].get("action_head") == "regression"
    ):
        from owac.frameworks.raw_compact.regression import RegressionPolicy

        return RegressionPolicy(cfg, initialize=initialize)
    if kind != "corp":
        raise ValueError(f"Unknown policy kind: {kind}")
    from owac.frameworks.raw_compact.policy import CompactPolicy

    return CompactPolicy(cfg, initialize=initialize)
