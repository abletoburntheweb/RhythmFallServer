# app/generation_intents.py
"""Chart intent resolution (Phase C). See docs/generation_intents.md."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from .generation_presets import (
    GENERATION_PRESETS,
    SECTION_PASS_DEFAULTS,
    resolve_generation_preset,
)

CHART_INTENTS = frozenset({"original", "groove", "arcade", "sparse"})

GOALS = frozenset({"original", "arcade"})
DIFFICULTIES = frozenset({"relaxed", "standard", "dense"})
DEFAULT_GOAL = "original"
DEFAULT_DIFFICULTY = "standard"

# Base preset id per goal×difficulty pair (see docs/generation_settings_screen.md).
GOAL_DIFF_BASE_PRESET: Dict[str, str] = {
    "original|relaxed": "minimal",
    "original|standard": "basic",
    "original|dense": "basic",
    "arcade|relaxed": "minimal",
    "arcade|standard": "basic",
    "arcade|dense": "enhanced",
}

# Goal axis: documentary vs playable arcade.
_GOAL_POLICY: Dict[str, Dict[str, Any]] = {
    "original": {
        "generation_goal": "original",
        "style_builder": "original",
        "groove_completion": False,
        "loop_reinforce": False,
        "loop4_reinforce": False,
        "fill_recover": False,
        "fill": 0,
        "dominant_onsets_policy": "classified_hits",
        "rhythm_hit_classes": ["kick", "snare", "hat", "tom", "cymbal"],
        # Documentary: merge more detector hits so Original stays leaner than Arcade.
        "hit_cluster_window": 0.14,
        "flam_merge_sec": 0.13,
    },
    "arcade": {
        "generation_goal": "arcade",
        "style_builder": "arcade",
        "groove_completion": True,
        "loop_reinforce": True,
        "loop4_reinforce": True,
        "fill_recover": True,
        "fill": 0,
        "dominant_onsets_policy": "classified_hits",
        "rhythm_hit_classes": ["kick", "snare", "hat", "tom", "cymbal"],
        # arcade_mode.md — playability passes + ergonomic lane router (pass 6).
        "arcade_policy": True,
        "arcade_tension_map": True,
        "arcade_phantom_gate": True,
        "arcade_backbeat": True,
        "arcade_texture_downsample": True,
        "ergonomic_router": True,
    },
}

# Difficulty axis layered on goal policy.
_DIFFICULTY_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "relaxed": {
        "fill": 0,
        "groove": 25,
        "density": 28,
        "grid_snap_strength": 72,
        "accent_strong_beats": True,
        "genre_template_strength": 42,
        "critic_strength": 32,
        "max_hits_per_second": 4,
        "max_notes_per_measure": 5,
        "section_pass_enabled": False,
        "dominant_onsets_policy": "kick_snare_core",
        "rhythm_hit_classes": ["kick", "snare"],
    },
    "standard": {
        "fill": 0,
        "groove": 50,
        "density": 50,
        "grid_snap_strength": 40,
        "accent_strong_beats": False,
        "genre_template_strength": 55,
        "critic_strength": 48,
        "max_hits_per_second": 9,
        "max_notes_per_measure": 12,
        "section_pass_enabled": True,
        # Let density caps bind; basic preset preserve_core would freeze kick/snare count.
        "preserve_core_hits": False,
        "section_runaway_cap_mult": 1.22,
        # Pattern trim when per-measure caps do not bind (sparse/metal tracks).
        "difficulty_event_ratio": 0.90,
    },
    "dense": {
        "fill": 15,
        "groove": 58,
        "density": 78,
        "grid_snap_strength": 28,
        "accent_strong_beats": False,
        "genre_template_strength": 62,
        "critic_strength": 55,
        "max_hits_per_second": 12,
        "max_notes_per_measure": 16,
        "section_pass_enabled": True,
        "groove_completion_max_add_per_measure": 2,
        "preserve_core_hits": False,
        "fill_dense_boost": True,
        "section_runaway_cap_mult": 1.42,
        # Add texture back when caps do not bind.
        "difficulty_texture_topup": 0.12,
    },
}

# Arcade relaxed: lighter completion than standard arcade.
_ARCADE_RELAXED_OVERRIDES: Dict[str, Any] = {
    "groove_completion": True,
    "groove_completion_max_add_per_measure": 1,
    "loop_reinforce": True,
    "loop4_reinforce": False,
    "fill_recover": False,
}

# Original dense: keep documentary policy but allow more detector output.
_ORIGINAL_DENSE_OVERRIDES: Dict[str, Any] = {
    "groove_completion": False,
    "loop_reinforce": False,
    "loop4_reinforce": False,
    "fill_recover": False,
    "preserve_core_hits": True,
    "hit_cluster_window": 0.08,
}

LEGACY_MODE_TO_INTENT: Dict[str, str] = {
    "minimal": "sparse",
    "basic": "groove",
    "enhanced": "groove",
    "natural": "original",
    "custom": "groove",
}

INTENT_TO_LEGACY_MODE: Dict[str, str] = {
    "original": "basic",
    "groove": "basic",
    "sparse": "minimal",
    "arcade": "basic",
}

INTENT_TO_BASE_PRESET: Dict[str, str] = {
    "original": "basic",
    "groove": "basic",
    "sparse": "minimal",
    "arcade": "basic",
}

# UI-aligned defaults; server bundles layer policy on top of base presets.
INTENT_UI_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "original": {
        "fill": 0,
        "groove": 50,
        "density": 50,
        "grid_snap_strength": 40,
        "accent_strong_beats": False,
        "genre_template_strength": 55,
        "critic_strength": 45,
        "groove_completion": False,
        "raw_adtof": False,
    },
    "groove": {
        "fill": 0,
        "groove": 50,
        "density": 50,
        "grid_snap_strength": 40,
        "accent_strong_beats": False,
        "genre_template_strength": 55,
        "critic_strength": 50,
        "groove_completion": True,
        "raw_adtof": False,
    },
    "sparse": {
        "fill": 0,
        "groove": 20,
        "density": 30,
        "grid_snap_strength": 85,
        "accent_strong_beats": True,
        "genre_template_strength": 45,
        "critic_strength": 30,
        "groove_completion": False,
        "raw_adtof": False,
    },
    "arcade": {
        "fill": 0,
        "groove": 50,
        "density": 55,
        "grid_snap_strength": 35,
        "accent_strong_beats": False,
        "genre_template_strength": 60,
        "critic_strength": 52,
        "groove_completion": True,
        "raw_adtof": False,
    },
}

# Policy bundles: add-passes, section pass, caps (beyond base preset).
INTENT_POLICY_BUNDLES: Dict[str, Dict[str, Any]] = {
    "original": {
        "groove_completion": False,
        "loop_reinforce": False,
        "loop4_reinforce": False,
        "fill_recover": False,
        "fill": 0,
        "dominant_onsets_policy": "classified_hits",
        "rhythm_hit_classes": ["kick", "snare", "hat", "tom", "cymbal"],
    },
    "groove": {
        "groove_completion": True,
        "loop_reinforce": True,
        "loop4_reinforce": True,
        "fill_recover": True,
        "fill": 0,
        "dominant_onsets_policy": "classified_hits",
        "rhythm_hit_classes": ["kick", "snare", "hat", "tom", "cymbal"],
    },
    "sparse": {
        "groove_completion": False,
        "loop_reinforce": False,
        "loop4_reinforce": False,
        "fill_recover": False,
        "section_pass_enabled": False,
        "dominant_onsets_policy": "kick_snare_core",
        "rhythm_hit_classes": ["kick", "snare"],
    },
    "arcade": {
        "groove_completion": True,
        "loop_reinforce": True,
        "loop4_reinforce": True,
        "fill_recover": True,
        "fill": 0,
        "dominant_onsets_policy": "classified_hits",
        "rhythm_hit_classes": ["kick", "snare", "hat", "tom", "cymbal"],
        "arcade_policy": True,
        "arcade_tension_map": True,
        "arcade_phantom_gate": True,
        "arcade_backbeat": True,
        "arcade_texture_downsample": True,
        "ergonomic_router": True,
    },
}

RAW_ADTOF_POLICY: Dict[str, Any] = {
    "grid_snap_strength": 0,
    "groove_completion": False,
    "loop_reinforce": False,
    "loop4_reinforce": False,
    "fill_recover": False,
    "section_pass_enabled": False,
    "genre_template_strength": 20,
    "hit_cluster_window": 0.025,
    "preserve_core_hits": False,
}

USER_PARAM_KEYS = (
    "fill",
    "groove",
    "density",
    "grid_snap_strength",
    "accent_strong_beats",
    "genre_template_strength",
    "include_hi_hats",
    "critic_strength",
    "groove_completion",
    "raw_adtof",
)


def normalize_chart_intent(
    chart_intent: Optional[str],
    generation_mode: Optional[str] = None,
) -> str:
    intent = str(chart_intent or "").strip().lower()
    if intent in CHART_INTENTS:
        return intent
    mode = str(generation_mode or "basic").strip().lower()
    return LEGACY_MODE_TO_INTENT.get(mode, "original")


def _pair_key(goal: str, difficulty: str) -> str:
    return f"{str(goal).strip().lower()}|{str(difficulty).strip().lower()}"


def normalize_goal(value: Optional[str]) -> str:
    key = str(value or "").strip().lower()
    return key if key in GOALS else DEFAULT_GOAL


def normalize_difficulty(value: Optional[str]) -> str:
    key = str(value or "").strip().lower()
    return key if key in DIFFICULTIES else DEFAULT_DIFFICULTY


def chart_stem_for_pair(goal: str, difficulty: str) -> str:
    g = normalize_goal(goal)
    # Original is one documentary chart → stem "original" (not original_standard).
    if g == "original":
        return "original"
    d = normalize_difficulty(difficulty)
    return f"{g}_{d}"


def pair_from_stem(stem: str) -> Dict[str, str]:
    key = str(stem or "").strip().lower()
    if key == "original" or key.startswith("original_"):
        if key.startswith("original_"):
            legacy_d = key[len("original_") :]
            if legacy_d in DIFFICULTIES:
                return {"goal": "original", "difficulty": legacy_d}
        return {"goal": "original", "difficulty": DEFAULT_DIFFICULTY}
    for difficulty in DIFFICULTIES:
        if chart_stem_for_pair("arcade", difficulty) == key:
            return {"goal": "arcade", "difficulty": difficulty}
    return {"goal": DEFAULT_GOAL, "difficulty": DEFAULT_DIFFICULTY}


# Legacy chart_intent → goal×difficulty when client omits explicit goal/difficulty.
CHART_INTENT_TO_GOAL_DIFF: Dict[str, Tuple[str, str]] = {
    "original": (DEFAULT_GOAL, "standard"),
    "groove": ("arcade", "standard"),
    "sparse": (DEFAULT_GOAL, "relaxed"),
    "arcade": ("arcade", "standard"),
}


def resolve_goal_difficulty_request(
    *,
    goal: Optional[str] = None,
    difficulty: Optional[str] = None,
    chart_intent: Optional[str] = None,
    chart_stem: Optional[str] = None,
    generation_mode: Optional[str] = None,
    user_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve goal×difficulty for API requests; infer from chart_intent or defaults when omitted."""
    goal_in = str(goal or "").strip()
    diff_in = str(difficulty or "").strip()
    if not goal_in and not diff_in:
        stem = str(chart_stem or "").strip().lower()
        if stem:
            pair = pair_from_stem(stem)
            goal_in = str(pair.get("goal", DEFAULT_GOAL))
            diff_in = str(pair.get("difficulty", DEFAULT_DIFFICULTY))
            resolution = f"stem:{stem}"
        elif str(chart_intent or "").strip():
            intent_n = normalize_chart_intent(chart_intent, generation_mode)
            goal_in, diff_in = CHART_INTENT_TO_GOAL_DIFF.get(
                intent_n, (DEFAULT_GOAL, DEFAULT_DIFFICULTY)
            )
            resolution = f"intent:{intent_n}"
        else:
            goal_in = DEFAULT_GOAL
            diff_in = DEFAULT_DIFFICULTY
            resolution = "default"
    else:
        resolution = "client"
    resolved = resolve_goal_difficulty(
        goal=goal_in,
        difficulty=diff_in,
        user_params=user_params,
    )
    resolved["resolution"] = resolution
    return resolved


def resolve_goal_difficulty(
    *,
    goal: Optional[str] = None,
    difficulty: Optional[str] = None,
    user_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve goal×difficulty + user metadata into a generator-ready preset dict."""
    user_params = dict(user_params or {})
    goal_n = normalize_goal(goal)
    diff_n = normalize_difficulty(difficulty)
    pair_key = _pair_key(goal_n, diff_n)
    base_id = GOAL_DIFF_BASE_PRESET.get(pair_key, "basic")
    preset = deepcopy(GENERATION_PRESETS[base_id])
    preset.update(_GOAL_POLICY.get(goal_n, {}))
    preset.update(_DIFFICULTY_OVERRIDES.get(diff_n, {}))
    if goal_n == "arcade" and diff_n == "relaxed":
        preset.update(_ARCADE_RELAXED_OVERRIDES)
    if goal_n == "original" and diff_n == "dense":
        preset.update(_ORIGINAL_DENSE_OVERRIDES)
    if goal_n == "original" and diff_n == "relaxed":
        preset["section_pass_enabled"] = False

    chart_stem = chart_stem_for_pair(goal_n, diff_n)
    _PAIR_TO_LEGACY_INTENT = {
        "original|relaxed": "sparse",
        "original|standard": "original",
        "original|dense": "original",
        "arcade|relaxed": "sparse",
        "arcade|standard": "groove",
        "arcade|dense": "groove",
    }
    legacy_intent = _PAIR_TO_LEGACY_INTENT.get(pair_key, "original")
    legacy_mode = base_id

    preset["allow_client_overrides"] = False
    preset["chart_intent"] = legacy_intent
    preset["chart_stem"] = chart_stem
    preset["generation_goal"] = goal_n
    preset["generation_difficulty"] = diff_n
    preset["mode"] = base_id
    preset["preset_id"] = chart_stem

    if "groove_completion" in user_params and user_params["groove_completion"] is not None:
        preset["groove_completion"] = bool(user_params["groove_completion"])

    for key in ("fill", "groove", "density", "grid_snap_strength", "genre_template_strength"):
        val = _clamp_int_0_100(user_params.get(key))
        if val is not None:
            preset[key] = val
    if user_params.get("accent_strong_beats") is not None:
        preset["accent_strong_beats"] = bool(user_params["accent_strong_beats"])

    raw_adtof = bool(user_params.get("raw_adtof", False))
    if raw_adtof:
        apply_raw_adtof_policy(preset)
        preset["groove_completion"] = False

    apply_critic_strength(preset, user_params.get("critic_strength"), legacy_intent)

    return {
        "goal": goal_n,
        "difficulty": diff_n,
        "chart_stem": chart_stem,
        "chart_intent": legacy_intent,
        "legacy_mode": legacy_mode,
        "preset_id": chart_stem,
        "preset": preset,
    }


def _clamp_int_0_100(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return None


def _lerp(base: float, delta: float, t: float) -> float:
    return base + delta * t


def apply_critic_strength(preset: Dict[str, Any], strength: Optional[int], intent: str) -> None:
    """Map UI critic_strength (0–100) to section-pass aggressiveness.

    50 ≈ SECTION_PASS_DEFAULTS (neutral). Lower = softer cleanup; higher = stricter.
    Sparse has no section pass — only playability lint thresholds are tuned.
    """
    clamped = _clamp_int_0_100(strength)
    if clamped is None:
        default = INTENT_UI_DEFAULTS.get(intent, {}).get("critic_strength", 50)
        clamped = int(default)
    t = (clamped - 50) / 50.0

    if preset.get("section_pass_enabled", "section_sparse_block_trigger_notes" in preset):
        trigger = int(SECTION_PASS_DEFAULTS["section_sparse_block_trigger_notes"])
        preset["section_sparse_block_trigger_notes"] = max(1, min(6, round(_lerp(trigger, -t * 2.0, 1.0))))
        ratio = float(SECTION_PASS_DEFAULTS["section_runaway_ratio"])
        preset["section_runaway_ratio"] = max(1.15, min(3.0, _lerp(ratio, -t * 0.55, 1.0)))
        cap_mult = float(SECTION_PASS_DEFAULTS["section_runaway_cap_mult"])
        preset["section_runaway_cap_mult"] = max(1.0, min(2.0, _lerp(cap_mult, -t * 0.25, 1.0)))
        mix_quiet = float(SECTION_PASS_DEFAULTS["section_mix_quiet_rel_max"])
        preset["section_mix_quiet_rel_max"] = max(0.35, min(0.70, _lerp(mix_quiet, t * 0.12, 1.0)))
        if clamped <= 8:
            preset["section_pass_enabled"] = False
        elif clamped >= 92:
            preset["section_pass_enabled"] = True

    lint_gap = float(preset.get("critic_lint_min_lane_gap_sec", 0.04) or 0.04)
    preset["critic_lint_min_lane_gap_sec"] = max(0.025, min(0.08, _lerp(lint_gap, -t * 0.015, 1.0)))
    preset["critic_strength"] = clamped


def apply_raw_adtof_policy(preset: Dict[str, Any]) -> None:
    for key, value in RAW_ADTOF_POLICY.items():
        preset[key] = value


def resolve_generation_request(
    *,
    chart_intent: Optional[str] = None,
    generation_mode: Optional[str] = None,
    preset_id: Optional[str] = None,
    user_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve chart intent + user metadata into a generator-ready preset dict."""
    user_params = dict(user_params or {})
    intent = normalize_chart_intent(chart_intent, generation_mode)

    if intent in CHART_INTENTS and (chart_intent or "").strip():
        base_id = INTENT_TO_BASE_PRESET[intent]
        preset = deepcopy(GENERATION_PRESETS[base_id])
        preset.update(INTENT_POLICY_BUNDLES.get(intent, {}))
        ui_defaults = INTENT_UI_DEFAULTS.get(intent, {})
        for key in ("fill", "groove", "density", "grid_snap_strength", "accent_strong_beats", "genre_template_strength"):
            if key in ui_defaults:
                preset[key] = ui_defaults[key]
        resolved_preset_id = intent
        legacy_mode = INTENT_TO_LEGACY_MODE[intent]
    else:
        preset = resolve_generation_preset(preset_id, generation_mode)
        resolved_preset_id = str(preset.get("preset_id", preset_id or generation_mode or "basic"))
        legacy_mode = str(preset.get("mode", generation_mode or "basic")).lower()
        intent = normalize_chart_intent(None, legacy_mode)

    preset["allow_client_overrides"] = True
    preset["chart_intent"] = intent
    preset["mode"] = legacy_mode
    preset["preset_id"] = resolved_preset_id

    if "groove_completion" in user_params and user_params["groove_completion"] is not None:
        preset["groove_completion"] = bool(user_params["groove_completion"])

    for key in ("fill", "groove", "density", "grid_snap_strength", "genre_template_strength"):
        val = _clamp_int_0_100(user_params.get(key))
        if val is not None:
            preset[key] = val
    if user_params.get("accent_strong_beats") is not None:
        preset["accent_strong_beats"] = bool(user_params["accent_strong_beats"])

    raw_adtof = bool(user_params.get("raw_adtof", False))
    if raw_adtof:
        apply_raw_adtof_policy(preset)
        preset["groove_completion"] = False

    apply_critic_strength(preset, user_params.get("critic_strength"), intent)

    return {
        "chart_intent": intent,
        "legacy_mode": legacy_mode,
        "preset_id": resolved_preset_id,
        "preset": preset,
    }
