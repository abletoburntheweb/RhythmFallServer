# app/generation_presets.py
from copy import deepcopy
from typing import Dict, Optional

PATTERN_EXPECTATION_DEFAULTS: Dict[str, object] = {
    "groove_completion": True,
    "groove_completion_radius": 4,
    "groove_completion_min_support": 3,
    "groove_completion_max_add_per_measure": 1,
    "expected_groove": "halfbeat_drive",
    "expected_groove_radius": 4,
    "expected_groove_min_support": 4,
    "expected_groove_max_add_per_measure": 2,
}

SECTION_TIMING_DEFAULTS: Dict[str, object] = {
    "section_timing_correction": False,
    "section_timing_correction_strength": 0.52,
    "section_timing_correction_cap_ms": 18.0,
    "section_timing_correction_ignore_below_ms": 5.0,
}

LOOP_RECOVERY_DEFAULTS: Dict[str, object] = {
    "loop_reinforce": True,
    "loop4_reinforce": True,
    "fill_recover": True,
}


GENERATION_PRESETS: Dict[str, Dict] = {
    "minimal": {
        "mode": "minimal",
        "fill": 0,
        "groove": 20,
        "density": 30,
        "grid_snap_strength": 85,
        "accent_strong_beats": True,
        "genre_template_strength": 45,
        "min_note_distance_floor": 0.075,
        "hit_cluster_window": 0.12,
        "preserve_core_hits": True,
        "core_hit_tolerance": 0.10,
        "max_hits_per_second": 3,
        "max_notes_per_measure": 4,
        **PATTERN_EXPECTATION_DEFAULTS,
        **SECTION_TIMING_DEFAULTS,
        **LOOP_RECOVERY_DEFAULTS,
        "dominant_onsets_policy": "drum_hits_with_dominant_fallback",
        "allow_client_overrides": False,
    },
    "basic": {
        "mode": "basic",
        "fill": 0,
        "groove": 50,
        "density": 50,
        "grid_snap_strength": 60,
        "accent_strong_beats": True,
        "genre_template_strength": 60,
        "min_note_distance_floor": 0.06,
        "hit_cluster_window": 0.09,
        "preserve_core_hits": True,
        "core_hit_tolerance": 0.11,
        **PATTERN_EXPECTATION_DEFAULTS,
        **SECTION_TIMING_DEFAULTS,
        **LOOP_RECOVERY_DEFAULTS,
        "section_timing_correction": True,
        "max_hits_per_second": 7,
        "max_notes_per_measure": 8,
        "dominant_onsets_policy": "drum_hits_with_dominant_fallback",
        "allow_client_overrides": False,
    },
    "enhanced": {
        "mode": "enhanced",
        "fill": 75,
        "groove": 55,
        "density": 70,
        "grid_snap_strength": 35,
        "accent_strong_beats": False,
        "genre_template_strength": 80,
        "min_note_distance_floor": 0.04,
        "hit_cluster_window": 0.055,
        "preserve_core_hits": True,
        "core_hit_tolerance": 0.09,
        "max_hits_per_second": 7,
        "max_notes_per_measure": 10,
        **PATTERN_EXPECTATION_DEFAULTS,
        **SECTION_TIMING_DEFAULTS,
        **LOOP_RECOVERY_DEFAULTS,
        "dominant_onsets_policy": "dominant_onsets",
        "allow_client_overrides": False,
    },
    "natural": {
        "mode": "natural",
        "fill": 0,
        "groove": 50,
        "density": 50,
        "grid_snap_strength": 0,
        "accent_strong_beats": False,
        "genre_template_strength": 20,
        "min_note_distance_floor": 0.055,
        "hit_cluster_window": 0.025,
        "preserve_core_hits": False,
        "core_hit_tolerance": 0.07,
        "max_hits_per_second": 5,
        "max_notes_per_measure": 8,
        "groove_completion": False,
        "expected_groove": "",
        **SECTION_TIMING_DEFAULTS,
        "loop_reinforce": False,
        "loop4_reinforce": False,
        "fill_recover": False,
        "dominant_onsets_policy": "dominant_onsets",
        "allow_client_overrides": False,
    },
    "custom": {
        "mode": "custom",
        "fill": 0,
        "groove": 50,
        "density": 50,
        "grid_snap_strength": 35,
        "accent_strong_beats": True,
        "genre_template_strength": 60,
        "min_note_distance_floor": 0.035,
        "hit_cluster_window": 0.0,
        "preserve_core_hits": False,
        "core_hit_tolerance": 0.08,
        "max_hits_per_second": 0,
        "max_notes_per_measure": 0,
        **PATTERN_EXPECTATION_DEFAULTS,
        **SECTION_TIMING_DEFAULTS,
        **LOOP_RECOVERY_DEFAULTS,
        "dominant_onsets_policy": "dominant_onsets",
        "allow_client_overrides": True,
    },
}


MODE_TO_PRESET_ID = {
    "minimal": "minimal",
    "basic": "basic",
    "enhanced": "enhanced",
    "natural": "natural",
    "custom": "custom",
}


def available_preset_ids() -> set:
    return set(GENERATION_PRESETS.keys())


def resolve_generation_preset(
    preset_id: Optional[str],
    generation_mode: Optional[str] = None,
) -> Dict:
    requested = str(preset_id or "").strip().lower()
    mode = str(generation_mode or "basic").strip().lower()

    resolved_id = requested or MODE_TO_PRESET_ID.get(mode, "basic")
    if resolved_id not in GENERATION_PRESETS:
        raise ValueError(f"Unknown generation preset: {resolved_id}")

    preset = deepcopy(GENERATION_PRESETS[resolved_id])
    preset["preset_id"] = resolved_id
    return preset
