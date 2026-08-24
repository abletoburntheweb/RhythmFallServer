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
    "section_timing_correction_min_events": 8,
}

DRUM_ENTRY_DEFAULTS: Dict[str, object] = {
    "drum_entry_recovery": True,
    "drum_entry_grace_beats": 2.0,
    "drum_entry_grace_sec": 0.65,
    "drum_entry_recovery_beats": 2.5,
    "drum_entry_max_recover": 4,
    # Sparse intro kick/snare before the dense section (beyond grace), e.g. TTFAF.
    "drum_entry_preamble_sec": 12.0,
    "drum_entry_preamble_max": 8,
    "drum_entry_merge_tol": 0.04,
    "drum_entry_use_classified": True,
    "drum_entry_ensure_first": True,
    "drum_entry_ensure_beats": 1.5,
}

SECTION_PASS_DEFAULTS: Dict[str, object] = {
    "section_pass_enabled": True,
    "preserve_strong_beats": True,
    "strong_beat_tolerance_beats": 0.18,
    "section_sparse_block_measures": 2,
    "section_core_quiet_max": 1,
    "section_sparse_block_trigger_notes": 3,
    # If a sparse_block measure already has this few notes, keep them (don't pulse-wipe).
    # Align with section_sparse_block_trigger_notes so orphan-at-threshold isn't thinned to 1.
    "section_sparse_keep_all_max": 3,
    "section_ks_orphan_strip": True,
    "section_runaway_ratio": 2.0,
    "section_runaway_neighbor_radius": 2,
    "section_runaway_cap_mult": 1.35,
    "section_runaway_min_events": 10,
    "section_eof_tail_grace_sec": 6.0,
    "section_log_always": True,
    "section_stem_energy_enabled": True,
    "section_stem_quiet_ratio": 0.42,
    "section_stem_quiet_floor": 0.006,
    "section_chart_stem_strip": True,
    "section_ks_weak_max": 1,
    "section_runaway_over_cap": True,
    "section_runaway_over_cap_mult": 1.0,
    "section_dropout_enabled": True,
    "section_dropout_last_dense_ks": 2,
    "section_dropout_min_measures": 2,
    "section_dual_energy_enabled": True,
    "section_contour_rolling_radius": 16,
    "section_mix_loud_rel_min": 0.85,
    "section_drum_quiet_rel_max": 0.55,
    "section_mix_quiet_gate_enabled": True,
    "section_mix_quiet_rel_max": 0.52,
    "section_mix_quiet_trigger_notes": 1,
    "section_phantom_orphan_enabled": True,
    "section_phantom_mix_rel_max": 0.62,
    "section_phantom_mix_absolute_ratio": 0.40,
    "section_phantom_min_notes": 1,
}

FILL_ZONE_DEFAULTS: Dict[str, object] = {
    "fill_zone_enabled": True,
    "fill_zone_metal_enabled": True,
    "fill_zone_spike_max_bpm": 240,
    "fill_zone_min_events": 12,
    "fill_zone_spike_min_events": 10,
    "fill_zone_spike_ratio": 1.42,
    "fill_zone_spike_neighbor_radius": 2,
    "fill_zone_spike_min_delta": 3,
    "fill_zone_spike_require_peak": True,
    "fill_zone_spike_max_consecutive": 2,
    "fill_zone_spike_halo_measures": 1,
    "fill_zone_metal_spike_ratio": 1.28,
    "fill_zone_metal_min_events": 14,
    "fill_zone_cluster_mult": 0.42,
    "fill_zone_flam_mult": 0.35,
    "fill_zone_metal_cluster_mult": 0.58,
    "fill_zone_metal_flam_mult": 0.50,
    "fill_zone_halo_cluster_mult": 0.72,
    "fill_zone_halo_flam_mult": 0.62,
    "fill_zone_metal_halo_cluster_mult": 0.68,
    "fill_zone_metal_halo_flam_mult": 0.58,
}

CRITIC_DEFAULTS: Dict[str, object] = {
    "critic_intro_no_add": True,
    "critic_intro_grace_beats": 1.0,
    "critic_intro_preserve_first_onset": True,
    "critic_intro_sparse_max_notes": 4,
    "critic_playability_lint": True,
    "critic_lint_min_lane_gap_sec": 0.04,
}

LOOP_RECOVERY_DEFAULTS: Dict[str, object] = {
    "loop_reinforce": True,
    "loop4_reinforce": True,
    "fill_recover": True,
}

CONSISTENCY_PASS_DEFAULTS: Dict[str, object] = {
    "rhythm_consistency": False,
    "rhythm_consistency_radius": 4,
    "rhythm_consistency_min_support": 3,
    "rhythm_consistency_max_add_per_measure": 2,
    "rhythm_consistency_max_add_per_class": 1,
    "rhythm_consistency_seek_window": 0.09,
    "rhythm_consistency_neighbor_radius": 2,
    "rhythm_consistency_outlier_ratio": 0.45,
    "rhythm_consistency_skip_fill": True,
    "rhythm_consistency_log_always": True,
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
        "dominant_onsets_policy": "kick_snare_core",
        "rhythm_hit_classes": ["kick", "snare"],
        "sparse_recovery_seconds": 150.0,
        "sparse_recovery_max_add": 5,
        **DRUM_ENTRY_DEFAULTS,
        "lane_by_drum": False,
        "allow_client_overrides": False,
    },
    "basic": {
        "mode": "basic",
        "fill": 0,
        "groove": 50,
        "density": 50,
        "grid_snap_strength": 40,
        "accent_strong_beats": False,
        "genre_template_strength": 55,
        "min_note_distance_floor": 0.05,
        "hit_cluster_window": 0.11,
        "hit_cluster_beat_fraction": 0.30,
        "flam_merge_sec": 0.11,
        "preserve_core_hits": True,
        "core_hit_tolerance": 0.10,
        "expected_groove": "",
        **CONSISTENCY_PASS_DEFAULTS,
        **SECTION_TIMING_DEFAULTS,
        **SECTION_PASS_DEFAULTS,
        **FILL_ZONE_DEFAULTS,
        **CRITIC_DEFAULTS,
        **LOOP_RECOVERY_DEFAULTS,
        "groove_completion": True,
        "groove_completion_radius": 4,
        "groove_completion_min_support": 3,
        "groove_completion_max_add_per_measure": 1,
        "section_timing_correction": False,
        "max_hits_per_second": 9,
        "max_notes_per_measure": 12,
        "dominant_onsets_policy": "classified_hits",
        "rhythm_hit_classes": ["kick", "snare", "hat", "tom", "cymbal"],
        "sparse_recovery_seconds": 150.0,
        "sparse_recovery_max_add": 5,
        **DRUM_ENTRY_DEFAULTS,
        # Spread notes across lanes for playability (not fixed kick→0 / snare→1).
        "lane_by_drum": False,
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
        "hit_cluster_window": 0.10,
        "hit_cluster_beat_fraction": 0.30,
        "flam_merge_sec": 0.11,
        "preserve_core_hits": True,
        "core_hit_tolerance": 0.09,
        "max_hits_per_second": 12,
        "max_notes_per_measure": 14,
        **PATTERN_EXPECTATION_DEFAULTS,
        **CONSISTENCY_PASS_DEFAULTS,
        **SECTION_TIMING_DEFAULTS,
        **SECTION_PASS_DEFAULTS,
        **FILL_ZONE_DEFAULTS,
        **CRITIC_DEFAULTS,
        **LOOP_RECOVERY_DEFAULTS,
        "groove_completion": True,
        "groove_completion_radius": 4,
        "groove_completion_min_support": 3,
        "groove_completion_max_add_per_measure": 2,
        "expected_groove": "",
        "section_timing_correction": False,
        "dominant_onsets_policy": "classified_hits",
        "rhythm_hit_classes": ["kick", "snare", "hat", "tom", "cymbal"],
        "sparse_recovery_seconds": 150.0,
        "sparse_recovery_max_add": 6,
        **DRUM_ENTRY_DEFAULTS,
        "lane_by_drum": False,
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
        "dominant_onsets_policy": "classified_hits",
        "lane_by_drum": False,
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
        "dominant_onsets_policy": "classified_hits",
        "lane_by_drum": False,
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
