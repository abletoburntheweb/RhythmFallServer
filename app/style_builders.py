# app/style_builders.py — Style axis: Original (documentary) vs Arcade (playable).
"""See docs/gen_styles.md — Goal builds the raw style map before Difficulty transforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass(frozen=True)
class StylePolicy:
    """Which generative passes are allowed for this generation goal."""

    goal: str
    run_groove_completion: bool = False
    run_loop_reinforce: bool = False
    run_loop4_reinforce: bool = False
    run_fill_recover: bool = False
    run_expected_groove: bool = False
    run_rhythm_consistency: bool = False
    run_arcade_pre_passes: bool = False
    run_arcade_post_passes: bool = False
    run_fill_augmentation: bool = False
    run_enhanced_topup: bool = False
    minimize_fill_zone: bool = False


def get_style_policy(goal: Optional[str], preset: Optional[Dict] = None) -> StylePolicy:
    goal_n = str(goal or (preset or {}).get("generation_goal", "original")).strip().lower()
    if goal_n == "arcade":
        return StylePolicy(
            goal="arcade",
            run_groove_completion=bool((preset or {}).get("groove_completion", True)),
            run_loop_reinforce=bool((preset or {}).get("loop_reinforce", True)),
            run_loop4_reinforce=bool((preset or {}).get("loop4_reinforce", True)),
            run_fill_recover=bool((preset or {}).get("fill_recover", True)),
            run_expected_groove=True,
            run_rhythm_consistency=bool((preset or {}).get("rhythm_consistency", False)),
            run_arcade_pre_passes=bool((preset or {}).get("arcade_policy", True)),
            run_arcade_post_passes=bool((preset or {}).get("arcade_policy", True)),
            run_fill_augmentation=int((preset or {}).get("fill", 0) or 0) > 0,
            run_enhanced_topup=str((preset or {}).get("mode", "")).lower() == "enhanced",
            minimize_fill_zone=False,
        )
    return StylePolicy(
        goal="original",
        run_groove_completion=False,
        run_loop_reinforce=False,
        run_loop4_reinforce=False,
        run_fill_recover=False,
        run_expected_groove=False,
        run_rhythm_consistency=bool((preset or {}).get("rhythm_consistency", False)),
        run_arcade_pre_passes=False,
        run_arcade_post_passes=False,
        run_fill_augmentation=False,
        run_enhanced_topup=False,
        minimize_fill_zone=True,
    )


@dataclass
class StyleBuildResult:
    times: List[float]
    fill_core: Set[int] = field(default_factory=set)
    fill_halo: Set[int] = field(default_factory=set)
    fill_core_list: List[int] = field(default_factory=list)
    fill_halo_list: List[int] = field(default_factory=list)
    recap: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StyleBuildContext:
    events_after_timing: List[float]
    filtered_events: List[float]
    beats: np.ndarray
    bpm: float
    preset: Dict
    mode: str
    genre_label: str
    classified_hits: List[Dict]
    kick_times: List[float]
    snare_times: List[float]
    dominant_onsets: List[float]
    analysis: Dict
    fill: int
    genre_template_strength: int
    verbose: bool
    fill_core: Set[int] = field(default_factory=set)
    fill_halo: Set[int] = field(default_factory=set)
    cancel_cb: Optional[Callable[[], None]] = None


def build_style_map(ctx: StyleBuildContext) -> StyleBuildResult:
    """Apply goal-specific passes to produce the raw style map (pre-difficulty)."""
    from . import drum_generator as dg
    from .arcade_passes import apply_arcade_passes

    policy = get_style_policy(ctx.preset.get("generation_goal"), ctx.preset)
    events = list(ctx.events_after_timing)
    recap: Dict[str, Any] = {"goal": policy.goal, "policy": policy.goal}
    fill_core = set(ctx.fill_core)
    fill_halo = set(ctx.fill_halo)

    if policy.run_rhythm_consistency and bool(ctx.preset.get("rhythm_consistency", False)):
        events = dg._apply_rhythm_consistency_pass(
            events,
            ctx.classified_hits,
            ctx.kick_times,
            ctx.snare_times,
            ctx.beats,
            ctx.bpm,
            ctx.preset,
            verbose=ctx.verbose,
        )
    elif policy.run_groove_completion and not policy.run_arcade_pre_passes:
        events = dg._complete_groove_from_neighbors(
            events, ctx.beats, ctx.bpm, ctx.preset, ctx.verbose
        )

    if policy.run_expected_groove:
        events = dg._apply_expected_groove_grid(
            events, ctx.filtered_events, ctx.beats, ctx.bpm, ctx.preset, ctx.verbose
        )
    if policy.run_loop_reinforce:
        events = dg._reinforce_repeating_measure_hits(
            events, ctx.filtered_events, ctx.beats, ctx.bpm, ctx.mode, ctx.preset, ctx.verbose
        )
    if policy.run_loop4_reinforce:
        events = dg._reinforce_four_bar_loop_hits(
            events, ctx.filtered_events, ctx.beats, ctx.bpm, ctx.mode, ctx.preset, ctx.verbose
        )
    if policy.run_fill_recover:
        events = dg._recover_fill_single_misses(
            events, ctx.filtered_events, ctx.beats, ctx.bpm, ctx.mode, ctx.preset, ctx.verbose
        )

    events = dg._apply_basic_section_timing_correction(
        events, ctx.beats, ctx.bpm, ctx.mode, ctx.preset, ctx.filtered_events, ctx.verbose
    )

    if policy.run_arcade_pre_passes:
        before_arcade = len(events)
        events, arcade_recap = apply_arcade_passes(
            events,
            candidate_events=ctx.filtered_events,
            beats=ctx.beats,
            bpm=ctx.bpm,
            preset=ctx.preset,
            classified_hits=ctx.classified_hits,
            snare_times=ctx.snare_times,
            kick_times=ctx.kick_times,
            genre_label=ctx.genre_label,
            mix_audio_path=ctx.analysis.get("original_path"),
            drum_audio_path=ctx.analysis.get("analysis_path"),
            verbose=ctx.verbose,
            phase="pre_section",
        )
        recap["arcade_pre"] = arcade_recap
        if ctx.verbose and len(events) != before_arcade:
            print(f"[DrumGen][этап] arcade_pass={before_arcade}->{len(events)}")

    return StyleBuildResult(
        times=events,
        fill_core=fill_core,
        fill_halo=fill_halo,
        fill_core_list=sorted(fill_core),
        fill_halo_list=sorted(fill_halo),
        recap=recap,
    )


def apply_style_post_passes(
    all_times: List[float],
    ctx: StyleBuildContext,
    base_times: List[float],
) -> Tuple[List[float], Dict[str, Any]]:
    """Arcade-only post-section passes (groove completion + backbeat)."""
    from . import drum_generator as dg
    from .arcade_passes import apply_arcade_passes

    policy = get_style_policy(ctx.preset.get("generation_goal"), ctx.preset)
    recap: Dict[str, Any] = {}
    if not policy.run_arcade_post_passes:
        return list(all_times), recap

    times = list(all_times)
    before = len(times)
    if policy.run_groove_completion:
        times = dg._complete_groove_from_neighbors(
            times, ctx.beats, ctx.bpm, ctx.preset, ctx.verbose
        )
    times, post_recap = apply_arcade_passes(
        times,
        candidate_events=ctx.filtered_events,
        beats=ctx.beats,
        bpm=ctx.bpm,
        preset=ctx.preset,
        classified_hits=ctx.classified_hits,
        snare_times=ctx.snare_times,
        kick_times=ctx.kick_times,
        genre_label=ctx.genre_label,
        mix_audio_path=ctx.analysis.get("original_path"),
        drum_audio_path=ctx.analysis.get("analysis_path"),
        verbose=ctx.verbose,
        phase="post_section",
    )
    times = dg._append_events_with_density_guardrails(
        times, [], ctx.beats, ctx.bpm, ctx.preset
    )
    recap["post_arcade"] = post_recap
    if ctx.verbose and len(times) != before:
        print(f"[DrumGen][этап] post_arcade={before}->{len(times)}")
    return times, recap
