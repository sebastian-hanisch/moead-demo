"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Buttons (Standardmuster aus dem Demo-Portfolio, siehe
nsga2_presets.py in nsga2-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import moead_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "pop_slider": SettingSpec("pop", int, C.DEFAULT_POP, C.POP_MIN, C.POP_MAX),
    "gens_slider": SettingSpec("gens", int, C.DEFAULT_GEN, C.GEN_MIN, C.GEN_MAX),
    "cx_slider": SettingSpec("cx", float, C.DEFAULT_CX, C.CX_MIN, C.CX_MAX),
    "mut_slider": SettingSpec("mut", float, C.DEFAULT_MUT, C.MUT_MIN, C.MUT_MAX),
    "t_slider": SettingSpec("t", int, C.DEFAULT_T, C.T_MIN, C.T_MAX),
    "neighbor_prob_slider": SettingSpec("nprob", float, C.DEFAULT_NEIGHBOR_PROB, C.NEIGHBOR_PROB_MIN, C.NEIGHBOR_PROB_MAX),
    "update_limit_slider": SettingSpec("nr", int, C.DEFAULT_UPDATE_LIMIT, C.UPDATE_LIMIT_MIN, C.UPDATE_LIMIT_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "run_seed_input": SettingSpec("rseed", int, C.DEFAULT_RUN_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {
    "n": "n_slider", "pop": "pop_slider", "gens": "gens_slider", "cx": "cx_slider", "mut": "mut_slider",
    "t": "t_slider", "neighbor_prob": "neighbor_prob_slider", "update_limit": "update_limit_slider",
    "seed": "seed_input", "run_seed": "run_seed_input",
}
KEPT = {}
STEPS = {"n_slider": C.N_STEP, "pop_slider": C.POP_STEP, "gens_slider": C.GEN_STEP, "cx_slider": C.CX_STEP,
         "mut_slider": C.MUT_STEP, "t_slider": C.T_STEP, "neighbor_prob_slider": C.NEIGHBOR_PROB_STEP, "update_limit_slider": C.UPDATE_LIMIT_STEP}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            lo = SETTING_SPECS[key].lo
            st.session_state[key] = lo + round((st.session_state[key] - lo) / step) * step
            if isinstance(SETTING_SPECS[key].default, int):
                st.session_state[key] = int(st.session_state[key])
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


def randomize_run_seed():
    st.session_state["run_seed_input"] = random.randint(0, C.SEED_MAX)
