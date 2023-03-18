from random import random


refresh_saved_versions = 0.2179541331269027
refresh_input_forms = 0.5692718850325763
refresh_confirmed_collab = 1
refresh_collab = 0.5199280392367341

def refresh_hist_inputs():
    ref_hist_inputs = f"{str(random())}"
    return ref_hist_inputs
def refresh_all_inputs():
    ref_hist_inputs = f"{str(random())}"
    ref_proj_inputs = f"{str(random())}"
    return ref_hist_inputs, ref_proj_inputs
def refresh_proj_inputs():
    ref_proj_inputs = f"{str(random())}"
    return ref_proj_inputs
def apply_man_adj():
    refresh_man_adjs = f"{str(random())}"