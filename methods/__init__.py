from methods.baselines import all_at_once, binary_search, step_by_step
from methods.dcfa import dcfa

METHODS = {
    "all_at_once": all_at_once,
    "step_by_step": step_by_step,
    "binary_search": binary_search,
    "dcfa": dcfa,
}
