import numpy as np
import pandas as pd
import pytest

from policy import frame_to_observations
from value_model import EvaluationDataError, make_policy_observations


@pytest.mark.parametrize("field", ["icu_mortality", "hospital_mortality"])
def test_future_mortality_is_not_a_policy_input(field):
    frame = pd.DataFrame({"map_mm_hg": [70], field: [0]})
    with pytest.raises(EvaluationDataError, match="Forbidden"):
        make_policy_observations(frame, ["map_mm_hg", field])
    with pytest.raises(ValueError, match="Forbidden"):
        frame_to_observations(frame, ["map_mm_hg", field])


def test_sdr_rejects_nonfinite_probability_inputs():
    from value_model import sequential_dr_returns

    with pytest.raises(ValueError, match="finite"):
        sequential_dr_returns(
            patient_ids=["SYN-A"],
            time_steps=[0],
            rewards=[1],
            logged_q_values=[0],
            current_policy_values=[0],
            next_policy_values=[0],
            terminal=[1],
            target_logged_probabilities=[np.nan],
            behavior_logged_probabilities=[0.5],
            horizon=1,
            gamma=1,
            caps=[10],
        )
