from core.behavioral_finance import cpt_decision_utility, cpt_expected_utility, get_cpt_profile


def test_cpt_profiles_have_expected_loss_aversion_order():
    retail = get_cpt_profile("retail")
    institution = get_cpt_profile("institution")
    quant = get_cpt_profile("quant")
    assert retail.params.lambda_ > institution.params.lambda_ > quant.params.lambda_


def test_cpt_decision_utility_reflects_loss_aversion():
    retail = get_cpt_profile("retail")
    institution = get_cpt_profile("institution")

    retail_u = cpt_decision_utility(-0.10, 0.6, profile=retail)
    inst_u = cpt_decision_utility(-0.10, 0.6, profile=institution)
    assert retail_u < inst_u


def test_cpt_expected_utility_is_numeric():
    profile = get_cpt_profile("retail")
    utility = cpt_expected_utility(
        outcomes=[0.08, 0.03, -0.12],
        probabilities=[0.2, 0.5, 0.3],
        profile=profile,
    )
    assert isinstance(utility, float)

