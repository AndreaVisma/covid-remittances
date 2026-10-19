"""Monthly individual-agent model. No target outcomes are used here."""
import numpy as np
import pandas as pd


def validate_parameters(config):
    for key in ['remittance_share', 'support_probability', 'support_budget_share']:
        if not np.isfinite(config[key]) or not 0 <= config[key] <= 1:
            raise ValueError(f'{key} must be in [0, 1]')
    for key in ['need_effect', 'persistence_effect']:
        if not np.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if config['target_mode'] not in ['current', 'reference']:
        raise ValueError('target_mode must be current or reference')
    if set(config['mechanisms']) != {'need', 'persistence', 'network'}:
        raise ValueError('Expected need, persistence and network switches')
    if any(type(v) is not bool for v in config['mechanisms'].values()):
        raise ValueError('Mechanism switches must be JSON booleans')


def allocate_support(groups, shortfall, offers):
    """Pool offers per recipient; allocate once, proportional to donor offers."""
    count = int(groups.max()) + 1
    required = np.bincount(groups, weights=shortfall, minlength=count)
    offered = np.bincount(groups, weights=offers, minlength=count)
    paid = np.minimum(required, offered)
    fraction = np.divide(paid, offered, out=np.zeros_like(paid), where=offered > 0)
    return offers * fraction[groups], required - paid


def simulate(inputs, config, seed, detail=False):
    validate_parameters(config)
    a, employment, needs = inputs
    n = len(a)
    weight = float(a.weight.iloc[0])
    groups, _ = pd.factorize(a.recipient_id, sort=True)
    rng = np.random.default_rng(seed)
    employed = a.initial_employed.to_numpy(bool).copy()
    previous = a.initial_willing.to_numpy(bool).copy()
    base = a.base_probability.to_numpy(float)
    reference = a.reference_income_usd.to_numpy(float)
    essential = a.essential_cost_usd.to_numpy(float)
    ties = a.tie_strength.to_numpy(float)
    e_index = employment.set_index(['date', 'sector'])
    h_index = needs.set_index(['date', 'recipient_id'])
    monthly, agent_rows = [], []
    flags = config['mechanisms']
    for date in pd.date_range(config['start'], config['end'], freq='MS'):
        # Fixed random-draw schedule enables paired mechanism comparisons.
        job_u, willing_u, support_u = rng.random((3, n))
        shock = e_index.loc[date].reindex(a.sector)
        employed = np.where(employed, job_u >= shock.job_loss_probability.to_numpy(),
                            job_u < shock.job_find_probability.to_numpy())
        income = reference * shock.wage_multiplier.to_numpy() * employed
        capacity = np.maximum(0, income - essential)
        need = h_index.loc[date].reindex(a.recipient_id)['need'].to_numpy()
        # Log-odds shifts preserve exact base probabilities 0 and 1.
        effect = (config['need_effect'] * need * ties if flags['need'] else np.zeros(n))
        if flags['persistence']:
            effect = effect + config['persistence_effect'] * previous
        multiplier = np.exp(np.clip(effect, -700, 700))
        probability = base * multiplier / (1 - base + base * multiplier)
        willing = willing_u < probability
        target_income = income if config['target_mode'] == 'current' else reference
        desired = willing * config['remittance_share'] * target_income
        own = np.minimum(desired, capacity)
        shortfall = desired - own
        offers = np.zeros(n)
        if flags['network']:
            accepts = support_u < config['support_probability'] * ties
            # A request can activate an otherwise unwilling donor; no imitation.
            offers = accepts * config['support_budget_share'] * (capacity - own)
        support, unmet_by_recipient = allocate_support(groups, shortfall, offers)
        sent = own + support
        if np.any(sent > capacity + 1e-8) or np.any(sent < -1e-8):
            raise AssertionError('Individual current-income budget violated')
        if not np.isclose(sent.sum() + unmet_by_recipient.sum(), desired.sum()):
            raise AssertionError('Recipient accounting does not reconcile')
        senders = sent > 0
        monthly.append({
            'date': date, 'total_usd': sent.sum() * weight,
            'own_usd': own.sum() * weight, 'network_usd': support.sum() * weight,
            'desired_usd': desired.sum() * weight, 'unmet_usd': unmet_by_recipient.sum() * weight,
            'labour_income_usd': income.sum() * weight,
            'essential_shortfall_usd': np.maximum(0, essential - income).sum() * weight,
            'willing_count': willing.sum() * weight, 'sender_count': senders.sum() * weight,
            'support_donor_count': (support > 0).sum() * weight,
            'employed_count': employed.sum() * weight,
            'mean_sent_per_sender_usd': sent.sum() / senders.sum() if senders.any() else np.nan,
            'newly_willing_count': (willing & ~previous).sum() * weight})
        if detail:
            agent_rows.append(pd.DataFrame({
                'date': date, 'agent_id': a.agent_id, 'recipient_id': a.recipient_id,
                'weight': weight, 'employed': employed, 'income_usd': income,
                'capacity_usd': capacity, 'probability': probability, 'willing': willing,
                'desired_usd': desired, 'own_usd': own, 'support_usd': support, 'sent_usd': sent}))
        previous = willing.copy()
    return pd.DataFrame(monthly), pd.concat(agent_rows, ignore_index=True) if detail else None
