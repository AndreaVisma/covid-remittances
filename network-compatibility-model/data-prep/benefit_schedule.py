"""Task 3: monthly schedule of US pandemic benefits and their amounts (2019-01 to 2022-12).

Run: python network-compatibility-model/data-prep/benefit_schedule.py  (instant, no inputs)
Output: data-prep/outputs/benefit_schedule_monthly.csv (public statutory facts, safe to commit)

Amounts are per eligible adult in USD. The model's base assumption (user decision, 28 Oct 2026) is
that all migrants receive benefits (pi = 1); in reality most programmes required an SSN or work
authorization (see NOTES), so pi = 1 is an upper bound.

Columns
  eip_round, eip_usd_round        stimulus (Economic Impact Payment) round and amount per adult
  eip_share_paid                  approximate share of the round's value disbursed that month
  eip_usd                         expected EIP per adult paid that month (= amount x share)
  fpuc_programme, fpuc_weekly_usd federal UI supplement in force (FPUC 600, LWA 300, FPUC 300)
  fpuc_weeks                      benefit weeks whose week-ending Saturday falls in the month
  fpuc_usd                        supplement per unemployed recipient, by entitlement week
  fpuc_active_share               share of Mexican-born migrants in states still paying FPUC
                                  (approximate, early opt-outs in June-July 2021)
  lwa_paid_share                  LWA (weeks Aug-Sep 2020) was paid retroactively; approximate
                                  share of LWA value paid in each month
  pandemic_ui_programmes          PUA/PEUC in force (extended benefits beyond regular state UI)
  ctc_usd_per_child_under6/6to17  advance Child Tax Credit, monthly, per child with an SSN
  stimulus_month                  model flag for abm.make_inputs: the main month of each EIP round
                                  (Apr 2020, Jan 2021, Mar 2021). User decision, 28 Oct 2026: only
                                  the three big stimulus months act in the model; UI supplements are
                                  listed for reference but have no flag (ui_months stays off).

Sources (checked 28 Oct 2026):
  EIP1: CARES Act, signed 27 Mar 2020; $1,200/adult; 88-89.5 M payments ($158-160 B) by 17 Apr,
        ~130 M (> $200 B) by 8 May, ~159 M ($267 B) by early June (IRS IR-2020-91, IR-2020-111).
  EIP2: CAA 2021, signed 27 Dec 2020; $600/adult; deliveries began 29 Dec 2020, ~147 M payments by
        ~8 Jan 2021 (113 M direct deposit), paper checks through end of January (IRS; House W&M).
  EIP3: ARPA, signed 11 Mar 2021; $1,400/person; deliveries began 12 Mar 2021, rolling batches
        through April and the rest of the year (IRS). Monthly split approximate.
  FPUC: $600/week for weeks ending 4 Apr - 25 Jul 2020 (CARES); $300/week for weeks ending
        2 Jan - 4 Sep 2021 (CAA, extended by ARPA to 6 Sep). Early opt-outs: 12 Jun to 10 Jul 2021
        (TX, GA, FL 26 Jun; AZ 10 Jul; about 25 states, mostly small Mexican-born populations).
  LWA:  FEMA, $300/week for weeks ending 1 Aug - 5 Sep 2020 (max 6 weeks), paid retroactively
        from late Aug (first states) to mid-Oct 2020 (e.g. Maine 11 Sep, Wisconsin 15 Oct).
  CTC:  ARPA advance payments on 15 Jul, 13 Aug, 15 Sep, 15 Oct, 15 Nov, 15 Dec 2021;
        $300/child under 6, $250/child 6-17.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / 'outputs'
MONTHS = pd.date_range('2019-01-01', '2022-12-01', freq='MS')

# Stimulus rounds: amount per adult and approximate share of the round's value paid per month.
EIP = {
    1: (1200, {'2020-04': 0.65, '2020-05': 0.25, '2020-06': 0.05, '2020-07': 0.03, '2020-08': 0.02}),
    2: (600, {'2020-12': 0.50, '2021-01': 0.50}),
    3: (1400, {'2021-03': 0.75, '2021-04': 0.15, '2021-05': 0.05, '2021-06': 0.05}),
}
# Federal UI supplements: (programme, USD per week, first and last week-ending Saturday).
FPUC = [('FPUC 600', 600, '2020-04-04', '2020-07-25'),
        ('LWA 300', 300, '2020-08-01', '2020-09-05'),
        ('FPUC 300', 300, '2021-01-02', '2021-09-04')]
LWA_PAID = {'2020-08': 0.05, '2020-09': 0.55, '2020-10': 0.40}
# Share of Mexican-born living in states still paying the 2021 supplement. TX (~22% of the
# Mexican-born, ACS 2019) and FL/GA stopped on 26 Jun, AZ (~5%) on 10 Jul, smaller states from
# 12 Jun. Approximate (~0.65 after July); to verify with CPS/ACS state shares.
FPUC_ACTIVE = {'2021-06': 0.90, '2021-07': 0.67, '2021-08': 0.65, '2021-09': 0.65}
# Main disbursement month of each round = the model's stimulus flag. EIP2 is split about evenly
# between late Dec 2020 (direct deposits from 29 Dec) and Jan 2021 (deposits posting, paper checks,
# debit cards); January is used.
STIMULUS_MONTHS = {1: '2020-04', 2: '2021-01', 3: '2021-03'}
PANDEMIC_UI = ('2020-04-01', '2021-09-01')        # PUA/PEUC in force (CARES -> CAA -> ARPA)
CTC = ('2021-07-01', '2021-12-01')

NOTES = {
    '2020-04': 'EIP1 begins (SSN required for every filer: mixed-status couples and ITIN filers excluded); '
               'FPUC 600 starts, but state UI backlogs delayed many first payments to May',
    '2020-08': 'FPUC 600 expired after week ending 25 Jul; LWA weeks begin (paid from late Aug)',
    '2020-10': 'No federal supplement Oct-Dec 2020 (regular UI, PUA, PEUC only)',
    '2020-12': 'EIP2 begins 29 Dec (CAA made mixed-status couples eligible, also retroactively for EIP1)',
    '2021-01': 'FPUC 300 starts (week ending 2 Jan)',
    '2021-03': 'EIP3 begins 12 Mar (SSN dependants of ITIN filers now eligible)',
    '2021-06': 'Early FPUC opt-outs from 12 Jun (TX, FL, GA on 26 Jun)',
    '2021-07': 'AZ opt-out 10 Jul; advance CTC begins 15 Jul (child needs SSN)',
    '2021-09': 'FPUC, PUA and PEUC end (week ending 4 Sep / 6 Sep)',
}


def build():
    ym = MONTHS.strftime('%Y-%m')
    df = pd.DataFrame({'date': MONTHS.date})
    df['eip_round'] = pd.array([pd.NA] * len(df), dtype='Int64')
    df['eip_usd_round'] = 0
    df['eip_share_paid'] = 0.0
    for rnd, (usd, shares) in EIP.items():
        assert abs(sum(shares.values()) - 1) < 1e-9
        for m, s in shares.items():
            i = ym.get_loc(m)
            df.loc[i, ['eip_round', 'eip_usd_round', 'eip_share_paid']] = [rnd, usd, s]
    df['eip_usd'] = df.eip_usd_round * df.eip_share_paid

    df['fpuc_programme'] = ''
    df['fpuc_weekly_usd'] = 0
    df['fpuc_weeks'] = 0
    for name, usd, first, last in FPUC:
        weeks = pd.date_range(first, last, freq='W-SAT')
        for m, n in weeks.to_period('M').value_counts().items():
            i = ym.get_loc(str(m))
            df.loc[i, ['fpuc_programme', 'fpuc_weekly_usd', 'fpuc_weeks']] = [name, usd, n]
    df['fpuc_usd'] = df.fpuc_weekly_usd * df.fpuc_weeks
    df['fpuc_active_share'] = np.where(df.fpuc_weeks > 0, 1.0, 0.0)
    for m, s in FPUC_ACTIVE.items():
        df.loc[ym.get_loc(m), 'fpuc_active_share'] = s
    df['lwa_paid_share'] = [LWA_PAID.get(m, 0.0) for m in ym]
    d = pd.to_datetime(df.date)
    df['pandemic_ui_programmes'] = d.between(*PANDEMIC_UI).astype(int)
    in_ctc = d.between(*CTC)
    df['ctc_usd_per_child_under6'] = np.where(in_ctc, 300, 0)
    df['ctc_usd_per_child_6to17'] = np.where(in_ctc, 250, 0)
    df['stimulus_month'] = ym.isin(list(STIMULUS_MONTHS.values())).astype(int)
    assert df.stimulus_month.sum() == 3
    df['notes'] = [NOTES.get(m, '') for m in ym]
    return df


if __name__ == '__main__':
    df = build()
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / 'benefit_schedule_monthly.csv', index=False)
    show = df[pd.to_datetime(df.date).between('2020-03-01', '2021-12-01')]
    print(show.drop(columns='notes').to_string(index=False))
    tot = df[['eip_usd', 'fpuc_usd']].sum()
    print(f"\nTotal EIP per adult {tot.eip_usd:,.0f} USD; FPUC/LWA per continuously unemployed "
          f"recipient {tot.fpuc_usd:,.0f} USD (before early opt-outs)")
