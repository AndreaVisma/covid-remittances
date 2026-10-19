# Minimal ENIGH remittance comparison

Run `python oct-26/analysis/enigh-remittances.py` from the project directory.
Dependencies: pandas, numpy, matplotlib. The script reads the two supplied
CONCENTRADOHOGAR CSV files and writes tables and four PNGs to
`oct-26/outputs/enigh-remittances/`. Source files are unchanged.

## Definitions and scope

- Unit: household. Each file is a separate cross-section, not a linked panel.
- Recipient: `remesas > 0`. This is recorded income from other countries
  (income code P041), not US-only remittances or the identity of a sender.
- Amounts are quarterly household amounts. They are not calendar-year totals.
- All population estimates, percentages, means and quantiles use `factor`.
  Raw sample counts are also exported. Weighted quantiles use the inverse
  cumulative weighted distribution (first value reaching the target share).
- Receipt rate denominator: all households in the indicated category.
  Recipient composition denominator: all remittance-receiving households.
  These measures must not be confused. The group table also includes each
  category's share of all households for comparison.
- `menores` means under 12, NOT under 18 or under 15. `p65mas` means 65+.
  The age-composition groups therefore differ from the earlier MMP analysis.
- `ocupados` counts employed people aged 14+ in both editions. A household
  with no employed member can still have pensions or other income.
- Rural means locality under 2,500 residents (`tam_loc=4`).
- Dependence is each recipient household's `remesas / ing_cor`. Total current
  income includes imputed rent and noncash components; this is not a share of
  cash income. Zero-income nonrecipients have an undefined ratio and are not
  included. Summary means average household ratios, rather than dividing totals.

## Price comparison

Nominal amounts are preserved in `national_summary.csv`. Plots express amounts
in approximately 2022 pesos. The 2020 multiplier is
`57370 / 50309.31315023509 = approximately 1.14035`: INEGI's published 2020
national mean current income expressed in 2022 pesos, divided by the mean
computed from the original 2020 file. The published numerator is rounded, so
this is an approximate common price conversion, not a reconstruction of the
official detailed deflation procedure. The script reproduces the published
mean remittances across ALL households (approximately 475 and 661 pesos).
Those figures are much smaller than the means among recipients because they
include households reporting zero remittances.

## Results

The weighted recipient share rises from 4.38% to 4.57% (+0.19 percentage points).
Estimated recipient households rise from 1.565 million to 1.716 million (+9.7%).
Average quarterly amounts among recipients rise from approximately 10,842 to
14,455 pesos at 2022 prices (+33.3%). The median rises from 5,608 to 8,310
(+48.2%). The survey's real remittance total rises 46.2%: arithmetically this
combines growth in recipient counts and in the conditional mean. It does not
identify continuing recipients or new entrants.

Recipients become more frequently female-headed (43.8% to 50.9%) and rural
(44.7% to 51.6%). The share with no employed member rises from 20.6% to 23.4%.
These composition changes can reflect both shifts in the overall population
and shifts in receipt rates. In particular, rural receipt rises from 9.1% to
10.3%, while urban receipt falls from 3.1% to 2.9%.

Mean household dependence rises from 25.5% to 30.5%. About 17.5% of recipients
in 2020 and 24.1% in 2022 obtain at least half their current income from
remittances. The distribution plot uses >50% as its upper band; exact 50%
belongs to the preceding band, whereas the national indicator includes it.

For microsimulation, these are useful national targets for recipient prevalence,
recipient composition, conditional amounts and reliance on transfers. They do
not establish migrant network structure. MMP can inform that unobserved part.

## Outputs and limits

1. `01_national_overview.png`: receipt share, recipient count and mean amount.
2. `02_recipient_composition.png`: distribution of recipient households.
3. `03_receipt_by_group.png`: receipt rates within each household group.
4. `04_amounts_and_dependence.png`: median/interquartile range and income shares.

Tables retain valid denominators, sample counts and weights. `group_changes.csv`
provides percentage-point changes; `states_for_later.csv` retains state summaries
without adding state plots. `national_changes.csv` reports level and relative
changes (for percentage variables, level differences are percentage points).

These are descriptive point estimates without survey-design standard errors or
significance tests. The checks validate unique keys, nonmissing inputs, positive
weights, plausible remittances, age-count reconciliation, category shares and
the identity total = recipient count x recipient mean. All four plots are
visually reviewed. Small state samples require caution before future use.

Changes are not causal pandemic effects: 2020 is already a pandemic year, and
the samples contain different households. Survey remittances are not calibrated
to Banco de Mexico aggregate flows. Neither an annualization nor a claim of
agreement with those flows is made. A larger median does not by itself establish
that remittance inequality fell; this minimal analysis does not estimate a Gini.

## Documentation

- 2020 variable construction:
  https://www.inegi.org.mx/contenidos/programas/enigh/nc/2020/doc/enigh2020_ns_descriptor_archivos_fd.pdf
- 2022 household dictionary (all incomes and expenditures quarterly):
  https://www.inegi.org.mx/rnm/index.php/catalog/901/data-dictionary/F64
- 2022 employed-member definition:
  https://www.inegi.org.mx/rnm/index.php/catalog/901/variable/F64/V3628?name=ocupados
- Published price-comparable 2020/2022 income results:
  https://www.inegi.org.mx/contenidos/saladeprensa/boletines/2023/ENIGH2022.pdf
