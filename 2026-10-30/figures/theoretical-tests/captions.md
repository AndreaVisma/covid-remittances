# Theoretical tests (13 Nov 2026)

Made by `2026-10-30/theoretical_tests.py` (2019 calibration, simplified forms; exact computation). Data: `theoretical_tests.csv`.

## T1_response_to_unemployment: How the system responds to unemployment

Expected monthly transfers (left) and households receiving (right) under 2019 conditions when the unemployment rate of every occupation rises by the same number of percentage points, relative to 2019, for three levels of stepping-in. Dotted: a fall proportional to the number of migrants in work. At +10 points transfers fall by 10% without stepping-in and by 3% with full stepping-in. Exact computation over all MMP migrant households, 2019 calibration, no need response.

## T2_household_buffer: The household buffer: who can step in

Left: household-months in which a migrant who wants to send is out of work, under 2019 conditions with unemployment raised in every occupation (all other household-months are untouched). Absorbed: the household's reserve (migrants who can send but were not planning to) is at least as large as its lost senders. Short: it is not. With full stepping-in, effective senders equal min(D, A), the smaller of how many want to send and how many can; only short households lose transfers. In 2019, 0.4% of household-months are short; at +10 points 1.3%, while 3.0% are absorbed. Right: expected lost senders per month and how many of them a reserve migrant of the same household could replace (orange gap: no one available).

## T3_unemployment_x_need: Unemployment and need interact

Left: transfers under unemployment as a share of transfers at the same level of need without the unemployment increase, with full stepping-in. Rows shift every migrant's desire by the same amount on the log-odds scale (as the crisis term β·N* does); in brackets the resulting share of migrants who want to send. Right: share of transfers lost when unemployment rises by 10 points, by level of need and stepping-in. Higher need moves households from wanting fewer transfers than they can send towards wanting more, so the reserve that absorbs job losses shrinks.

## T4_who_bears_the_loss: Who bears the loss

Change in expected receipt when unemployment rises by 10 points in every occupation, by network type and number of linked migrants, without and with full stepping-in (2019 conditions, no need response). Households with one migrant have nobody to step in, so stepping-in cannot protect them; larger networks are protected almost fully.

## T5_two_margins: Two margins: who receives and how much

Log change in the number of households receiving (blue) and in transfers per receiving household (orange); the two add up to the log change in transfers. Left: unemployment raised in every occupation, without (dashed) and with (solid) full stepping-in. Right: every migrant's desire shifted on the log-odds scale, κ = 1, 2019 employment.
