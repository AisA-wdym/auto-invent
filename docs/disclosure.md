# Publishing a round's evidence

A published score nobody can decompose is a number to be trusted rather than checked. This describes
what a round must disclose once execution closes, what is already computed, and what has to change to
get it out to a reader.

## What is published today

The public round document carries the problems after disclosure, each laboratory's gate failures and
RCC spent, and the standings with daily and rolling scores.

That is the outcome without the working. A miner who scores badly is told a number. A third party
auditing the subnet cannot see whether the judges were sane, whether a criterion was scored at all,
or what the laboratory actually produced.

## What must be published

Per round: the problems, the pack hash, the generator split, and the rejection counts from
generation.

Per laboratory, per challenge:

| Evidence | Why a reader needs it |
|---|---|
| The raw portfolio | What the laboratory actually produced |
| The canonical fact sheet | What the judges saw, after presentation was stripped |
| All eight criterion scores | A total nobody can decompose cannot be checked |
| Whether the mechanism floor applied | It caps two criteria at 0.50, which explains a low total |
| Omitted criteria | A criterion no judge could score is the validator's gap, not the miner's |
| Every judge vote, with its reasoning and whether it abstained | Abstention rate is how a broken panel is spotted |
| Every pairwise verdict, both presentation orders | Order sensitivity is measurable only from the pair |
| Gate outcomes for all thirteen | One fatal gate invalidates a response; a reader must see which |
| The receipt chain | Every external call in order, hash-linked |
| Measured usage against the laboratory's claim | A claim that disagrees with measurement is evidence |
| Prior-art resolution per citation | A citation the validator could not resolve fails the portfolio |

The receipt chain is the strongest of these. Each entry commits to the previous one, so the record
cannot be reordered or trimmed after the fact — a validator cannot publish a flattering trace of a
laboratory it scored badly.

## What already exists

All of it is computed inside `validator.rounds.score_round` and discarded when the function returns:

    canonical  → the fact sheets, keyed (uid, challenge_id)
    screens    → per-criterion aggregates, keyed the same
    pairwise   → the tournament's verdicts
    labs       → the reduced per-challenge and daily scores

`ChallengeScore` already keeps `criteria_ppm`, `mechanism_floor_applied` and `omitted_criteria`.
`PointwiseScore` already keeps `reasoning` and `abstained`. `Execution` already carries the receipt
calls, the measured usage and the gate outcomes.

One thing is not carried: `_screen_all` aggregates the individual votes and returns only the
aggregate, so the reasoning text is lost. It should return both.

## The shape to store it in

Not one document. A round at mainnet scale is twenty challenges against every registered laboratory,
each with five ideas judged by three model families in both presentation orders. That is tens of
megabytes, and a page that must load all of it to show anything will not load at all.

    {ns}:public:{validator}:{date}            the summary, as now, plus an index of what detail exists
    {ns}:detail:{validator}:{date}:{uid}      one laboratory's full evidence for that round

The summary names the uids that have detail, so the dashboard fetches one laboratory at a time when a
reader opens it.

## The constraint that matters

**Detail is under the same seal as the problems.** `public_view` withholds `challenges` until
`disclosed()`, and the detail documents contain the problems by implication — a canonical fact sheet
is an answer to a specific problem, and a portfolio quotes it. A detail key written during execution
would hand a laboratory the problems it has not been given yet.

So the write path must refuse to publish any detail before disclosure, by construction rather than by
a filter: build the detail documents only inside the `disclosed()` branch, so there is no object for
a rendering change to start writing early. The existing assertion in `PublicOnlyRedisStore.write`
should cover the detail keys too, and a test should pin it by attempting a write in each undisclosed
phase.

## Order of work

1. `_screen_all` returns the individual votes as well as the aggregate.
2. A `validator/disclosure.py` that builds one laboratory's evidence from the artefacts in
   `score_round`, with no chain, no clock and no store — so it is testable as a function.
3. `RoundScores` carries the evidence; the driver hands it to the store.
4. `PublicOnlyRedisStore` writes the detail keys, inside the disclosure branch only.
5. The dashboard gains `/api/round/{date}/detail/{uid}` and a per-laboratory view.

Steps 1 to 4 are the subnet's; step 5 cannot be written against a contract that does not exist yet.
