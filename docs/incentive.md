# The incentive mechanism

What gets rewarded, how the number is computed, and — where a design decision could have gone the
other way — why it went this way.

---

## The shape of it

```
  per idea      ──rank weights──▶  per criterion  ──criterion weights──▶  challenge score
  (five ideas)   0.40/0.25/0.15        (eight)         summing to 1.0            │
                 /0.12/0.08                                                      │
                                                                    20 challenges/day
                                                                                 │
                             daily score = 0.70·mean + 0.30·Q25 ◀────────────────┘
                                          │
                     rolling = median(7d)·0.60 + median(30d)·0.40
                                          │
                      softmax(τ=0.10) over the gap above the floor
                                          │
                              cap at 17.5% · burn the remainder
                                          │
                                  on-chain weight vector
```

Every ratio is a **parts-per-million integer**. No float reaches anything that gets hashed —
`protocol/canonical.py` refuses one and names the path where it found it, because a value read as 0.1
and a value computed as 0.05 + 0.05 are different doubles, and two hosts would then disagree about
the bytes.

---

## Per idea: rank weights, and why missing ranks are forfeit

The five positions carry 0.40, 0.25, 0.15, 0.12, 0.08.

A portfolio with three ideas scores the first three weights and **forfeits** the remaining 0.20. It is
not rescaled to the ranks present.

That distinction was a defect once, and the measurement is worth repeating: with redistribution, a
portfolio containing **one excellent idea** scored 900,000 ppm while a portfolio of **five genuinely
diverse ideas** scored 777,000. Padding beat diversity. The subnet would have been paying for
"submit your best idea and leave the rest empty", which is the opposite of what §1 asks for.

The general rule the fix expresses:

| Gap | Whose fault | What happens |
|---|---|---|
| a criterion no judge could score | the *validator's* | weight redistributes over the criteria present |
| a rank with no distinct idea | the *miner's* | weight is forfeit |

Both are "missing data". Only one of them is the miner's to fix.

---

## Per criterion: eight, summing to exactly one whole

| Criterion | Weight | The question |
|---|---|---|
| originality | 25.0% | materially different from prior art, or renamed? |
| value | 20.0% | would this plausibly create meaningful value? |
| mechanism | 15.0% | is there a causal chain, or a description of the effect? |
| constraint_fit | 12.0% | an answer to *this* problem under *these* constraints? |
| diversity | 10.0% | five directions, or five variations? |
| self_selection | 8.0% | was the strongest idea ranked first? |
| falsifiability | 7.0% | a prediction whose failure would change what you believe? |
| cost_reliability | 3.0% | a concrete next step at a defensible cost? |

The weights are asserted to sum to exactly `1,000,000` ppm, with **no tolerance**. A tolerance would
let a set that nearly sums to one pass, and "nearly" is a silent rescaling of every score computed
with it — a defect that changes rankings without changing anything visible.

### The mechanism floor

If **mechanism** scores below 0.40, **value and originality are capped at 0.50**.

This is the one place the criteria are not independent, and it is deliberate. An idea with no coherent
mechanism can still read as valuable and original — that combination is precisely what a plausible
non-idea looks like, and it is what a language model produces when it is generating confidently about
something it has not worked out. Without the floor, the highest-scoring strategy would be to write
compelling prose about a mechanism that does not exist.

---

## Per day: the lower quartile carries 30%

```
daily = 0.70 · mean(challenge scores) + 0.30 · Q25(challenge scores)
```

Twenty challenges. The mean rewards being good; the lower quartile punishes being *inconsistent*.

A laboratory that scores brilliantly on five problems and fails on fifteen is not a good laboratory —
it is a laboratory that fits five problems. Weighting the bottom quartile means breadth is worth
paying for, and it makes narrow overfitting visibly unprofitable rather than merely suboptimal.

---

## Rolling: median, and never rescaled

| History | Estimator |
|---|---|
| under 7 days | mean of what exists |
| 7 days or more | `0.60 · median(7d) + 0.40 · median(30d)` |

Median rather than mean over the rolling window, because one catastrophic day — a provider outage, a
malformed submission — should not erase a month of consistency.

The important property is what the estimator switch does *not* do: it never scales the result. A new
miner's mean and an established miner's median are on the same scale, so crossing the seven-day
boundary does not step a miner's score.

**There is no credibility multiplier.** §18.6 forbids one, and this implementation has none. A
multiplier that scaled a newcomer's score down would make the floor unreachable for exactly the
laboratories the subnet needs to attract, and it would be indistinguishable from incumbent
protection.

---

## Pairwise at 75%, anchored pointwise at 25%

```
combined = 0.75 · bradley_terry(pairwise verdicts) + 0.25 · anchored(pointwise)
```

**Pairwise dominates** because a comparison is a far more reliable judgement than a score. Asked "is
this portfolio worth 0.7?", a model answers differently on different days. Asked "which of these two
is better on mechanism?", it is consistent.

**Pointwise anchors it** because a pure tournament has no absolute scale. Bradley-Terry produces
relative strengths, and a field where everyone is weak would still produce a clear ranking — with a
leader that should not qualify. The anchored component is what makes "better than the reference
template" a meaningful sentence.

### The order swap is the measurement, not a refinement

Every pair is judged **twice**, once in each presentation order.

- both orders agree → a win for the agreed winner;
- the orders disagree → a **tie**.

A judge that preferred laboratory X in one order and Y in the other expressed a preference for a
*position*. Language models have a substantial position bias; a tournament that presented each pair
once would measure that bias and the answers together, inseparably.

Disagreements become ties rather than being discarded, because discarding would silently delete the
comparisons the judge found hardest — and those are exactly the near-equal pairs Swiss pairing exists
to produce. The disagreement *rate* is also the panel's bias measurement, which §19 compares against
a declared ceiling. It is not a nuisance to minimise away; it is the number that says whether the
panel can be trusted at all.

---

## Judge panels: the cap is on families, not routes

At least three model families per criterion, and no family may hold more than 40% of it.

Every judge is reached through OpenRouter, so "provider" in the *routing* sense is always
`openrouter` — a cap read that way is a cap on nothing. What matters is who **trained** the model
behind the route. Three routes to three Anthropic snapshots is **one family** and breaches the cap,
however many distinct slugs it uses, because two versions of one model share their failure modes.

This is why the season config *declares* a `family` per judge rather than deriving it from the slug: a
miner-hosted fine-tune routed through OpenRouter has a slug that says nothing about what trained it.

A consequence falls out for free. The challenge generators are GPT and Claude, and every panel
contains both — so a problem written by GPT is judged by a panel including Claude, and the reverse. No
family both sets a problem and unilaterally decides the answer.

---

## Weights: the burn, and the cap that does not always bind

**Qualification is absolute, not relative.** A miner qualifies by exceeding the reference template's
own rolling score. If nobody does, **100% of the emission burns.**

A rank-based floor would always pay someone. §20.4 refuses that: if no laboratory beats direct
frontier-model use, the subnet has not demonstrated that competing architectures add value, and
emitting anyway would be paying for the appearance of competition.

**Allocation is a softmax on the gap above the floor**, not on the ratio, at τ = 0.10. Scores are
bounded, so the gap is what carries information: 0.60 against 0.30 means something quite different in
a tight field than in a spread one.

**The cap is 17.5% — and it does not bind below six qualifiers.**

That last clause is the correction of a real defect. With two qualifiers, capping the leader leaves one
receiver whose headroom equals the overflow, so it lands on the cap too — and Bittensor renormalises
the vector it receives, so `[cap, cap]` becomes 50/50. Two laboratories with very different scores
would receive identical emission, and there would be no incentive to be the better of the two. The
flattening is arithmetic, not a tuning problem: the cap simply cannot bind until
`N ≥ ceil(1 / cap) = 6`.

---

## Hard gates cannot be compensated for

Thirteen deterministic gates. A failure invalidates the challenge response entirely — no partial
credit, no weighting. A high judge score on a portfolio that exceeded its budget is not a
consideration.

Eleven are decided from bytes and every validator reaches the same verdict. Two are not:

- **fabricated citation** requires fetching a URL, whose outcome varies with the day;
- **prompt injection** requires a judgement.

Both are handled narrowly and deliberately. The injection gate fires only on unambiguous forms — an
imperative addressed to a scorer — because on a *fatal* gate a false positive is less recoverable than
a missed subtle attempt, and the canonicalizer strips injections from the judged text regardless.

Where the evidence is not deterministic, the response is to **neutralise rather than to invalidate**.
That way a fatal consequence never depends on a non-reproducible input, and a laboratory cannot be
invalidated on one validator and pass on another.

---

## What the mechanism is trying to make true

Read together, the parts push toward one thing: **the profitable strategy should be to build a
laboratory that finds good ideas on problems it has not seen.**

Every rule above closes off a cheaper route to the same payment:

| Cheaper strategy | What closes it |
|---|---|
| write persuasively | the canonicalizer strips presentation before judging |
| submit one great idea | rank weights are forfeit, not redistributed |
| five variations on one idea | duplicates collapse to one lineage before scoring |
| confident prose about a mechanism that does not exist | the mechanism floor caps value and originality |
| fit a handful of problems | the lower quartile carries 30% of the day |
| invent supporting citations | gate 8, checked by resolution |
| instruct the judge | gate 9, plus stripping |
| be hard to search for, and call it novel | novelty confidence is a statement about the search, and never raises originality on its own |
| overfit to one generator's house style | half the pack comes from the other family, and the per-family score gap is measured |
| out-spend rivals | one RCC ceiling, measured by the gateway, not self-reported |
| be the best of a weak field | the floor is absolute; otherwise the emission burns |

None of that is a claim that the mechanism is unexploitable. It is a claim about where the remaining
exploits would have to live — and §27's measurement gates exist to find them before mainnet, by
checking that the validator ranks deliberately strong, weak, copied, impossible and superficially
novel portfolios in the right order.
