# Security

## Reporting

Email **coolaisaworld@gmail.com** with `[auto-invent security]` in the subject. Please do not open a
public issue for anything that affects the reward path or a credential.

Include what you can reproduce. A working exploit against a local validator is more useful than a
description, and the localnet fixtures in `tests/localnet/` exist to make one cheap to build.

## What we consider a vulnerability

The subnet's job is to pay for research architecture. Anything that lets a miner get paid for
something else is in scope, and so is anything that lets a validator decide who gets paid.

**Reward-path integrity**

- Reading a challenge before it was issued to you, or reading one issued to another laboratory.
- Any way to make a validator generate a pack after seeing a submission — a break in the
  salt → randomness → pack-hash ordering (§7.3).
- Making a hard gate pass when it should fail, or fail when it should pass. Both matter: a gate is
  fatal, so a false failure is as damaging as a false pass and less recoverable.
- Scoring above the reference floor without a laboratory that earns it.
- Influencing another miner's score.

**Credential and budget separation**

- Any path by which a laboratory obtains a provider credential. It receives a session token bound to
  one run and one challenge, and nothing else (§5.4.1).
- Any path by which validator work is billed to a miner's account, or the reverse (§3.4.4). Under a
  single provider surface this *succeeds* silently, so it is the failure we most want reported.
- Spending outside the meter — any egress from the sandbox that is not the gateway.
- Exceeding a round ceiling by more than a single in-flight call.

**Isolation**

- Escaping the laboratory container, or reaching the validator process, Redis, or the host from it.
- Making one laboratory's run affect another's measured wall time or budget.

## What is not a vulnerability

- **Validators disagreeing.** Each generates its own problems, so they are *expected* to rank
  differently — §27 requires cross-validator rank correlation of only 0.60. Same-bundle rerun
  correlation must be 0.80, and a break in *that* is a real finding.
- **Reading another miner's published bundle.** §6.3 publishes source, prompts and orchestration
  after execution closes, deliberately. Forking a rival's design in the next round is the intended
  mechanic, not an attack.
- **A laboratory spending its whole budget.** That is what the budget is for.
- **The portal showing standings.** It is meant to. A portal showing *problems* before execution
  closes would be a finding.

## What we have deliberately accepted

Stated here rather than left to be discovered:

- **The validator runs with the Docker socket.** A process with the socket can start a privileged
  container. The mitigation is that nothing miner-controlled reaches that process: laboratory code
  runs in a sibling container under `--cap-drop=ALL --read-only --user 1000:1000`, and the only thing
  crossing back is a file read with a bounded read. If you can get miner-controlled bytes to the
  validator process itself, that is a finding.
- **A `declared_spend_cap_usd` cannot be verified.** The protocol does not control a miner's
  OpenRouter account. It is recorded and reconciled against provider-reported usage rather than
  trusted (§27), and a discrepancy is treated as an incident.
- **Judges are language models.** Their verdicts are estimates. The order-swap measurement, the
  three-family cap and the calibration floors bound how wrong they can be; they do not make them
  right. A systematic bias that survives all three is a finding worth reporting.
