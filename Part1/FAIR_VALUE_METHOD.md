# Fair Value Method

Historical salary is not the same as current market value. Players may be on
rookie-scale deals or extensions signed before a breakout season. The project
therefore reports two different estimates:

1. **Historical contract benchmark** — the machine-learning model's estimate
   from observed 2010–2025 contracts.
2. **Fair annual value** — a salary-cap-aware decision-support estimate based
   on peer performance, years of service, honors, and maximum-salary rules.

## Calculation

- Player statistics are converted to a weighted empirical percentile against
  the 7,296 player-seasons in the project dataset.
- **Stats season** controls the model's historical comparison year, while
  **contract start year** independently controls the salary cap. If the
  contract starts later, elapsed seasons are added to years of service before
  determining maximum-contract eligibility.
- Years of service determine a simplified 25%, 30%, or 35% maximum-salary tier.
- MVP or All-NBA status can unlock the 35% designated-veteran tier for players
  with at least seven years of service.
- All-Star, All-NBA, and MVP selections provide explicit, visible adjustments.
- Fair value cannot exceed the estimated eligible first-year maximum.
- A four-year illustration applies 8% annual raises for planning purposes.
- Salary caps after 2026 are projections using 10% annual growth and are labeled
  as projected in the interface.

This is a transparent valuation framework, not an implementation of every NBA
CBA exception. Options, trade bonuses, extension timing, award lookback rules,
and team-specific Bird-rights details require a contract specialist.
