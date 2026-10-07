-- ---------------------------------------------------------------------------
-- feature_engineering.sql
--
-- All row-relative features are computed in SQL with window functions and
-- pushed down to SQLite, so the heavy ranking/aggregation runs in the engine
-- (over the whole 150k-row table at once) instead of row-by-row in pandas.
-- Python then reads the finished `loans_features` table and trains on it.
--
-- Window functions used:
--   * NTILE(100) OVER (ORDER BY income)      -> income_percentile
--   * NTILE(5)   OVER (ORDER BY composite)   -> risk_tier bucketing
--   * AVG(...)   OVER (PARTITION BY age_band) -> peer-relative delinquency
-- ---------------------------------------------------------------------------

DROP TABLE IF EXISTS loans_features;

CREATE TABLE loans_features AS
WITH base AS (
    SELECT
        rowid AS loan_id,
        *,
        -- Aggregated delinquency scoring: weight later-stage delinquency more.
        (1.0 * "NumberOfTime30-59DaysPastDueNotWorse"
       + 2.0 * "NumberOfTime60-89DaysPastDueNotWorse"
       + 4.0 * "NumberOfTimes90DaysLate") AS delinquency_score,
        -- Coarse age band used for peer comparisons and dashboard segmentation.
        CASE
            WHEN age < 30 THEN '<30'
            WHEN age < 40 THEN '30-39'
            WHEN age < 50 THEN '40-49'
            WHEN age < 60 THEN '50-59'
            WHEN age < 70 THEN '60-69'
            ELSE '70+'
        END AS age_band
    FROM loans_raw
),
scored AS (
    SELECT
        base.*,
        -- Peer-relative delinquency: how far this borrower sits above/below the
        -- average delinquency of everyone in the same age band.
        AVG(delinquency_score) OVER (PARTITION BY age_band) AS peer_delinquency_avg,
        -- Income percentile across the whole book (NULL incomes sort to the
        -- bottom, which is a reasonable "unknown ~ risky" default).
        NTILE(100) OVER (ORDER BY "MonthlyIncome") AS income_percentile,
        -- Composite score used only to rank borrowers into risk tiers. It's a
        -- transparent heuristic (utilization + delinquency + debt), not the
        -- model output; the model learns its own weighting downstream.
        (MIN("RevolvingUtilizationOfUnsecuredLines", 3.0)
         + delinquency_score
         + MIN("DebtRatio", 2.0)) AS composite_risk
    FROM base
),
tiered AS (
    SELECT
        scored.*,
        (delinquency_score - peer_delinquency_avg) AS peer_delinquency_gap,
        -- Risk-tier bucketing: 1 = safest quintile ... 5 = riskiest quintile.
        NTILE(5) OVER (ORDER BY composite_risk) AS risk_tier
    FROM scored
)
SELECT
    loan_id,
    "RevolvingUtilizationOfUnsecuredLines",
    age,
    age_band,
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    delinquency_score,
    peer_delinquency_gap,
    income_percentile,
    risk_tier,
    "SeriousDlqin2yrs"
FROM tiered;

CREATE INDEX IF NOT EXISTS idx_features_tier ON loans_features(risk_tier);
CREATE INDEX IF NOT EXISTS idx_features_age  ON loans_features(age_band);
