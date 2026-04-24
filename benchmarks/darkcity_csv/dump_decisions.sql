-- ══════════════════════════════════════════════════════════════════
-- DarkCity decision-log dump for Fathom CSV-to-P&L analysis
-- ══════════════════════════════════════════════════════════════════
--
-- Run on the DarkCity production Postgres (Railway). Produces a
-- JSON array, one row per scored agent decision, that pipes into
-- benchmarks/darkcity_csv/analyze.py.
--
-- Usage (Railway Postgres CLI):
--   psql $DATABASE_URL -At -f dump_decisions.sql > decisions.json
--
-- Or paste into Railway dashboard → Postgres → Query, export result.
-- ══════════════════════════════════════════════════════════════════

WITH decisions AS (
  SELECT
    d.id                                        AS eval_id,
    d.citizen_id                                AS agent_id,
    d.action_type                               AS action_type,
    d.reasoning_trace                           AS reasoning_trace,
    d.depth_score                               AS depth_score,
    d.depth_tier                                AS depth_tier,
    d.agent_state                               AS agent_state,
    d.counterfactual                            AS counterfactual,
    d.chain_id                                  AS chain_id,
    d.created_at                                AS created_at
  FROM depth_evaluations d
  WHERE d.reasoning_trace IS NOT NULL
    AND d.depth_score IS NOT NULL
    AND length(d.reasoning_trace) >= 20
  ORDER BY d.created_at DESC
  LIMIT 2000
),
with_transfers AS (
  SELECT
    x.*,
    COALESCE(
      (SELECT SUM(t.amount)::float
       FROM styxx_transfers t
       WHERE t.to_agent_id = x.agent_id
         AND t.created_at BETWEEN x.created_at - INTERVAL '5 minutes'
                              AND x.created_at + INTERVAL '5 minutes'),
      0.0
    ) AS styxx_inflow_5min,
    COALESCE(
      (SELECT SUM(t.amount)::float
       FROM styxx_transfers t
       WHERE t.from_agent_id = x.agent_id
         AND t.created_at BETWEEN x.created_at - INTERVAL '5 minutes'
                              AND x.created_at + INTERVAL '5 minutes'),
      0.0
    ) AS styxx_outflow_5min
  FROM decisions x
)
SELECT json_agg(row_to_json(with_transfers)) FROM with_transfers;
