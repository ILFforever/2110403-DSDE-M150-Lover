{{
    config(
        materialized='table',
        tags=['mart', 'analytics']
    )
}}

WITH fct_complaints AS (
    SELECT * FROM {{ ref('fct_complaints') }}
),

district_metrics AS (
    SELECT
        district,
        year,
        month,

        -- Volume metrics
        COUNT(DISTINCT ticket_id) AS total_complaints,
        COUNT(DISTINCT CASE WHEN is_completed = 1 THEN ticket_id END) AS completed_complaints,
        COUNT(DISTINCT CASE WHEN is_flood = 1 THEN ticket_id END) AS flood_complaints,
        COUNT(DISTINCT CASE WHEN is_traffic = 1 THEN ticket_id END) AS traffic_complaints,
        COUNT(DISTINCT CASE WHEN is_waste = 1 THEN ticket_id END) AS waste_complaints,

        -- Performance metrics
        ROUND(AVG(solve_days), 2) AS avg_resolution_days,
        ROUND(MEDIAN(solve_days), 2) AS median_resolution_days,
        MIN(solve_days) AS min_resolution_days,
        MAX(solve_days) AS max_resolution_days,

        -- Completion rate
        ROUND(100.0 * COUNT(DISTINCT CASE WHEN is_completed = 1 THEN ticket_id END) /
              NULLIF(COUNT(DISTINCT ticket_id), 0), 2) AS completion_rate_pct,

        -- Quality metrics
        ROUND(AVG(service_quality_score), 2) AS avg_quality_score,

        -- Seasonal patterns
        COUNT(DISTINCT CASE WHEN season = 'rainy' THEN ticket_id END) AS rainy_season_complaints,
        COUNT(DISTINCT CASE WHEN season = 'dry' THEN ticket_id END) AS dry_season_complaints,

        -- Temporal patterns
        COUNT(DISTINCT CASE WHEN is_weekend = 1 THEN ticket_id END) AS weekend_complaints,
        COUNT(DISTINCT CASE WHEN is_weekend = 0 THEN ticket_id END) AS weekday_complaints,

        -- Metadata
        CURRENT_TIMESTAMP AS dbt_updated_at

    FROM fct_complaints
    GROUP BY district, year, month
),

ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY year, month ORDER BY total_complaints DESC) AS volume_rank,
        ROW_NUMBER() OVER (PARTITION BY year, month ORDER BY avg_resolution_days ASC) AS efficiency_rank
    FROM district_metrics
)

SELECT * FROM ranked
