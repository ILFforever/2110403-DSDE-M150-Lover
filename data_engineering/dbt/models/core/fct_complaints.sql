{{
    config(
        materialized='table',
        tags=['core', 'fact']
    )
}}

WITH stg_complaints AS (
    SELECT * FROM {{ ref('stg_complaints') }}
),

enriched AS (
    SELECT
        ticket_id,
        complaint_timestamp,
        last_activity_timestamp,
        type,
        organization,
        district,
        subdistrict,
        longitude,
        latitude,
        state,
        solve_days,

        -- Temporal dimensions
        year,
        month,
        day,
        hour,
        day_of_week,

        -- Derived metrics
        CASE
            WHEN solve_days <= 7 THEN 'fast'
            WHEN solve_days <= 30 THEN 'moderate'
            WHEN solve_days <= 90 THEN 'slow'
            ELSE 'very_slow'
        END AS resolution_speed_category,

        CASE
            WHEN month BETWEEN 5 AND 10 THEN 'rainy'
            ELSE 'dry'
        END AS season,

        CASE
            WHEN day_of_week IN (0, 6) THEN 1
            ELSE 0
        END AS is_weekend,

        CASE
            WHEN hour BETWEEN 6 AND 12 THEN 'morning'
            WHEN hour BETWEEN 12 AND 18 THEN 'afternoon'
            WHEN hour BETWEEN 18 AND 22 THEN 'evening'
            ELSE 'night'
        END AS time_of_day,

        -- Status flags
        is_completed,
        is_in_progress,

        -- Category flags
        is_flood,
        is_traffic,
        is_waste,
        is_sidewalk,

        -- Quality score (example heuristic)
        CASE
            WHEN solve_days <= 7 AND is_completed = 1 THEN 5
            WHEN solve_days <= 14 AND is_completed = 1 THEN 4
            WHEN solve_days <= 30 AND is_completed = 1 THEN 3
            WHEN solve_days <= 90 THEN 2
            ELSE 1
        END AS service_quality_score,

        dbt_updated_at

    FROM stg_complaints
)

SELECT * FROM enriched
