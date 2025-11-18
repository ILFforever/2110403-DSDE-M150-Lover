{{
    config(
        materialized='view',
        tags=['staging']
    )
}}

WITH source AS (
    SELECT * FROM {{ source('traffy', 'complaints_raw') }}
),

cleaned AS (
    SELECT
        ticket_id,
        CAST(timestamp AS TIMESTAMP) AS complaint_timestamp,
        CAST(last_activity AS TIMESTAMP) AS last_activity_timestamp,
        type,
        organization,
        comment,
        district,
        subdistrict,
        province,
        state,

        -- Parse coordinates
        CAST(SPLIT_PART(coords, ',', 1) AS FLOAT) AS longitude,
        CAST(SPLIT_PART(coords, ',', 2) AS FLOAT) AS latitude,

        -- Calculate resolution time
        DATEDIFF('day',
                CAST(timestamp AS TIMESTAMP),
                CAST(last_activity AS TIMESTAMP)) AS solve_days,

        -- Extract temporal features
        EXTRACT(YEAR FROM CAST(timestamp AS TIMESTAMP)) AS year,
        EXTRACT(MONTH FROM CAST(timestamp AS TIMESTAMP)) AS month,
        EXTRACT(DAY FROM CAST(timestamp AS TIMESTAMP)) AS day,
        EXTRACT(HOUR FROM CAST(timestamp AS TIMESTAMP)) AS hour,
        EXTRACT(DOW FROM CAST(timestamp AS TIMESTAMP)) AS day_of_week,

        -- Status flags
        CASE WHEN state = 'เสร็จสิ้น' THEN 1 ELSE 0 END AS is_completed,
        CASE WHEN state = 'กำลังดำเนินการ' THEN 1 ELSE 0 END AS is_in_progress,

        -- Category flags
        CASE WHEN type LIKE '%น้ำท่วม%' THEN 1 ELSE 0 END AS is_flood,
        CASE WHEN type LIKE '%จราจร%' OR type LIKE '%ถนน%' THEN 1 ELSE 0 END AS is_traffic,
        CASE WHEN type LIKE '%ความสะอาด%' OR type LIKE '%ขยะ%' THEN 1 ELSE 0 END AS is_waste,
        CASE WHEN type LIKE '%ทางเท้า%' THEN 1 ELSE 0 END AS is_sidewalk,

        -- Metadata
        CURRENT_TIMESTAMP AS dbt_updated_at

    FROM source
    WHERE province LIKE '%กรุงเทพ%'
        AND district IS NOT NULL
        AND subdistrict IS NOT NULL
        AND timestamp IS NOT NULL
)

SELECT * FROM cleaned
