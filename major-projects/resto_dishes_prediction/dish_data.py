import random
import pandas as pd
import numpy as np

def _generate_cuisine_dataset(
    cuisine: str,
    n_rows: int,
    random_state: int,
    flavor_profile: dict,
    price_ranges: dict,
    visibility_bias: tuple
) -> pd.DataFrame:
    """
    Internal helper to generate cuisine-specific but non-biased datasets.
    """

    random.seed(random_state)
    np.random.seed(random_state)

    course_types = ["Starter", "Main", "Dessert"]
    price_categories = ["budget", "mid", "premium"]

    data = []

    for i in range(n_rows):
        # --- Flavor profile (culture-aware but flexible) ---
        spiciness = int(np.clip(
            np.random.normal(flavor_profile["spice_mean"], 1), 1, 5
        ))
        sweetness = int(np.clip(
            np.random.normal(flavor_profile["sweet_mean"], 1), 1, 5
        ))
        saltiness = int(np.clip(
            np.random.normal(flavor_profile["salt_mean"], 1), 2, 5
        ))
        umami = int(np.clip(
            np.random.normal(flavor_profile["umami_mean"], 1), 2, 5
        ))

        flavor_balance = round(
            10 - abs(spiciness - sweetness) - abs(saltiness - umami)
            + random.uniform(-1, 1),
            1
        )
        flavor_balance = max(1, min(10, flavor_balance))

        # --- Sensory ---
        texture = random.randint(2, 5)
        aroma = random.randint(2, 5)
        presentation = random.randint(2, 5)
        portion = random.randint(2, 5)

        # --- Pricing ---
        price_category = random.choice(price_categories)
        price = random.randint(*price_ranges[price_category])

        # --- Operations ---
        prep_time = random.randint(5, 25)
        cook_time = random.randint(5, 30)
        ingredient_cost = round(price * random.uniform(0.25, 0.45), 2)
        complexity = random.randint(1, 5)
        availability = random.randint(2, 5)

        # --- Customer metrics ---
        avg_rating = round(
            (flavor_balance / 2 + presentation + aroma) / 3
            + random.uniform(-0.6, 0.6),
            2
        )
        avg_rating = max(1.0, min(5.0, avg_rating))

        rating_count = random.randint(30, 450)

        repeat_rate = round((avg_rating / 5) * random.uniform(0.35, 0.85), 2)
        complaint_rate = round(random.uniform(0.02, 0.18), 2)
        return_rate = round(random.uniform(0.01, 0.09), 2)

        # --- Sales ---
        avg_daily_orders = int(avg_rating * random.uniform(7, 16))
        peak_orders = int(avg_daily_orders * random.uniform(0.3, 0.6))
        growth_rate = round(random.uniform(-0.15, 0.35), 2)
        menu_visibility = random.randint(*visibility_bias)

        # --- Target logic (adjusted for more realistic distribution) ---
        success_score = (
            avg_rating * 2
            + repeat_rate * 5
            + menu_visibility
            - complaint_rate * 3
            - (price / 320)
            + random.uniform(-0.5, 0.5)  # Add more variability
        )

        # Adjust the thresholds to balance success, average, and unsuccessful outcomes
        performance_prob = random.uniform(0, 1)

        if success_score >= 9 and performance_prob > 0.25:
            performance = "successful"
        elif success_score >= 6 and performance_prob > 0.1:
            performance = "average"
        else:
            performance = "unsuccessful"

        data.append({
            "dish_id": f"{cuisine[:3].upper()}_{i + 1}",
            "cuisine_type": cuisine,
            "course_type": random.choice(course_types),
            "spiciness_level": spiciness,
            "sweetness_level": sweetness,
            "saltiness_level": saltiness,
            "umami_level": umami,
            "flavor_balance_score": flavor_balance,
            "texture_rating": texture,
            "aroma_intensity": aroma,
            "presentation_score": presentation,
            "portion_satisfaction": portion,
            "price": price,
            "price_category": price_category,
            "prep_time_minutes": prep_time,
            "cook_time_minutes": cook_time,
            "ingredient_cost": ingredient_cost,
            "complexity_score": complexity,
            "ingredient_availability_score": availability,
            "avg_customer_rating": avg_rating,
            "rating_count": rating_count,
            "repeat_order_rate": repeat_rate,
            "complaint_rate": complaint_rate,
            "return_rate": return_rate,
            "avg_daily_orders": avg_daily_orders,
            "peak_hour_orders": peak_orders,
            "order_growth_rate": growth_rate,
            "menu_visibility_score": menu_visibility,
            "performance_tier": performance
        })

    return pd.DataFrame(data)

def generate_themed_menu_datasets(
    n_rows_per_cuisine: int = 250,
    random_state: int = 42
):
    """
    Generate 3 culturally-aware but unbiased themed restaurant datasets.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (indian_df, japanese_df, italian_df)
    """

    indian_df = _generate_cuisine_dataset(
        cuisine="Indian",
        n_rows=n_rows_per_cuisine,
        random_state=random_state,
        flavor_profile={
            "spice_mean": 3.6,
            "sweet_mean": 2.1,
            "salt_mean": 3.8,
            "umami_mean": 4.2
        },
        price_ranges={
            "budget": (120, 220),
            "mid": (250, 420),
            "premium": (450, 750)
        },
        visibility_bias=(2, 5)
    )

    japanese_df = _generate_cuisine_dataset(
        cuisine="Japanese",
        n_rows=n_rows_per_cuisine,
        random_state=random_state + 1,
        flavor_profile={
            "spice_mean": 1.3,
            "sweet_mean": 1.8,
            "salt_mean": 3.2,
            "umami_mean": 4.5
        },
        price_ranges={
            "budget": (180, 280),
            "mid": (320, 480),
            "premium": (500, 900)
        },
        visibility_bias=(2, 4)
    )

    italian_df = _generate_cuisine_dataset(
        cuisine="Italian",
        n_rows=n_rows_per_cuisine,
        random_state=random_state + 2,
        flavor_profile={
            "spice_mean": 1.7,
            "sweet_mean": 2.0,
            "salt_mean": 3.4,
            "umami_mean": 3.9
        },
        price_ranges={
            "budget": (160, 260),
            "mid": (300, 450),
            "premium": (480, 800)
        },
        visibility_bias=(2, 5)
    )

    return indian_df, japanese_df, italian_df
