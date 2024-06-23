"""
Here, I have created a sample A/B testing to test user interactions with our two Product Description
Models and two Recommendation Systems.
We can generate random A/B test results and then do basic analysis
"""

import random
import pandas as pd
from datetime import datetime
import Recommendation_System1 , Recommendation_System2
import Description_Generation


# Content generation strategies
def generate_description_variant_a(product_attributes):
    # Simple descriptive strategy
    prompt = Description_Generation.construct_prompt(product_attributes)
    models = ["EleutherAI/gpt-neo-2.7B"] #,"gpt2"]

    description = Description_Generation.generate_descriptions(models , prompt)
    return description
    
def generate_description_variant_b(product_attributes):
    # More creative and engaging strategy
    prompt = Description_Generation.construct_prompt(product_attributes)
    models = ["gpt2"]

    description = Description_Generation.generate_descriptions(models , prompt)
    return description

    
# Recommendation algorithms
def recommend_products_variant_a(user_id):
    # Simple popularity-based recommendation
    recommended_products = Recommendation_System1.recommend_products(user_id)
    return recommended_products #[101, 102, 103]

def recommend_products_variant_b(user_id):
    # Collaborative filtering recommendation
    recommended_products = Recommendation_System2.recommend_products(user_id)
    return recommended_products #[104, 105, 106]

# AB test
def ab_test(user_id, variants):
    # Randomly assign the user to one of the variants
    chosen_variant = random.choice(variants)
    return chosen_variant


def simulate_user_interactions(user_id, variant, product_attributes):
    # Generate product description
    if variant == 'content_variant_a':
        description = generate_description_variant_a(product_attributes)
    else:
        description = generate_description_variant_b(product_attributes)
    
    # Get product recommendations
    if variant == 'recommendation_variant_a':
        recommendations = recommend_products_variant_a(user_id)
    else:
        recommendations = recommend_products_variant_b(user_id)
    
    # Simulate interactions (views, clicks)
    interactions = {
        'user_id': user_id,
        'variant': variant,
        'description': description,
        'recommendations': recommendations,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return interactions

# Create a DataFrame to store interactions
interactions_df = pd.DataFrame(columns=['user_id', 'variant', 'description', 'recommendations', 'timestamp'])

# Simulate interactions for a set of users
user_ids = [1, 2, 3, 4, 5]
product_attributes = {
    'product_category' : 'Mobile Phone',
    'product_name' : 'Samsung S21',
    'price' : 50000,
    'features' : 'Ultrathin , Silver Color',
    'rating' : 4
}

for user_id in user_ids:
    for variant_type in ['content', 'recommendation']:
        chosen_variant = ab_test(user_id, [f'{variant_type}_variant_a', f'{variant_type}_variant_b'])
        interaction = simulate_user_interactions(user_id, chosen_variant, product_attributes)
        interactions_df = interactions_df.append(interaction, ignore_index=True)

print(interactions_df)


# Simple analysis of interactions
content_interactions = interactions_df[interactions_df['variant'].str.contains('content')]
recommendation_interactions = interactions_df[interactions_df['variant'].str.contains('recommendation')]

# Count interactions per variant
content_counts = content_interactions['variant'].value_counts()
recommendation_counts = recommendation_interactions['variant'].value_counts()

print("Content Variant Counts:\n", content_counts)
print("Recommendation Variant Counts:\n", recommendation_counts)

# Further analysis can include conversion rates, engagement metrics, etc.

