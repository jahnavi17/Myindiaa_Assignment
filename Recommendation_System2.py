"""
This is a simple Recommendation System that uses dot product of user and product matrices to 
recommend products.
This model also recommends top 3 products to users.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd


# Mock data setup
user_ids = [1, 1, 2, 2, 3, 3]
product_ids = [101, 102, 101, 103, 102, 104]
interactions = ['view', 'click', 'view', 'purchase', 'view', 'click']

user2user_encoded = {x: i for i, x in enumerate(set(user_ids))}
product2product_encoded = {x: i for i, x in enumerate(set(product_ids))}

num_users = len(user2user_encoded)
num_products = len(product2product_encoded)
embedding_size = 50


# Recommender Model
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_products, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_products = num_products
        self.embedding_size = embedding_size

        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        self.product_embedding = tf.keras.layers.Embedding(num_products, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))

        self.user_bias = tf.keras.layers.Embedding(num_users, 1)
        self.product_bias = tf.keras.layers.Embedding(num_products, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        product_vector = self.product_embedding(inputs[:, 1])

        user_bias = self.user_bias(inputs[:, 0])
        product_bias = self.product_bias(inputs[:, 1])

        dot_user_product = tf.tensordot(user_vector, product_vector, 2)

        x = dot_user_product + user_bias + product_bias

        return tf.nn.sigmoid(x)

x = np.array([[user2user_encoded[u], product2product_encoded[p]] for u, p in zip(user_ids, product_ids)])
y = np.array([1 if i in ['click', 'purchase'] else 0 for i in interactions])

# Reshape labels to have shape (num_samples, 1)
y = y.reshape(-1, 1)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

model = RecommenderNet(num_users, num_products, embedding_size)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5, batch_size=2)

product2product_original = {v: k for k, v in product2product_encoded.items()}

def recommend_products(user_id) :#, model, user2user_encoded, product2product_encoded, product2product_original):
    # Get the encoded user ID
    user_encoded = user2user_encoded.get(user_id)
    if user_encoded is None:
        print("User not found")
        return []
    
    # Generate predictions for all products
    product_ids = list(product2product_encoded.values())
    user_product_array = [[user_encoded, product] for product in product_ids]
    
    predictions = model.predict(user_product_array)
    predictions = tf.squeeze(predictions).numpy()
    
    # Get the indices of the top predicted products
    top_indices = predictions.argsort()[-3:][::-1]
    recommended_product_ids = [product2product_original[product_ids[i]] for i in top_indices]
    
    return recommended_product_ids

# Example usage

user_id = 1
recommended_products = recommend_products(user_id) #, model, user2user_encoded, product2product_encoded, product2product_original)
print("Recommended Products for User 1:", recommended_products)
