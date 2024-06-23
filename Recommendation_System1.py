"""
This Recommender System uses Tensorflow to craete a Matrix Factorization Recommendation System.
This is a Form of Collaborative Recommendation system where products are recommended based on
user's similarity to other users with similar interests.
This model recommends top three products to users.
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Mock data setup
user_ids = [1, 1, 2, 2, 3, 3]
product_ids = [101, 102, 101, 103, 102, 104]
interactions = ['view', 'click', 'view', 'purchase', 'view', 'click']

# Encoding user and product IDs
user2user_encoded = {x: i for i, x in enumerate(set(user_ids))}
product2product_encoded = {x: i for i, x in enumerate(set(product_ids))}

num_users = len(user2user_encoded)
num_products = len(product2product_encoded)

# Preparing the interaction data
x = np.array([[user2user_encoded[u], product2product_encoded[p]] for u, p in zip(user_ids, product_ids)])
y = np.array([1 if i in ['click', 'purchase'] else 0 for i in interactions])

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Recommendation Model
class MatrixFactorization(tf.keras.Model):
    def __init__(self, num_users, num_products, latent_dim, **kwargs):
        super(MatrixFactorization, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_products = num_products
        self.latent_dim = latent_dim

        self.user_embedding = tf.keras.layers.Embedding(num_users, latent_dim, embeddings_initializer='he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        self.product_embedding = tf.keras.layers.Embedding(num_products, latent_dim, embeddings_initializer='he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        product_vector = self.product_embedding(inputs[:, 1])

        dot_user_product = tf.reduce_sum(user_vector * product_vector, axis=1)

        return tf.nn.sigmoid(dot_user_product)

# Define hyperparameters
latent_dim = 50

# Instantiate the model
model = MatrixFactorization(num_users, num_products, latent_dim)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5, batch_size=2)

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(x_val, y_val)
#print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

product2product_original = {v: k for k, v in product2product_encoded.items()}
def recommend_products(user_id): # model, user2user_encoded, product2product_encoded, product2product_original):
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

user_id =1
recommended_products = recommend_products(user_id) # model = model, user2user_encoded = user2user_encoded, product2product_encoded = product2product_encoded, product2product_original=product2product_original)
print("Recommended Products for User 1:", recommended_products)
