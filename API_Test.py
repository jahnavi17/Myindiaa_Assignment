"""
This script has unit tests to test the endpoints of our Flask API.
Once this script is run, it tests all the cases and gives the final result of the tests
"""

import unittest
import json
from flask_testing import TestCase
from API_integration import app, users, products
import Recommendation_System1 , Recommendation_System2

class MockAPITestCase(TestCase):

    def create_app(self):
        app.config['TESTING'] = True
        return app

    def test_get_user(self):
        user_id = 1
        response = self.client.get(f'/api/user/{user_id}')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, users[user_id])

    def test_get_user_not_found(self):
        user_id = 999
        response = self.client.get(f'/api/user/{user_id}')
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json, {"error": "User not found"})

    def test_get_product(self):
        product_id = 101
        response = self.client.get(f'/api/product/{product_id}')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, products[product_id])

    def test_get_product_not_found(self):
        product_id = 999
        response = self.client.get(f'/api/product/{product_id}')
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json, {"error": "Product not found"})

    def test_get_recommendations1(self) :
        user_id = 2
        response = self.client.get(f'/api/productrecommendation1/{user_id}')
        self.assertEqual(response.status_code , 200)
        self.assertEqual(response.json , Recommendation_System1.recommend_products(user_id))

    def test_get_recommendations2(self) :
        user_id = 2
        response = self.client.get(f'/api/productrecommendation2/{user_id}')
        self.assertEqual(response.status_code , 200)
        self.assertEqual(response.json , Recommendation_System2.recommend_products(user_id))


if __name__ == '__main__':
    unittest.main()
