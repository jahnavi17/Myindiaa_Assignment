"""
This script creates a mock API using Flask Library.
I have given some sample data to test the API.
Run the script and the API opens up on http://127.0.0.1:5000/
To check different endpoints, use the route mentioned above each function below ex: http://127.0.0.1:5000/api/user/1
I have created endpoints to 
1. Retrieve user data
2. Retrieve product data
3. Test Description Generation
4. Test Recommendation Systems
"""


from flask import Flask, jsonify, request
import Recommendation_System1, Recommendation_System2
import Description_Generation

app = Flask(__name__)

# Mock user data
users = {
    1: {"name": "Alice", "browsing_history": [101, 102]},
    2: {"name": "Bob", "browsing_history": [101, 103]},
    3: {"name": "Charlie", "browsing_history": [102, 104]},
}

# Mock product data
products = {
    101: {"product_category": "Kitchen" ,"product_name": "Eco-Friendly Water Bottle", "price": 20.00 , 'features':'BPA-free, Made from recycled materials, Leak-proof, Dishwasher safe', 'rating' : 4},
    102: {"product_category": "Kitchen","product_name": "Reusable Straw", "price": 5.00 , 'features' : 'Made from stainless steel, Comes with a cleaning brush, Reusable and eco-friendly, Portable case included' , 'rating' : 3},
    103: {"product_category": "Personal care","product_name": "Bamboo Toothbrush", "price": 3.00 , 'features':'Biodegradable handle, Soft bristles, Eco-friendly packaging, Suitable for sensitive teeth' , 'rating' : 5},
    104: {"product_category": "Kitchen","product_name": "Stainless Steel Lunch Box", "price": 25.00 , 'featres' :'Durable and rust-proof, Leak-proof seal, Multiple compartments, BPA-free, Dishwasher safe' ,'rating' : 5},
}

# Endpoint to retrieve user data
@app.route('/api/user/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = users.get(user_id)
    if user is not None:
        return jsonify(user)
    else:
        return jsonify({"error": "User not found"}), 404

# Endpoint to retrieve product data
@app.route('/api/product/<int:product_id>', methods=['GET'])
def get_product(product_id):
    product = products.get(product_id)
    if product is not None:
        return jsonify(product)
    else:
        return jsonify({"error": "Product not found"}), 404

#Endpoint to retrieve Description 
@app.route('/api/productdescription1/<int:product_id>', methods=['GET'])
def get_product_description1(product_id):
    product = products.get(product_id)
    models = ["EleutherAI/gpt-neo-2.7B"] #,"gpt2"]
    prompt = Description_Generation.construct_prompt(product)
    if product is not None:
        return Description_Generation.generate_descriptions(models , prompt)
    else:
        return jsonify({"error": "Product not found"}), 404
    
#Endpoint to retrieve Description
@app.route('/api/productdescription2/<int:product_id>', methods=['GET'])
def get_product_description2(product_id):
    product = products.get(product_id)
    models = ["gpt2"]
    prompt = Description_Generation.construct_prompt(product)
    if product is not None:
        return Description_Generation.generate_descriptions(models , prompt)
    else:
        return jsonify({"error": "Product not found"}), 404
    
#Endpoint to retrieve Recommendation
@app.route('/api/productrecommendation1/<int:user_id>', methods=['GET'])
def get_product_recommendation1(user_id):
    products = Recommendation_System1.recommend_products(user_id)
    return products

@app.route('/api/productrecommendation2/<int:user_id>', methods=['GET'])
def get_product_recommendation2(user_id):
    products = Recommendation_System2.recommend_products(user_id)
    return products

if __name__ == '__main__':
    app.run(debug=True)


