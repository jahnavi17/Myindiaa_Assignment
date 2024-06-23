"""
This script uses Transformers library and it's Pipeline frature to generate Product Descriptions.
For sample, I have used two Large Language Models. The performance of these models is not good 
But I have exhausted my free tier benefits of OpenAI API. So I have used these models only.

Once this script is executed, it asks for specific product details and then the product description
is generated using both the sample models used.
"""

from transformers import pipeline

# Construct the prompt from the product attributes
def construct_prompt(attributes):
    return f"Generate a product description for the following attributes:\n" \
           f"Product Category: {attributes['product_category']}\n" \
           f"Product Name: {attributes['product_name']}\n" \
           f"Price: {attributes['price']}\n" \
           f"features: {attributes['features']}\n" \
           f"rating: {attributes['rating']}\n" \
           f"The description of the product with above details is : "

# Initializing sample models
models = ["EleutherAI/gpt-neo-2.7B","gpt2"]

# Generate descriptions using different models
def generate_descriptions(models, prompt):
    descriptions = {}
    for model_name in models:
        generator = pipeline('text-generation', model=model_name)
        output = generator(prompt, max_length=150, num_return_sequences=1, temperature=0.7)
        descriptions[model_name] = output[0]['generated_text']
    return descriptions

if __name__ == "__main__" :
     attributes = {}
     attributes['product_category'] = input("Enter the Product Category : ")
     attributes['product_name'] = input("Enter the Product Name :")
     attributes['price'] = input("Enter the price :")
     attributes['features'] = input("Enter the special features of the Product :")
     attributes['rating'] = input("Enter the ratings of the Product :")

     #print(attributes)
     prompt = construct_prompt(attributes)
     #print(prompt)
     description = generate_descriptions(models, prompt)
     print(description)

