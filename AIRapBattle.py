import random
from transformers import pipeline

# Load AI Rap Model
rap_generator = pipeline("text-generation", model="gpt-4")

# Define Opponents
rapper_1 = "Tupac AI"
rapper_2 = "Jay-Z AI"

# Generate AI Rap Battle
def generate_battle():
    topic = random.choice(["money", "power", "street life", "AI takeover"])
    print(f"\n🔥 Rap Battle Theme: {topic} 🔥\n")
    
    verse_1 = rap_generator(f"{rapper_1} raps about {topic}, dissing {rapper_2}:")
    verse_2 = rap_generator(f"{rapper_2} responds to {rapper_1} with a diss:")
    
    print(f"\n🎤 {rapper_1}:\n{verse_1[0]['generated_text']}")
    print(f"\n🎤 {rapper_2}:\n{verse_2[0]['generated_text']}")

generate_battle()
