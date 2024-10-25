import numpy as np
from search_fusion import fusion
from search_fusion import spice

embedding_model = "mxbai-embed-large"
search_querry = 'hiking'
prompt_count = 5
results_per_prompt = 5

titles = [
    "The Ultimate Guide to Healthy Eating",
    "10 Tips for Mastering Time Management",
    "Exploring the Wonders of the Amazon Rainforest",
    "Unveiling the Secrets of Ancient Egypt",
    "The Rise of Artificial Intelligence in Healthcare",
    "10 Must-Visit Destinations for Adventure Seekers",
    "The Power of Meditation: Finding Inner Peace",
    "Unlocking the Potential of Renewable Energy",
    "The Art of Effective Communication",
    "The Future of Space Travel: Colonizing Mars",
    "The Impact of Social Media on Mental Health",
    "10 Delicious and Nutritious Smoothie Recipes",
    "Understanding Cryptocurrency: A Beginner's Guide",
    "The Psychology of Happiness: How to Find True Joy",
    "The Evolution of Fashion: From Victorian Era to Modern Times",
    "Exploring the Mysteries of the Bermuda Triangle",
    "The Benefits of Yoga for Mind, Body, and Soul",
    "10 Steps to a Successful Job Interview",
    "The Fascinating World of Wildlife Photography",
    "The Healing Power of Music Therapy",
    "Discovering the Marvels of the Great Barrier Reef",
    "The Secrets of Longevity: Healthy Habits for a Longer Life",
    "10 DIY Home Improvement Projects to Transform Your Space",
    "The Rise of Veganism: A Sustainable Lifestyle Choice",
    "Unveiling the Truth: Debunking Common Myths",
    "The Future of Artificial Intelligence in Education",
    "The Science of Sleep: Unlocking the Secrets of a Good Night's Rest",
    "10 Mind-Blowing Facts About the Universe",
    "The Benefits of Mindfulness in Everyday Life",
    "The History and Evolution of Video Games",
    "Exploring the Marvels of Underwater Life: The Great Barrier Reef",
    "The Power of Positive Thinking: Transforming Your Life",
    "10 Essential Skills for Success in the Digital Age",
    "The Impact of Climate Change on Wildlife",
    "The Secrets of Successful Entrepreneurship",
    "The Magic of Books: Escaping into Other Worlds",
    "The Benefits of Regular Exercise: A Path to Wellness",
    "10 Easy Ways to Reduce Stress and Find Inner Peace",
    "The Future of Transportation: Electric and Autonomous Vehicles",
    "Unveiling Ancient Civilizations: The Aztecs and the Incas",
    "The Science of Happiness: How to Cultivate a Positive Mindset",
    "The Art of Public Speaking: Mastering the Stage",
    "10 Delicious Plant-Based Recipes for a Healthy Lifestyle",
    "The Impact of Social Media on Society",
    "The Wonders of the World: Exploring Spectacular Landmarks",
    "The Importance of Emotional Intelligence in Leadership",
    "The Benefits of Organic Farming: A Sustainable Approach",
    "Exploring the Enchanting Beauty of Paris: The City of Love",
    "The Power of Gratitude: Transforming Your Outlook on Life",
    "10 Tips for Successful Weight Loss and Maintenance",
    "The Evolution of Technology: From the First Computer to Artificial Intelligence",
    "The Secrets of Successful Negotiation",
    "The Beauty of Nature: Capturing Stunning Landscape Photography",
    "The Impact of Social Media on Relationships",
    "The Healing Power of Nature: The Benefits of Ecotherapy",
    "The Future of Work: Embracing Remote and Flexible Jobs",
    "Unlocking Creativity: Discovering Your Inner Artist",
    "10 Inspiring Quotes to Motivate and Empower",
    "The Science of Nutrition: Understanding the Building Blocks of a Healthy Diet",
    "The History and Influence of Classical Music",
    "Exploring the Marvels of Ancient Rome: The Eternal City",
    "The Art of Effective Leadership: Inspiring Others to Succeed",
    "10 DIY Crafts for Creative Minds",
    "The Rise of Sustainable Fashion: Ethical and Eco-Friendly Clothing",
    "Unveiling the Mysteries of the Human Brain",
    "The Benefits of Volunteering: Making a Difference in the World",
    "The Future of Healthcare: Innovations and Breakthroughs",
    "The Power of Positive Affirmations: Manifesting Your Dreams",
    "10 Tips for a Healthy and Happy Relationship",
    "The Evolution of Cinema: From Silent Films to Virtual Reality",
    "The Secrets of Effective Time Management",
    "The Fascinating World of Marine Life: Exploring the Ocean Depths",
    "The Impact of Technology on Education",
    "The Benefits of Mindful Eating: Nourishing Your Body and Soul",
    "The Art of Problem-Solving: Strategies for Success",
    "10 Essential Travel Tips for Exploring New Destinations",
    "Unveiling the Wonders of the Universe: Exploring Outer Space",
    "The Science of Happiness: Cultivating Joy and Contentment",
    "The Importance of Mental Health Awareness",
    "The Future of Energy: Sustainable Solutions for a Greener World",
    "Discovering the Rich History of Ancient Greece",
    "The Secrets of Effective Communication: Building Strong Connections",
    "10 Easy Ways to Boost Your Productivity",
    "The Impact of Social Media on Mental Wellbeing",
    "The Healing Power of Art Therapy: Expressing Yourself Creatively",
    "The Future of Transportation: Flying Cars and Hyperloop",
    "The Magic of Photography: Capturing Moments in Time",
    "The Benefits of Mindful Parenting: Nurturing Happy and Resilient Children",
    "10 Tips for a Healthy Work-Life Balance",
    "The Evolution of Fashion: From Runway to Street Style",
    "Exploring the Marvels of Ancient Egypt: The Land of Pharaohs",
    "The Art of Effective Decision-Making: Choosing Wisely",
    "10 Delicious and Nutritious Recipes for a Plant-Based Diet",
    "The Impact of Social Media on Self-Esteem",
    "The Wonders of the Natural World: Discovering Breathtaking Landscapes",
    "The Power of Positive Parenting: Nurturing Confident and Resilient Children",
    "The Benefits of Sustainable Living: Reducing Your Carbon Footprint",
    "Unveiling the Secrets of the Human Mind: Exploring Psychology",
    "The Future of Travel: Virtual Reality and Augmented Reality Experiences",
    "The Science of Happiness: Cultivating a Fulfilling and Meaningful Life",
]
print("---------------------------------------------------------------------------------------------------------------------------------------------")
print("-> load vectors ...")

embeddings = fusion.embedd_list(titles, model=embedding_model)

print("-> loaded article vectors")
print("-> generate prompts ...")


prompts = spice.spice_up(model="llama3", prompt=search_querry, count=prompt_count)
prompts.append(search_querry)

search_res = fusion.search_for_querries(texts=titles, querries=prompts, m=results_per_prompt, text_embeddings=embeddings, model=embedding_model)
default_arr = search_res[-1]

results_list = fusion.frequency_sort(search_results=search_res)

print("\nresult count:", len(results_list))
print(results_list)

similarity_scores, similiarity_indices = fusion.similarity_matrix(results_list, model=embedding_model)

similiarity_list = fusion.similarity_sort(results_list, (similarity_scores, similiarity_indices))

print("\n------------prompts----------------\n")

for prompt in prompts:
    print(prompt)

print("\n------------similarity sorted----------------\n")

for i, text in enumerate(similiarity_list[:results_per_prompt]):

    print(f"{i}\t{text}")


print("\n------------frequency sorted----------------\n")

for i, text in enumerate(results_list[:results_per_prompt]):

    print(f"{i}\t{text}")


print("\n-----------no fusion------------------------\n")

for i, text in enumerate(default_arr[:results_per_prompt]):

    print(f"{i}\t{text}")
