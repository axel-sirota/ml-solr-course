import pandas as pd

listings = pd.read_csv("dataset/listings.csv", header=0)
new_listings = listings[["id", "listing_url", "name", "description", "neighborhood_overview", "host_id", "host_url", "host_name",
     "neighbourhood", "neighbourhood_cleansed", "property_type", "room_type", "accommodates", "bathrooms",
     "bathrooms_text", "bedrooms", "beds", "amenities", "price", "number_of_reviews", "reviews_per_month"]]
new_listings["id"] = pd.to_numeric(new_listings["id"])
new_listings["host_id"] = pd.to_numeric(new_listings["host_id"])
new_listings["accommodates"] = pd.to_numeric(new_listings["accommodates"])
new_listings["bathrooms"] = pd.to_numeric(new_listings["bathrooms"])
new_listings["bedrooms"] = pd.to_numeric(new_listings["bedrooms"])
new_listings["beds"] = pd.to_numeric(new_listings["beds"])
new_listings["number_of_reviews"] = pd.to_numeric(new_listings["number_of_reviews"])
new_listings["reviews_per_month"] = pd.to_numeric(new_listings["reviews_per_month"])
new_listings.to_csv("dataset/listings_small.csv", index=False)

