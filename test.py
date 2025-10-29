from pymongo import MongoClient

client = MongoClient("mongodb://root:tyan123@192.168.50.217:27017")

db = client["speech2IPA"]
collection = db["phonemes"]

document = {"name": "Alice", "age": 23}
insert_result = collection.insert_one(document)
print("Inserted ID:", insert_result.inserted_id)
