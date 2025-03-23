import os

dataset_path = r"C:\Users\<yourname>\Documents\Facial_Recognition_project\dataset"

if os.path.exists(dataset_path):
    print("✅ Folder found! Listing files:")
    print(os.listdir(dataset_path))
else:
    print("❌ Error: 'dataset' folder not found!")
