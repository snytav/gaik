import os
directory_path = '.'
for root, _, files in os.walk(directory_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            print(file_path)
