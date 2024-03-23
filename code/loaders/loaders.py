import os

class TextFileLoader:
    def __init__(self, path:str, is_folder:bool=False):
        self.path = path
        self.files = []
        if is_folder:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".txt"):
                        self.files.append(os.path.join(root, file))
        else:
            self.files.append(path)
    
    def load(self):
        documents = []
        for idx, file in enumerate(self.files, start=1):
            with open(file, "r", encoding='utf-8') as f:
                documents.append(
                    {
                        "content": f.read(),
                        "metadata": {
                            "source": os.path.basename(file),
                            "page_count": 1,
                            "creation_date": None
                        }
                    }
                )
            print(f"Loaded {idx}/{len(self.files)}")
        return documents

