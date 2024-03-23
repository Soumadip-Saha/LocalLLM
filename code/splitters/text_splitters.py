import re
from typing import List

def _split_text_with_regex(text:str, separator:str, keep_separator:bool=False):
    if separator:
        if keep_separator:
            _splits = re.split(f"({separator})", text)
            splits = [_splits[i] + _splits[i+1] for i in range(1, len(_splits), 2)]
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s!=""]


class RecursiveCharacterTextSplitter:
    def __init__(
        self, 
        chunk_size:int, 
        chunk_overlap:int, 
        length_function,
        separators=None,
        keep_separator:bool=False,
        strip_whitespace:bool=True
        ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]
        self.separators = separators
        self.keep_separator = keep_separator
        self._strip_whitespace = strip_whitespace
    
    def _join_docs(self, docs:List[str], separator:str):
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        if text == "":
            return None
        else:
            return text
    
    def _merge_splits(self, splits:List[str], separator:str):
        separator_len = self.length_function(separator)

        docs = []
        current_doc = []
        total = 0
        for d in splits:
            _len = self.length_function(d)
            if total + _len + (separator_len if len(current_doc)>0 else 0) > self.chunk_size:
                if total > self.chunk_size:
                    print(f"Created a chunk of size {total} ",
                    f"Which is longer than the specified {self.chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)

                    while total > self.chunk_overlap or (
                        total + _len + (separator_len if len(current_doc)>0 else 0) > self.chunk_size and total > 0
                    ):
                        total -= self.length_function(current_doc[0]) + (
                            separator_len if len(current_doc)>1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc)>1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs
    
    def split_text(self, text:str, separators:List[str]):
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i+1:]
                break
        _separator = re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self.keep_separator)

        _good_splits = []
        _separator = "" if self.keep_separator else separator
        for s in splits:
            if self.length_function(s) < self.chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self.split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks



    def split_documents(self, documents:List[dict]):
        chunked_documents = []
        for document in documents:
            chunks = self.split_text(document["content"], self.separators)
            chunked_documents.extend(
                [
                    {
                        "content": chunk,
                        "metadata": {**document["metadata"], "chunk_idx": idx}
                    }
                    for idx, chunk in enumerate(chunks, start=1)
                ]
            )
        return chunked_documents
