import re
from typing import Callable


def fixed_size_chunking(text, chunk_size=500, overlap=50):
    """Basic fixed-size chunking with overlap."""
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        if end == text_length:
            break
        start += chunk_size - overlap
    
    return chunks


def sentence_chunker(text, max_length=500, n_sentences=5):
    """Split text into chunks by sentence count and max length."""
    if not text:
        return []
    
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        new_length = current_length + sentence_length + (1 if current_chunk else 0)
        
        if current_chunk and (new_length > max_length or len(current_chunk) >= n_sentences):
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length + (1 if len(current_chunk) > 1 else 0)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def recursive_character_text_splitter(text, max_length=500, separators=None):
    """Recursively split text using a hierarchy of separators."""
    if separators is None:
        separators = ['\n\n', '\n', '. ', ' ', '']
    
    if not text or len(text) <= max_length:
        return [text] if text else []
    
    for sep in separators:
        if sep == '':
            return fixed_size_chunking(text, chunk_size=max_length, overlap=0)
        
        parts = text.split(sep)
        if len(parts) == 1:
            continue
        
        chunks = []
        current_chunk = ""
        
        for part in parts:
            potential = f"{current_chunk}{sep}{part}" if current_chunk else part
            
            if len(potential) <= max_length:
                current_chunk = potential
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                if len(part) <= max_length:
                    current_chunk = part
                else:
                    remaining_seps = separators[separators.index(sep) + 1:]
                    chunks.extend(recursive_character_text_splitter(part, max_length, remaining_seps))
                    current_chunk = ""
        
        if current_chunk:
            chunks.append(current_chunk)
        
        if chunks:
            return chunks
    
    return fixed_size_chunking(text, chunk_size=max_length, overlap=0)


def semantic_sentence_chunker(text, max_length=500, min_length=100):
    """Chunk by sentences while respecting paragraph boundaries."""
    if not text:
        return []
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_length:
            current_chunk = f"{current_chunk}\n\n{para}".strip()
        else:
            if current_chunk and len(current_chunk) >= min_length:
                chunks.append(current_chunk)
                current_chunk = ""
            
            if len(para) > max_length:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    if len(current_chunk) + len(sent) + 1 <= max_length:
                        current_chunk = f"{current_chunk} {sent}".strip()
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sent
            else:
                current_chunk = para
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks if chunks else [text]


def sliding_window_chunker(text, chunk_size=500, overlap_ratio=0.2):
    """Sliding window that breaks at sentence boundaries."""
    if not text:
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    overlap = int(chunk_size * overlap_ratio)
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        
        # Find sentence boundary in overlap zone
        search_start = end - overlap
        search_zone = text[search_start:end]
        
        match = None
        for m in re.finditer(r'[.!?]\s+', search_zone):
            match = m
        
        if match:
            end = search_start + match.end()
        
        chunks.append(text[start:end].strip())
        start = end - overlap
    
    return chunks


def hybrid_chunker(text, target_size=400, max_size=600, min_size=100):
    """Hybrid approach balancing structure and consistent sizes."""
    if not text:
        return []
    
    if len(text) <= max_size:
        return [text]
    
    separators = [r'\n\n+', r'(?<=[.!?])\s+', r',\s+', r'\s+']
    
    def split_recursive(txt, sep_idx=0):
        if len(txt) <= max_size:
            return [txt] if len(txt) >= min_size else [txt] if txt.strip() else []
        
        if sep_idx >= len(separators):
            return [txt[i:i+target_size] for i in range(0, len(txt), target_size)]
        
        parts = re.split(separators[sep_idx], txt)
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) <= 1:
            return split_recursive(txt, sep_idx + 1)
        
        chunks = []
        current = ""
        
        for part in parts:
            if len(part) > max_size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(split_recursive(part, sep_idx + 1))
            elif len(current) + len(part) + 1 <= target_size:
                current = f"{current} {part}".strip() if current else part
            else:
                if current:
                    chunks.append(current)
                current = part
        
        if current:
            if len(current) >= min_size:
                chunks.append(current)
            elif chunks and len(chunks[-1]) + len(current) + 1 <= max_size:
                chunks[-1] = f"{chunks[-1]} {current}"
            else:
                chunks.append(current)
        
        return chunks
    
    return split_recursive(text)


def late_chunking(text, chunk_size=500, context_window=100):
    """Chunks include context from surrounding text for better embeddings."""
    if not text:
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    base_chunks = []
    current = ""
    
    for sent in sentences:
        if len(current) + len(sent) + 1 > chunk_size - context_window:
            if current:
                base_chunks.append(current)
            current = sent
        else:
            current = f"{current} {sent}".strip() if current else sent
    
    if current:
        base_chunks.append(current)
    
    # Add surrounding context
    chunks = []
    for i, chunk in enumerate(base_chunks):
        prefix = ""
        suffix = ""
        
        if i > 0:
            prev = base_chunks[i-1]
            prefix = f"[...{prev[-context_window:]}] " if len(prev) > context_window else f"[{prev}] "
        
        if i < len(base_chunks) - 1:
            next_chunk = base_chunks[i+1]
            suffix = f" [{next_chunk[:context_window]}...]" if len(next_chunk) > context_window else f" [{next_chunk}]"
        
        chunks.append(f"{prefix}{chunk}{suffix}")
    
    return chunks