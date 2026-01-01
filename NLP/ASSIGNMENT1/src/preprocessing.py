import emoji
import unicodedata
import re
from typing import Iterator


def _generate_sentences(raw_text: str) -> Iterator[str]:
    """Generator that yields processed sentences one at a time."""
    # Normalize and lowercase once upfront
    raw_text = unicodedata.normalize('NFKC', raw_text).lower()
    
    # Process article by article without storing all articles
    for article in re.split(r'\n\s*\n', raw_text.strip()):
        # Collapse internal newlines
        single_line = ' '.join(article.split())
        if not single_line:
            continue
        
        # Remove links
        single_line = ' '.join(
            word for word in single_line.split()
            if not (word.startswith('http://') or word.startswith('https://'))
        )
        
        # Remove emojis
        single_line = emoji.replace_emoji(single_line, replace='')
        
        # Normalize numbers
        single_line = re.sub(r'\d+', '<num>', single_line)
        
        # Split into sentences and process each
        for sentence in re.split(r'(?<=[.!?])\s+', single_line):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Remove punctuation
            sentence = re.sub(r'[^\w\s]', '', sentence)
            
            # Remove extra spaces
            sentence = ' '.join(sentence.split())
            
            if sentence:
                yield sentence


def process_text(text: str) -> str:
    """Memory-optimized text processing. Drop-in replacement."""
    return '\n'.join(_generate_sentences(text))