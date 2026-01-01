import kenlm

def get_model(model_path: str):
    with open(model_path, 'r', encoding='utf-8') as f:
        lines =  [next(f) for _ in range(50)]
        lines = ' '.join(lines)

    if 'ngram 2' not in lines:
        return UniGramModel(model_path)

    return kenlm.Model(model_path)

class UniGramModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        word_logprob = self.get_word_logprob_dict()
        self.word_logprob = word_logprob
        self.unknown_value = word_logprob.get('<unk>', -7.726105)  # log10 prob for unknown words

    def score(self, sent: str, bos: bool = True, eos: bool = True) -> float:
        total_logprob = 0.0
        for word in sent.split():
            total_logprob += self.word_logprob.get(word, self.unknown_value)

        if bos:
            total_logprob += self.word_logprob.get('<s>', self.unknown_value)  # Assuming log10(1) = 0 for BOS
        if eos:
            total_logprob += self.word_logprob.get('</s>', self.unknown_value)  # Assuming log10(1) = 0 for EOS
        return total_logprob
    
    def get_word_logprob_dict(self) -> dict[str, float]:
        word_logprob = {}
        with open(self.model_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                logprob_str, word = parts
                word_logprob[word] = float(logprob_str)
        return word_logprob
