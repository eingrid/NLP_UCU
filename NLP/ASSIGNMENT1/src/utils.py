def sentence_logprob(model, sent: str) -> float:
    # log10 probability
    return model.score(sent, bos=True, eos=True)

def perplexity(model, sent: str) -> float:
    logprob = sentence_logprob(model, sent)
    n = len(sent.split())
    return 10 ** (-logprob / n)

def select_best_sentence(model, sentences: list[str], use_perplexity: bool = True) -> str:
    best_sent = None
    best_score = float('inf') if use_perplexity else float('-inf')
    for sent in sentences:
        if use_perplexity:
            score = perplexity(model, sent)  # lower is better
            if score < best_score:
                best_score = score
                best_sent = sent
        else:
            score = sentence_logprob(model, sent)  # higher is better
            if score > best_score:
                best_score = score
                best_sent = sent
    return best_sent

def process_dataset(model, dataset: list[dict], use_perplexity: bool = True) -> list[str]:
    results = []
    for item in dataset:
        sentences = item.get("hypotheses", [])
        best_sent = select_best_sentence(model, sentences, use_perplexity)
        results.append(best_sent)
    return results

def evaluate_accuracy(predictions: list[str], dataset: list[dict]) -> float:
    correct = sum(p == item.get("reference", "") for p, item in zip(predictions, dataset))
    return correct / len(dataset) if dataset else 0.0


def get_incorrect_samples(model, dataset: list[dict]) -> list[tuple[str, tuple[str,float]]]:
    """Return list of (reference, predicted) for incorrect predictions.
    Where predicted is tuple of (sentence, prob) for each sentence in hypotheses.
    """
    incorrect_samples = []
    for item in dataset:
        reference = item.get("reference", "")
        hypotheses = item.get("hypotheses", [])
        best_sent = select_best_sentence(model, hypotheses)
        if best_sent != reference:
            sent_probs = [(sent, perplexity(model, sent)) for sent in hypotheses]
            sorted_sent_probs = sorted(sent_probs, key=lambda x: x[1], reverse=False)
            incorrect_samples.append((reference, sorted_sent_probs))
    return incorrect_samples
  