# Knowledge Distillation for Ukrainian Translation

In this task, you will implement **Knowledge Distillation (KD)** to compress a large instruction-tuned "teacher" model into a smaller "student" model. Specifically, you will distill the knowledge from **MamayLM-Gemma-3-4B-IT** (4B parameters) into **Gemma-3-270M-IT** (270M parameters) for the task of English-to-Ukrainian translation.

### Task Overview

1.  **Dataset Preparation**

      * Use the [lapa-llm/fiftyfive-best](https://www.google.com/search?q=https://huggingface.co/datasets/lapa-llm/fiftyfive-best) dataset.
      * Implement **completion-only tokenization**: The model should only learn to generate the translation (target), not the instruction (source). You must set the labels for instruction tokens to `-100` so they are ignored by the loss function.
      * *Hint:* Ensure padding tokens are handled correctly, as Gemma-3 models may not have a default padding token set.

2.  **Model Setup**

      * Load the Teacher model: `INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0`.
      * Load the Student model: `google/gemma-3-270m-it`.
      * **Important:** Ensure both models utilize `bfloat16` precision and align their vocabulary sizes (you may need to resize the student's token embeddings).

3.  **Implement Distillation Trainer**

      * Create a custom `DistillationTrainer` class that inherits from Hugging Face's `Trainer`.
      * Override the `compute_loss` method to calculate a combined loss:
          * **Cross-Entropy Loss:** Standard loss between student predictions and ground truth labels.
          * **Distillation Loss (KL Divergence):** The divergence between the student's *softened* logits and the teacher's *softened* logits.
      * The final loss should be a weighted sum: $L = \alpha_{CE} \cdot L_{CE} + \alpha_{KL} \cdot L_{KL}$.

4.  **Training & Evaluation**

      * Train the student model using your custom trainer.
      * Evaluate three scenarios on a held-out test set using **BLEU** and **chrF** metrics:
        1.  Teacher Model (Upper bound performance).
        2.  Baseline Student Model (Untrained/Zero-shot).
        3.  Distilled Student Model (Your result).

### Technical Details & Hints

  * **Custom Trainer:** This is the core of the assignment. You cannot use the standard `SFTTrainer` easily here because it hides the loss computation logic. You must subclass `transformers.Trainer`.
      * Reference: [Hugging Face Trainer Documentation](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer.compute_loss).
      * Reference: [PyTorch KLDivLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html) (watch out for the inputs).
  * **Temperature:** Remember to apply temperature scaling ($T$) to logits before softmax when computing KL Divergence. A common choice is $T=2.0$.
  * **Reference Code:** You can look at the [DistilBERT implementation](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation) for inspiration on how the loss is constructed.

### Reference Material

A reference Jupyter Notebook containing most of the implementation flow is available **[here](reference.ipynb)**. While the notebook contains most of the required code, you are strongly encouraged to implement it from scratch, using the notebook as a cheatsheet. The notebook contains many more tips and explanations for the important details that are easy to overlook.