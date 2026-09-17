# swahili-chatbot-flan-t5
Fine-Tuning a flan-t5 transformer model for Swahili conversational responses, with a custom evaluation pipeline comparing base vs fine-tuned performance

This project fine-tunes a pre-trained flan-t5 sequence-to-sequence model on a custom Swahili conversational dataset to build a chatbot capable of generating contextually relevant responses in Swahili- a low-resource language than most off-the-shelf language models aren't optimized for.

Approach: Rather than just fine-tuning and assuming it worked, I built a dedicated evaluation script to directly compare the base (pre-trained) model against the fine-tuned version, using BLUE score (a standard machine translation/generation metric) alongside token-level precision, recall, and F1 to measure how closely generated responses matched expected ones.

Tools Used: Python, Hugging Face Transformers, PyTorch, NLTK

Key Concepts applied: Transfer learning, tokenization, sequence-to-sequence generation, model fine-tuning and quantitative evaluation of generative model output.
