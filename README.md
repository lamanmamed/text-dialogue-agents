# Text Dialogue Agents

This repository contains five text-based conversational AI projects: identifying what a speaker is doing in a conversation, generating replies with an encoder-decoder model, adding Wikipedia retrieval before answering factual questions, running restaurant bookings and orders through structured actions, and fine-tuning a small chat model.

The projects use different kinds of dialogue systems rather than one shared model. The code is organized by the job each system performs.

## Recognizing dialogue acts

The first system predicts the function of an utterance in a conversation. Examples of dialogue acts include statements, questions, acknowledgements, and backchannels.

I used the Switchboard Dialogue Act corpus and reduced its original labels to 43 classes. I compared three models:

1. a two-layer BiLSTM that classifies each utterance on its own
2. the same BiLSTM trained with class-weighted cross-entropy
3. a context model that encodes each utterance with a CNN and then runs a BiLSTM across a seven-utterance window

One saved run produced:

| Model | Test accuracy | Test loss |
| --- | ---: | ---: |
| Utterance BiLSTM | **49.80%** | 3.3020 |
| Class-weighted BiLSTM | 41.87% | — |
| CNN + context BiLSTM | 48.34% | **1.7036** |

The main problem with the unweighted model was not visible from overall accuracy alone. For the rare `br` and `bf` dialogue acts, it made **zero correct predictions** and never predicted either class.

After adding class weights, the saved notebook run raised accuracy on `br` from 0% to **49.33%** and on `bf` from 0% to **11.57%**. Overall accuracy fell because the model stopped relying as heavily on the most common labels.

The context model also fixed individual ambiguous cases. One saved example contains the target utterance `Yeah.`. The utterance-only model predicted `b`, while the context model used the neighbouring turns to predict the correct `b^m` label.

A second recorded run produced slightly different aggregate numbers. I use the values above because all three models were evaluated together in the same run.

## Generating replies with an encoder-decoder model

I implemented a dialogue response model with a bidirectional GRU encoder, a GRU decoder, and Bahdanau attention. The decoder generates a reply one token at a time.

I also implemented beam search instead of only taking the most likely token at each step. Beam search keeps several possible replies alive and compares their accumulated probabilities before choosing a final sequence.

The model was trained on two datasets:

- Cornell Movie-Dialogs for open-ended movie conversations
- HybridDialogue for more factual exchanges

I tested several changes to the model rather than assuming that a larger or more regularized network would automatically help. Increasing dropout made the replies more generic. A deeper encoder gave the clearest qualitative improvement in the recorded experiments. Replacing Bahdanau attention with Luong attention did not produce a clear improvement, and changing the teacher-forcing schedule did not remove the tendency to generate short, safe answers.

The examples are useful mainly as an examination of a small sequence-to-sequence dialogue model. Even after training, many open-ended answers remain generic, and the HybridDialogue model sometimes repeats words when trying to answer factual questions.

## Adding Wikipedia retrieval before answering

The RAG experiment starts with **Qwen3-1.7B** answering multi-turn HybridDialogue questions directly. I then added a retrieval step over Wikipedia passages encoded with Contriever.

For every user question, the system:

1. builds a search query from the recent conversation
2. retrieves two Wikipedia passages
3. adds those passages to the model prompt
4. asks Qwen to answer using the retrieved information

I then changed the retrieval query itself. The second version puts the current question first, extracts names and places from recent turns, and resolves simple references such as `he`, `she`, and `these two` before searching.

One saved run gives:

| System | Token F1 | BLEU |
| --- | ---: | ---: |
| Qwen without retrieval | 23.29 | 2.01 |
| RAG with recent dialogue as query | 28.59 | 4.39 |
| RAG with rewritten query + extracted entities | **30.69** | **5.01** |

The scores are based on 243 test responses. Retrieval improved both metrics, but it did not solve every error. In some examples the correct passage was retrieved and the language model still produced the wrong answer. In others, the query did not identify the correct Wikipedia passage at all.

Token F1 and BLEU also penalize answers that are factually correct but phrased differently from the reference. The repository keeps those metrics because they were used consistently across the experiments, but the README does not treat them as a complete measure of answer quality.

## Restaurant agent with structured actions

The restaurant agent separates conversation from operations that need exact data.

The language model handles the dialogue, but restaurant search, booking, menu lookup, order creation, order confirmation, order lookup, and cancellation are implemented as deterministic Python functions. The model requests one of those actions in JSON instead of inventing a booking reference or order state itself.

For example, the saved conversation goes through this flow:

```text
find cheap Chinese restaurant in the centre
→ Centre Noodle House
→ collect day, time, party size, and name
→ create booking
→ show menu
→ create 3 × Spring Rolls = $15.00
→ confirm order
→ look up the order by ORD-xxxxxx
→ cancel it
→ later lookup returns status = cancelled
```

The cleaned implementation keeps the backend independent of the language model. `RestaurantBackend` can therefore be tested without running Ollama. `OllamaRestaurantAgent` adds the model-driven JSON action loop on top when a local Ollama server is available.

## Full fine-tuning vs LoRA on Gemma

I used **Gemma 3 270M** with the `everyday-conversations` split of SmolTalk and compared the base model with two adaptation methods.

Full supervised fine-tuning updates the model weights directly. The LoRA version freezes the base model and trains low-rank adapters with rank 8, alpha 16, and 0.05 dropout.

One saved run records:

| Model | Token F1 | BLEU |
| --- | ---: | ---: |
| Gemma 3 270M, zero-shot | 23.60 | 4.20 |
| Full supervised fine-tuning | **30.60** | **11.40** |
| LoRA | 23.62 | 4.33 |

Both fine-tuning runs used two epochs and an effective batch size of 8. The full model used a learning rate of `2e-5`, while LoRA used `1e-4`.

In this run, full fine-tuning clearly improved the overlap-based evaluation scores. LoRA trained successfully but finished close to the zero-shot model. This is a useful result because parameter-efficient fine-tuning did not automatically give the best task performance for this small model and dataset.

A second full-fine-tuning run reached F1 32.39 and BLEU 12.16. I use the values in the table because the zero-shot, full fine-tuning, and LoRA models were all evaluated together in the same run.

## Repository structure

```text
text-dialogue-agents/
├── src/text_dialogue_agents/
│   ├── dialogue_acts.py
│   ├── seq2seq.py
│   ├── rag.py
│   ├── restaurant_agent.py
│   └── finetuning.py
├── experiments/
│   ├── dialogue_act_shapes.py
│   ├── seq2seq_shapes.py
│   ├── rag_query_example.py
│   ├── restaurant_demo.py
│   └── finetuning_configs.py
├── tests/
├── data/
│   └── README.md
├── pyproject.toml
└── README.md
```

## Running the code

Install the core package:

```bash
pip install -e .
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

Install the libraries needed for the pretrained-model experiments:

```bash
pip install -e ".[experiments]"
```

The lightweight experiment scripts can be run independently. For example:

```bash
python experiments/dialogue_act_shapes.py
python experiments/seq2seq_shapes.py
python experiments/rag_query_example.py
python experiments/restaurant_demo.py
python experiments/finetuning_configs.py
```

The RAG and fine-tuning experiments require external model weights and datasets. The restaurant language-model loop requires a running Ollama server, but the backend itself has no model dependency.

## Data

The training datasets are not copied into this repository. `data/README.md` lists the data used by each part of the project.