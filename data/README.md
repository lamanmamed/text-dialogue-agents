# Data

The datasets used for the recorded model runs are not copied into this repository.

The dialogue-act models use the **Switchboard Dialogue Act (SWDA)** corpus after reducing the original labels to 43 dialogue-act classes.

The sequence-to-sequence response model was trained on the **Cornell Movie-Dialogs Corpus** and **HybridDialogue**.

The retrieval experiments use **HybridDialogue** for conversations and the introductory paragraphs from a **2020 English Wikipedia dump** as the external knowledge collection. Dense retrieval was performed with Contriever embeddings.

The Gemma fine-tuning experiments use the `everyday-conversations` configuration of **HuggingFaceTB/SmolTalk**.

The restaurant backend uses a small structured restaurant and menu database. The public example script contains a minimal sample so the API flow can be run without the original data files.
