"""Print the full fine-tuning and LoRA recipes used with Gemma-3-270M."""

from pprint import pprint

from text_dialogue_agents.finetuning import full_sft_settings, lora_settings

print("FULL SFT")
pprint(full_sft_settings())
print("\nLORA")
pprint(lora_settings())
