# Friday April 4th
# %%import torch
from typing import cast
import torch
from circuitsvis.tokens import colored_tokens, colored_tokens_multi  # type: ignore
from datasets import Dataset, load_dataset  # type: ignore
from transformer_lens import HookedTransformer  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

# %%
DATASET_PATH = "ServiceNow-AI/R1-Distill-SFT"
DATASET_NAME = "v1"
SPLIT = "train"  # there's only a train split
CACHE_DIR = ".cache"

dataset = cast(
    Dataset, load_dataset(DATASET_PATH, DATASET_NAME, split=SPLIT, cache_dir=CACHE_DIR)
)

# %%


# BASE = "Qwen/Qwen2.5-7B-Instruct"
# BASE_MATH = "Qwen/Qwen2.5-Math-7B"
# R1_TUNED = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

BASE = "Qwen/Qwen2.5-1.5B"
BASE_MATH = "Qwen/Qwen2.5-Math-1.5B"
R1_TUNED = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


# %%
def make_model(name: str):
    hf_model = AutoModelForCausalLM.from_pretrained(name, cache_dir=CACHE_DIR)
    hf_tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=CACHE_DIR)
    tl_model = HookedTransformer.from_pretrained_no_processing(
        BASE,
        hf_model=hf_model,
        cache_dir=CACHE_DIR,
        dtype=torch.bfloat16,
        tokenizer=hf_tokenizer,
    )
    return tl_model, hf_model, hf_tokenizer


# %%

llm_math, llm_math_hf, llm_math_tokenizer = make_model(BASE_MATH)
llm_r1_tuned, llm_r1_tuned_hf, llm_r1_tuned_tokenizer = make_model(R1_TUNED)

# %%


def tokenwise_kl(probs_P_SV: torch.Tensor, probs_Q_SV: torch.Tensor):
    """S = seq, V = vocab"""
    tokens_kl_SV = torch.sum(probs_P_SV * torch.log(probs_P_SV / probs_Q_SV), dim=-1)
    return tokens_kl_SV


Range = tuple[int, int]

KlDivSection = tuple[tuple[Range, Range], torch.Tensor]

hookpoints = [f"blocks.{i}.hook_resid_pre" for i in range(llm_r1_tuned.cfg.n_layers)]

# %%
list(llm_r1_tuned.hook_dict.keys())
# %%


def get_seq_data(prompt: str) -> list[KlDivSection]:
    math_toks: torch.Tensor = llm_math_tokenizer.encode(prompt, return_tensors="pt")  # type: ignore
    assert math_toks.shape[0] == 1
    math_logits, cache = llm_math.run_with_cache(
        math_toks,
        names_filter=lambda name: name in hookpoints,
    )
    assert math_logits.shape[0] == 1
    math_seq_logits = math_logits[0]

    r1_toks: torch.Tensor = llm_r1_tuned_tokenizer.encode(prompt, return_tensors="pt")  # type: ignore
    assert r1_toks.shape[0] == 1
    r1_logits = llm_r1_tuned.forward(r1_toks, return_type="logits")
    assert r1_logits.shape[0] == 1
    r1_seq_logits = r1_logits[0]

    common_subsections = find_common_subsections(
        math_toks[0].tolist(), r1_toks[0].tolist()
    )

    kl_sections: list[KlDivSection] = []
    for (start_math, end_math), (start_r1, end_r1) in common_subsections:
        math_section_logits_SV = math_seq_logits[start_math:end_math]
        r1_section_logits_SV = r1_seq_logits[start_r1:end_r1]

        input_math_section = math_toks[0][start_math:end_math]
        input_r1_section = r1_toks[0][start_r1:end_r1]
        assert (input_math_section == input_r1_section).all()

        math_seq_probs_SV = math_section_logits_SV.softmax(dim=-1)
        r1_seq_probs_SV = r1_section_logits_SV.softmax(dim=-1)

        math_seq_preds_S = math_section_logits_SV.argmax(dim=-1)
        r1_seq_preds_S = r1_section_logits_SV.argmax(dim=-1)

        # (KL, would've predicted different, alternative prediction)
        data = torch.zeros(len(math_seq_probs_SV), 4, dtype=torch.float32)
        data[:, 0] = tokenwise_kl(
            probs_P_SV=math_seq_probs_SV, probs_Q_SV=r1_seq_probs_SV
        )
        data[:, 1] = (math_seq_preds_S != r1_seq_preds_S).float()
        data[:, 2] = math_seq_preds_S
        data[:, 3] = r1_seq_preds_S

        kl_sections.append((((start_math, end_math), (start_r1, end_r1)), data))

    return kl_sections


# %%


def fmt_msg(msg: dict[str, str]):
    if "content" not in msg:
        raise ValueError(f"No content in message: {msg}")
    if msg["role"] == "user":
        return f"User: {msg['content']}"
    elif msg["role"] == "assistant":
        return f"Assistant: {msg['content']}"
    else:
        raise ValueError(f"Unknown role: {msg['role']}")


# %%


def fmt_conversation(msgs: list[dict[str, str]]):
    return "\n\n".join([fmt_msg(msg) for msg in msgs])


# %%


def examine(seq: str):
    kl_sections = get_seq_data(seq)

    # index into one of the tokenizations
    math_ranges = [math_range for ((math_range, _), _) in kl_sections]
    valid_indices = {i for (start, end) in math_ranges for i in range(start, end)}
    math_toks: torch.Tensor = llm_math_tokenizer.encode(seq, return_tensors="pt")[0]

    math_tokens = [
        llm_math_tokenizer.decode(tok) if i in valid_indices else "?"
        for i, tok in enumerate(math_toks)
    ]
    # (KL, would've predicted different, alternative prediction)
    values = torch.zeros(len(math_tokens), 4)
    for ((math_start, math_end), _), data in kl_sections:
        values[math_start:math_end] = data
    # return colored_tokens(
    #     tokens=math_tokens,
    #     values=values.tolist(),
    #     min_value=0,
    #     max_value=10,
    # )

    return colored_tokens_multi(
        tokens=math_tokens,
        values=values,
        labels=["kl", "they differ", "math pred", "r1 pred"],
        # min_value=0,
        # max_value=10,
    )


# %%
dataset_iter = iter(dataset)
# %%
seq = fmt_conversation(next(dataset_iter)["reannotated_messages"])
len(seq)
# %%
seq
# %%
examine(seq)
# %%

llm_math_tokenizer.decode(532)
# %%

llm_r1_tuned_tokenizer.decode(92)

# %%

get_r1_token(5109)
get_math_token(2661)

# %%


def find_common_subsections(
    seq1: list[int], seq2: list[int]
) -> list[tuple[Range, Range]]:
    """
    Find common subsections between two token sequences.

    Returns:
        List of tuples, where each tuple contains:
        ((seq1_start, seq1_end), (seq2_start, seq2_end))

        The ranges are inclusive for start and exclusive for end.
    """
    # Find all common substrings
    common_substrings = []

    i = 0
    while i < len(seq1):
        j = 0
        while j < len(seq2):
            # Skip if not a match
            if seq1[i] != seq2[j]:
                j += 1
                continue

            # Found a match, find length
            length = 1
            while (
                i + length < len(seq1)
                and j + length < len(seq2)
                and seq1[i + length] == seq2[j + length]
            ):
                length += 1

            common_substrings.append((i, j, length))

            # Move forward
            j += 1
        i += 1

    # Sort by length (descending)
    common_substrings.sort(key=lambda x: -x[2])

    # Filter out overlapping substrings
    result: list[tuple[Range, Range]] = []
    used_indices1: set[int] = set()
    used_indices2: set[int] = set()

    for start1, start2, length in common_substrings:
        # Create range sets for this substring
        range1 = set(range(start1, start1 + length))
        range2 = set(range(start2, start2 + length))

        # Check if there's any overlap with previously selected substrings
        if not (range1 & used_indices1) and not (range2 & used_indices2):
            result.append(((start1, start1 + length), (start2, start2 + length)))
            used_indices1.update(range1)
            used_indices2.update(range2)

    # Sort by position in seq1
    return sorted(result, key=lambda x: x[0][0])


# example:
seq1 = [1, 2, 3, 8, 9]
seq2 = [1, 2, 3, 4, 5, 6, 7, 8, 9]

find_common_subsections(seq1, seq2)
# %%
