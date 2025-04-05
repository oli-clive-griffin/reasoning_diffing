# %%import torch
from dataclasses import dataclass
import numpy as np
import plotly.express as px
from typing import cast
from einops import rearrange, reduce
import torch
from circuitsvis.tokens import colored_tokens, colored_tokens_multi  # type: ignore
from datasets import Dataset, load_dataset  # type: ignore
from transformer_lens import HookedTransformer  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer  # type: ignore
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
def make_model(
    name: str,
) -> tuple[HookedTransformer, AutoModelForCausalLM, PreTrainedTokenizer]:
    hf_model = AutoModelForCausalLM.from_pretrained(name, cache_dir=CACHE_DIR)
    hf_tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=CACHE_DIR)
    tl_model = HookedTransformer.from_pretrained_no_processing(
        BASE,
        hf_model=hf_model,
        cache_dir=CACHE_DIR,
        dtype=torch.bfloat16,
        tokenizer=hf_tokenizer,
    )
    return tl_model, hf_model, hf_tokenizer  # type: ignore


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

llm_math, llm_math_hf, llm_math_tokenizer = make_model(BASE_MATH)
llm_r1_tuned, llm_r1_tuned_hf, llm_r1_tuned_tokenizer = make_model(R1_TUNED)


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


def tokenwise_kl(probs_P_SV: torch.Tensor, probs_Q_SV: torch.Tensor):
    """S = seq, V = vocab"""
    tokens_kl_S = torch.sum(probs_P_SV * torch.log(probs_P_SV / probs_Q_SV), dim=-1)
    return tokens_kl_S


# %%


Range = tuple[int, int]

KlDivSection = tuple[tuple[Range, Range], torch.Tensor, torch.Tensor]

hookpoints = [f"blocks.{i}.hook_resid_pre" for i in range(llm_r1_tuned.cfg.n_layers)]
# hookpoints = [item for sublist in hookpoints for item in sublist]


def get_logits_and_resid(
    prompt: str, tokenizer: PreTrainedTokenizer, model: HookedTransformer
):
    toks: torch.Tensor = tokenizer.encode(prompt, return_tensors="pt")  # type: ignore
    assert toks.shape[0] == 1
    seq_logits, cache = model.run_with_cache(
        toks,
        names_filter=lambda name: name in hookpoints,
    )
    toks_S = toks[0]
    seq_logits_SV = seq_logits[0]
    resid_SLD = cache.remove_batch_dim().stack_activation("resid_pre").transpose(0, 1)
    return seq_logits_SV, resid_SLD, toks_S


# %%

@dataclass
class SeqData:
    input_tokens_S: torch.Tensor
    math_pred_toks_S: torch.Tensor
    r1_pred_toks_S: torch.Tensor
    kl_div_S: torch.Tensor
    mse_SL: torch.Tensor

def get_seq_data(prompt: str) -> list[SeqData]:
    math_logits_SV, math_resid_SLD, math_toks_S = get_logits_and_resid(
        prompt, llm_math_tokenizer, llm_math
    )
    r1_logits_SV, r1_resid_SLD, r1_toks_S = get_logits_and_resid(
        prompt, llm_r1_tuned_tokenizer, llm_r1_tuned
    )

    common_subsections = find_common_subsections(
        math_toks_S.tolist(), r1_toks_S.tolist()
    )

    sections: list[SeqData] = []
    for (start_math, end_math), (start_r1, end_r1) in common_subsections:
        math_section_logits_SV = math_logits_SV[start_math:end_math]
        r1_section_logits_SV = r1_logits_SV[start_r1:end_r1]

        input_math_section = math_toks_S[start_math:end_math]
        input_r1_section = r1_toks_S[start_r1:end_r1]
        assert (input_math_section == input_r1_section).all()
        input_tokens_S = input_math_section

        math_seq_probs_SV = math_section_logits_SV.softmax(dim=-1)
        r1_seq_probs_SV = r1_section_logits_SV.softmax(dim=-1)

        math_seq_preds_S = math_section_logits_SV.argmax(dim=-1)
        r1_seq_preds_S = r1_section_logits_SV.argmax(dim=-1)

        kl_div_S = tokenwise_kl(probs_P_SV=math_seq_probs_SV, probs_Q_SV=r1_seq_probs_SV)

        math_resid_section_SLD = math_resid_SLD[start_math:end_math]
        r1_resid_section_SLD = r1_resid_SLD[start_r1:end_r1]
        sq_err_SLD = (math_resid_section_SLD - r1_resid_section_SLD) ** 2
        mse_SL = reduce(sq_err_SLD, "S L D -> S L", "mean")

        sections.append(
            SeqData(
                input_tokens_S=input_tokens_S,
                math_pred_toks_S=math_seq_preds_S,
                r1_pred_toks_S=r1_seq_preds_S,
                kl_div_S=kl_div_S,
                mse_SL=mse_SL,
            )
        )

    return sections


# %%
dataset_iter = iter(dataset)
# %%
seq = fmt_conversation(next(dataset_iter)["reannotated_messages"])
len(seq)
# %%
seq
# %%

sections = get_seq_data(seq)
# %%


# mses_SL_cpu = mses_SL.T.to("cpu").float().numpy()
# tokenwise_kl_S_cpu = tokenwise_kl_S.detach().to("cpu").float().numpy()


# %%
sec = sections[0]
print(f"kl_div_S.shape: {sec.kl_div_S.shape}")
print(f"mse_SL.shape: {sec.mse_SL.shape}")
print(f"input_tokens_S.shape: {sec.input_tokens_S.shape}")
print(f"math_pred_toks_S.shape: {sec.math_pred_toks_S.shape}")
print(f"r1_pred_toks_S.shape: {sec.r1_pred_toks_S.shape}")
# %%

kl_div_S = torch.cat([section.kl_div_S for section in sections]).detach().float().cpu().numpy()
mse_SL = torch.cat([section.mse_SL for section in sections]).detach().float().cpu().numpy()
input_seq_toks: list[str] = [llm_math_tokenizer.decode(tok) for tok in torch.cat([section.input_tokens_S for section in sections]).detach().cpu().numpy()]
math_pred_toks: list[str] = [llm_math_tokenizer.decode(tok) for tok in torch.cat([section.math_pred_toks_S for section in sections]).detach().cpu().numpy()]
r1_pred_toks: list[str] = [llm_r1_tuned_tokenizer.decode(tok) for tok in torch.cat([section.r1_pred_toks_S for section in sections]).detach().cpu().numpy()]

# %%
# kl_div_S.shape, mse_SL.shape
# input_seq_toks[0], math_pred_toks[0], r1_pred_toks[0]
# %%

from utils import visualise_text_sequence
# %%

visualise_text_sequence(
    kl_div_S=kl_div_S[1:],
    mse_SL=mse_SL[1:],
    input_sequence=input_seq_toks[1:],
    math_pred_toks=math_pred_toks[1:],
    r1_pred_toks=r1_pred_toks[1:],
).show(renderer="browser")

# %%

# demo
visualise_text_sequence(
    kl_div_S=np.array([0.1, 0.2]),
    mse_SL=np.array([[0.1, 100], [0.3, 0.4]]),
    input_sequence=["a", "b"],
    math_pred_toks=["a", "b"],
    r1_pred_toks=["a", "b"],
)

# %%

llm_r1_tuned_tokenizer.decode(92)

# %%# %%

