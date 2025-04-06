# %%
from typing import cast

import torch
from circuitsvis.tokens import colored_tokens_multi  # type: ignore
from transformer_lens import HookedTransformer  # type: ignore
from transformers import (  # type: ignore
    AutoModelForCausalLM,  # type: ignore
    AutoTokenizer,  # type: ignore
    PreTrainedTokenizer,  # type: ignore
)

from utils import (
    DATASET,
    SeqData,
    get_seq_data,
    visualise_text_sequence_vertical,
)

CACHE_DIR = ".cache"
BASE = "Qwen/Qwen2.5-7B"
BASE_MATH = "Qwen/Qwen2.5-Math-7B"
R1_TUNED = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


# %%
def make_model(
    name: str,
) -> tuple[HookedTransformer, AutoModelForCausalLM]:
    hf_model = AutoModelForCausalLM.from_pretrained(name, cache_dir=CACHE_DIR)
    hf_tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=CACHE_DIR)
    tl_model = HookedTransformer.from_pretrained_no_processing(
        BASE,
        hf_model=hf_model,
        cache_dir=CACHE_DIR,
        dtype=torch.bfloat16,
        tokenizer=hf_tokenizer,
    )
    return tl_model, hf_model  # type: ignore


# %%

llm_math, llm_math_hf = make_model(BASE_MATH)
llm_r1_tuned, llm_r1_tuned_hf = make_model(R1_TUNED)

# %%

llm_r1_tuned.cfg.rotary_base = llm_r1_tuned_hf.config.rope_theta  # type: ignore

for block in llm_r1_tuned.blocks:
    attn = block.attn  # type: ignore
    sin, cos = attn.calculate_sin_cos_rotary(  # type: ignore
        llm_r1_tuned.cfg.rotary_dim,
        llm_r1_tuned.cfg.n_ctx,
        base=llm_r1_tuned.cfg.rotary_base,
        dtype=llm_r1_tuned.cfg.dtype,
    )
    attn.register_buffer("rotary_sin", sin.to(llm_r1_tuned.cfg.device))  # type: ignore
    attn.register_buffer("rotary_cos", cos.to(llm_r1_tuned.cfg.device))  # type: ignore

# %%


def fmt_math(q: str) -> str:
    return f"A conversation between a user and an assistant. The user asks a question and the assistant answers.\n\nUser: {q}\n\nAssistant:"


def fmt_r1(q: str) -> str:
    return cast(
        str,
        llm_r1_tuned.tokenizer.apply_chat_template(  # type: ignore
            [{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False
        ),
    )


# %%
q, c = DATASET[6].values()
# %%
print(f"Question: {q}")
print(f"Correct answer: {c}")
# %%
math_answer: str = cast(
    str, llm_math.generate(fmt_math(q), max_new_tokens=600, do_sample=False)
)  # type: ignore
# %%
r1_answer: str = cast(
    str, llm_r1_tuned.generate(fmt_r1(q), max_new_tokens=2000, do_sample=False)
)  # type: ignore

# %%

print(f"Math answer:\n{math_answer}")
# %%
print(f"R1 answer:\n{r1_answer}")

# %%
print(c)

# %%

torch.cuda.empty_cache()
# %%
len(r1_answer)


# %%


def process_sections(
    sections: list[SeqData],
    llm_math_tokenizer: PreTrainedTokenizer,
    llm_r1_tuned_tokenizer: PreTrainedTokenizer,
):
    kl_div_S = ( torch.cat([section.kl_div_S for section in sections]) .detach() .float() .cpu() .numpy())
    mse_SL = ( torch.cat([section.mse_SL for section in sections]) .detach() .float() .cpu() .numpy())
    acts_math_SLD = ( torch.cat([section.acts_math_SLD for section in sections]) .detach() .float() .cpu() .numpy())
    acts_r1_SLD = ( torch.cat([section.acts_r1_SLD for section in sections]) .detach() .float() .cpu() .numpy())
    input_seq_toks: list[str] = [ llm_math_tokenizer.decode(tok) for tok in torch.cat([section.input_tokens_S for section in sections]) .detach() .cpu() .numpy() ]
    math_pred_toks: list[str] = [ llm_math_tokenizer.decode(tok) for tok in torch.cat([section.math_pred_toks_S for section in sections]) .detach() .cpu() .numpy() ]
    r1_pred_toks: list[str] = [ llm_r1_tuned_tokenizer.decode(tok) for tok in torch.cat([section.r1_pred_toks_S for section in sections]) .detach() .cpu() .numpy() ]
    return (
        kl_div_S,
        mse_SL,
        acts_math_SLD,
        acts_r1_SLD,
        input_seq_toks,
        math_pred_toks,
        r1_pred_toks,
    )


# %%


# seq_logits_SV, resid_SLD, toks_S = get_logits_and_resid(r1_answer[:600], llm_r1_tuned, 4)

# prompt = r1_answer
# model = llm_r1_tuned
# every_n_layers = 4


# hookpoints = [f"blocks.{i}.hook_resid_pre" for i in range(model.cfg.n_layers) if i % every_n_layers == 0]
# toks: torch.Tensor = model.tokenizer.encode(prompt, return_tensors="pt")  # type: ignore
# assert toks.shape[0] == 1
# # %%
# len(toks[0])
# # %%
# print(model.tokenizer.decode(toks[0, 1000:]))
# # %%
# seq_logits, cache = model.run_with_cache(
#     toks[:500],
#     names_filter=lambda name: name in hookpoints,
#     pos_slice=slice(1000, None),
# )
# # %%
# seq_logits.shape
# # %%
# cache['blocks.0.hook_resid_pre'].shape
# # %%
# toks_S = toks[0]
# seq_logits_SV = seq_logits[0]
# cache_ = cache.remove_batch_dim()
# resid_SLD = torch.stack([cache_.cache_dict[hp] for hp in hookpoints ]).transpose(0, 1)

# %%

answer = r1_answer[:600]

sections = get_seq_data(answer, llm_math, llm_r1_tuned, every_n_layers=4)

(
    kl_div_S,
    mse_SL,
    acts_math_SLD,
    acts_r1_SLD,
    input_seq_toks,
    math_pred_toks,
    r1_pred_toks,
) = process_sections(sections, llm_math.tokenizer, llm_r1_tuned.tokenizer)  # type: ignore

# %%
pref_len = 19

seq_vis = visualise_text_sequence_vertical(
    kl_div_S=kl_div_S[pref_len:],
    mse_SL=mse_SL[pref_len:],
    acts_math_SLD=acts_math_SLD[pref_len:],
    acts_r1_SLD=acts_r1_SLD[pref_len:],
    input_sequence=input_seq_toks[pref_len:],
    math_pred_toks=math_pred_toks[pref_len:],
    r1_pred_toks=r1_pred_toks[pref_len:],
)
seq_vis
# %%
idcs = [len(kl_div_S[pref_len:]) - 1 - neg_offset for neg_offset in (127, 48, 20)]

[input_seq_toks[pref_len:][i] for i in idcs]
# %%
space_acts_SD = acts_r1_SLD[pref_len:][idcs, 4]
# %%

sdf = "The expression \\(1 + 2\\) represents adding 1 to 2.\n\nThe expression \\("
l0_SD = llm_math.embed(llm_math.tokenizer.encode(sdf, return_tensors='pt')[0])
end_emb_1D = llm_math.embed(llm_math.tokenizer.encode(r"\)", return_tensors='pt')[0])
# %%
# cts_toks
for space_act_D in torch.tensor(space_acts_SD, device='cuda').unbind(0):
    inp = torch.cat([l0_SD, space_act_D.unsqueeze(0), end_emb_1D], dim=0).unsqueeze(0)
    # print(inp)
    logits = llm_math.generate(inp, return_type="tokens")
    # print(logits)
    print(llm_math.tokenizer.decode(logits[0]))

    # input_toks = cts_toks + [
# [llm_math.tokenizer.decode(tok) for tok in cts_toks]

# %%

hookpoints = [
    f"blocks.{i}.hook_resid_pre" for i in range(llm_r1_tuned.cfg.n_layers) if i % 4 == 0
]

tokens_cropped = input_seq_toks[pref_len:]
mses_cropped_SL = torch.tensor(mse_SL[pref_len:])

max_by_layer_L = torch.max(mses_cropped_SL, dim=0).values
values_normed = mses_cropped_SL / max_by_layer_L[None]

toks_html = str(
    colored_tokens_multi(
        tokens=tokens_cropped,
        values=values_normed,
        labels=hookpoints,
    )
)
# %%

# seq on left in vertically scrolling div, toks on right, no scoll
website = f"""
<div style="display: flex; height: 1000px;">
    <div style="flex: 1; overflow-y: auto;">
        {seq_vis.to_html()}
    </div>
    <div style="flex: 1; overflow-y: auto;">
        {toks_html}
    </div>
</div>
"""
# render in the machine's browser

from IPython.display import HTML  # noqa: E402

HTML(website)
# %%

