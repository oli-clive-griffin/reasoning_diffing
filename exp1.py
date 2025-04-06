# import torch
# import transformer_lens
# # import transformers

# %%
import torch
from einops import reduce
from transformer_lens import HookedTransformer  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer  # type: ignore

from utils import find_common_subsections, get_logits_and_resid, get_seq_data, visualise_text_sequence, DATASET

CACHE_DIR = ".cache"
BASE = "Qwen/Qwen2.5-7B"
BASE_MATH = "Qwen/Qwen2.5-Math-7B"
R1_TUNED = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


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

llm_math, llm_math_hf, llm_math_tokenizer = make_model(BASE_MATH)
llm_r1_tuned, llm_r1_tuned_hf, llm_r1_tuned_tokenizer = make_model(R1_TUNED)

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
    return llm_r1_tuned.tokenizer.apply_chat_template(
        [{ "role": "user", "content": q }],
        add_generation_prompt=True,
        tokenize=False
    )

# %%

# def run_question(llm: HookedTransformer, question: str, toks: int = 400) -> str:
#     return llm.generate(question, max_new_tokens=toks)


# %%
q, c = DATASET[6].values()
# %%
print(f"Question: {q}")
print(f"Correct answer: {c}")
# %%
math_answer = llm_math.generate(fmt_math(q), max_new_tokens=300, do_sample=False)
# %%
r1_answer = llm_r1_tuned.generate(fmt_r1(q), max_new_tokens=2000, do_sample=False)

# %%

print(f"Math answer:\n{math_answer}")
# %%
print(f"R1 answer:\n{r1_answer}")

# %%
print(c)

# %%

torch.cuda.empty_cache()
# %%
r1_answer[:400]
# %%
prompt = r1_answer[:400]
data = get_seq_data(prompt, llm_math, llm_r1_tuned)
# %%

math_logits_SV, math_resid_SLD, math_toks_S = get_logits_and_resid(prompt, llm_math)
r1_logits_SV, r1_resid_SLD, r1_toks_S = get_logits_and_resid(prompt, llm_r1_tuned)
# %%
# math_logits_SV = math_logits_SV[x:]
# math_resid_SLD = math_resid_SLD[x:]
# math_toks_S = math_toks_S[x:]
# r1_logits_SV = r1_logits_SV[x:]
# r1_resid_SLD = r1_resid_SLD[x:]
# r1_toks_S = r1_toks_S[x:]
common_subsections = find_common_subsections(
    math_toks_S.tolist(), r1_toks_S.tolist()
)
# %%

sections: list[SeqData] = []
for (start_math, end_math), (start_r1, end_r1) in common_subsections:
    math_section_logits_SV = math_logits_SV[start_math:end_math]
    r1_section_logits_SV = r1_logits_SV[start_r1:end_r1]

    input_math_section = math_toks_S[start_math:end_math]
    input_r1_section = r1_toks_S[start_r1:end_r1]
    assert (input_math_section == input_r1_section).all()
    input_tokens_S = input_math_section

    math_resid_section_SLD = math_resid_SLD[start_math:end_math]
    r1_resid_section_SLD = r1_resid_SLD[start_r1:end_r1]

    math_seq_probs_SV = math_section_logits_SV.softmax(dim=-1)
    r1_seq_probs_SV = r1_section_logits_SV.softmax(dim=-1)

    math_seq_preds_S = math_section_logits_SV.argmax(dim=-1)
    r1_seq_preds_S = r1_section_logits_SV.argmax(dim=-1)

    kl_div_S = tokenwise_kl(
        probs_P_SV=math_seq_probs_SV, probs_Q_SV=r1_seq_probs_SV
    )

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




sections = get_seq_data(r1_answer[:400], llm_math, llm_r1_tuned)


# %%

kl_div_S = (
    torch.cat([section.kl_div_S for section in sections]).detach().float().cpu().numpy()
)
mse_SL = (
    torch.cat([section.mse_SL for section in sections]).detach().float().cpu().numpy()
)
input_seq_toks: list[str] = [
    llm_math_tokenizer.decode(tok)
    for tok in torch.cat([section.input_tokens_S for section in sections])
    .detach()
    .cpu()
    .numpy()
]
math_pred_toks: list[str] = [
    llm_math_tokenizer.decode(tok)
    for tok in torch.cat([section.math_pred_toks_S for section in sections])
    .detach()
    .cpu()
    .numpy()
]
r1_pred_toks: list[str] = [
    llm_r1_tuned_tokenizer.decode(tok)
    for tok in torch.cat([section.r1_pred_toks_S for section in sections])
    .detach()
    .cpu()
    .numpy()
]

# %%
kl_div_S.shape
# %%

visualise_text_sequence(
    kl_div_S=kl_div_S[18:],
    mse_SL=mse_SL[18:],
    input_sequence=input_seq_toks[18:],
    math_pred_toks=math_pred_toks[18:],
    r1_pred_toks=r1_pred_toks[18:],
)
#%% 

# %%

import plotly.express as px
px.imshow(mse_SL)
# %%
