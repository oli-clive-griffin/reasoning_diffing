import torch
import numpy as np
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer
from einops import reduce
from dataclasses import dataclass



def visualise_text_sequence(
    input_sequence: list[str],
    math_pred_toks: list[str],
    r1_pred_toks: list[str],
    kl_div_S: np.ndarray,
    mse_SL: np.ndarray,
):
    """
    Visualize a text sequence with corresponding KL divergence and MSE values.

    Args:
        input_sequence: List of input tokens/text segments
        math_pred_toks: Token predictions from math model
        r1_pred_toks: Token predictions from r1 model
        kl_div_S: KL divergence values per token (shape: [S])
        mse_SL: MSE values per token and layer (shape: [S, L])
    """
    assert kl_div_S.shape[0] == mse_SL.shape[0]
    assert kl_div_S.shape[0] == len(input_sequence)
    assert len(math_pred_toks) == len(r1_pred_toks)
    assert len(math_pred_toks) == len(input_sequence)

    # Calculate number of tokens and layers
    num_tokens = len(input_sequence)
    num_layers = mse_SL.shape[1]

    # Create figure with specified layout
    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.2, 0.2, 0.6],  # Adjusted heights for better visibility
        specs=[
            [{"type": "table"}],  # For tokens (vertical)
            [{"type": "heatmap"}],  # For KL divergence
            [{"type": "heatmap"}],  # For MSE
        ],
        subplot_titles=(
            "Input Tokens",
            "KL Divergence by Token",
            "MSEs by Token and Layer (normed by layer)",
        ),
        vertical_spacing=0.03,
    )

    # Row 1: Create a vertical display of tokens
    # Transpose the input sequence to make it display vertically
    token_table = [[token, math_pred_tok, r1_pred_tok] for token, math_pred_tok, r1_pred_tok in zip(input_sequence, math_pred_toks, r1_pred_toks)]

    # Add tokens as a table with rotated header
    fig.add_trace(
        go.Table(
            cells=dict(
                values=token_table,
                align="center",
                font=dict(size=10),
                height=25,
            ),
        ),
        row=1,
        col=1,
    )

    # Row 2: KL divergence heatmap
    # Create hover text with token information
    hover_texts = [
        f"Token: '{input_sequence[i]}' | Math: '{math_pred_toks[i]}' | R1: '{r1_pred_toks[i]}' | KL: {kl_div_S[i]:.4f}"
        for i in range(num_tokens)
    ]

    fig.add_trace(
        go.Heatmap(
            z=[kl_div_S],  # Make it 2D for heatmap
            x=list(range(num_tokens)),
            y=[0],  # Single row
            coloraxis="coloraxis1",
            hoverinfo="text",
            text=[hover_texts],
            showscale=True,
        ),
        row=2,
        col=1,
    )

    # Transpose MSE matrix for better visualization (layers on y-axis)
    mse_LS = mse_SL.T
    max_per_layer_L = mse_LS.max(axis=1)
    mse_normed_by_layer_LS = mse_LS / max_per_layer_L[:, None]


    fig.add_trace(
        go.Heatmap(
            z=mse_normed_by_layer_LS,
            x=list(range(num_tokens)),  # Tokens on x-axis
            y=list(range(num_layers)),  # Layers on y-axis
            # coloraxis="coloraxis2",
            hovertemplate="Token index: %{x}<br>Layer: %{y}<br>MSE: %{z:.4f}<extra></extra>",
            # zmin=0,
            # zmax=10,
        ),
        row=3,
        col=1,
    )

    # Update layout
    fig.update_layout(
        coloraxis1=dict(
            colorscale="Reds",
            colorbar=dict(
                title="KL Divergence",
                y=0.8,  # Position for KL colorbar
                len=0.2,
            ),
        ),
        height=1000,  #max(800, num_layers * 20 + 400),  # Scale height based on layers
        width=num_tokens * 60,  # Scale width based on tokens
        title="Token-wise Analysis with KL Divergence and MSE",
        margin=dict(t=80, b=50, l=80, r=50),
    )

    # Ensure the plots align by adjusting axes
    # For KL divergence plot
    fig.update_xaxes(
        title="Token Index",
        range=[-0.5, num_tokens - 0.5],
        row=2,
        col=1,
        tickmode="array",
        tickvals=list(range(num_tokens)),
        ticktext=[f"{i}" for i in range(num_tokens)],
    )
    fig.update_yaxes(
        showticklabels=False,  # Hide y-axis labels for KL (single row)
        row=2,
        col=1,
    )

    # For MSE plot
    fig.update_xaxes(
        title="Token Index",
        range=[-0.5, num_tokens - 0.5],
        row=3,
        col=1,
        tickmode="array",
        tickvals=list(range(num_tokens)),
        ticktext=[f"{i}" for i in range(num_tokens)],
    )
    fig.update_yaxes(
        title="Layer",
        range=[-0.5, num_layers - 0.5],
        row=3,
        col=1,
        tickmode="array",
        tickvals=list(range(num_layers)),
        ticktext=[f"{i}" for i in range(num_layers)],
    )

    # Return the figure for display
    return fig


def find_common_subsections(
    seq1: list[int], seq2: list[int]
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
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
    result: list[tuple[tuple[int, int], tuple[int, int]]] = []
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
if __name__ == "__main__":
    seq1 = [1, 2, 3, 8, 9]
    seq2 = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    print(find_common_subsections(seq1, seq2))


DATASET = [
  {
    "question": "Calculate 347 + 689.",
    "answer": "1036"
  },
  {
    "question": "Solve for x: 3x - 7 = 14",
    "answer": "x = 7"
  },
  {
    "question": "Find the area of a circle with radius 5 cm.",
    "answer": "Area = 25π ≈ 78.54 cm²"
  },
  {
    "question": "Solve the system of equations: 2x + y = 5 and x - y = 1",
    "answer": "x = 2, y = 1"
  },
  {
    "question": "Factor the expression: x² - 9x + 20",
    "answer": "(x - 4)(x - 5)"
  },
  {
    "question": "Calculate the derivative of f(x) = 4x³ - 7x² + 2x - 5",
    "answer": "f'(x) = 12x² - 14x + 2"
  },
  {
    "question": "Evaluate the definite integral: ∫₁³ (2x² + 3x) dx",
    "answer": "29.33"
  },
  {
    "question": "Find the maximum value of f(x) = -2x² + 12x - 10",
    "answer": "8"
  },
  {
    "question": "Solve the differential equation: dy/dx = 3x² - 2, given y(1) = 4",
    "answer": "y = x³ - 2x + 5"
  },
  {
    "question": "Find the eigenvalues of the matrix: [3 1; 2 2]",
    "answer": "λ = 4, λ = 1"
  },
  {
    "question": "Evaluate the limit: lim(x→∞) (3x² + 2x - 1)/(5x² - 3)",
    "answer": "3/5"
  },
  {
    "question": "Solve the recurrence relation: a₍ₙ₊₂₎ = a₍ₙ₊₁₎ + a₍ₙ₎ with a₀ = 1, a₁ = 1",
    "answer": "This is the Fibonacci sequence with a₀ = 1, a₁ = 1. The general formula is aₙ = [φⁿ - (1-φ)ⁿ]/√5 where φ = (1+√5)/2"
  },
  {
    "question": "Find the volume of the solid obtained by rotating the region bounded by y = x², y = 0, and x = 2 around the y-axis.",
    "answer": "8π cubic units"
  },
  {
    "question": "Prove that the function f(x) = e^x is its own derivative.",
    "answer": "f'(x) = lim(h→0) [f(x+h) - f(x)]/h = lim(h→0) [e^(x+h) - e^x]/h = lim(h→0) [e^x·e^h - e^x]/h = e^x·lim(h→0) [e^h - 1]/h = e^x·1 = e^x. Therefore, f'(x) = e^x"
  },
  {
    "question": "Solve the partial differential equation: ∂²u/∂x² + ∂²u/∂y² = 0",
    "answer": "This is Laplace's equation. Some solutions include u(x,y) = x² - y², u(x,y) = ln(x² + y²), and u(x,y) = e^x·cos(y). General solutions depend on boundary conditions."
  }
] 





def get_logits_and_resid(
    prompt: str, model: HookedTransformer
):
    hookpoints = [f"blocks.{i}.hook_resid_pre" for i in range(model.cfg.n_layers) if i % 4 == 0]
    toks: torch.Tensor = model.tokenizer.encode(prompt, return_tensors="pt")  # type: ignore
    assert toks.shape[0] == 1
    seq_logits, cache = model.run_with_cache(
        toks,
        names_filter=lambda name: name in hookpoints,
    )
    toks_S = toks[0]
    seq_logits_SV = seq_logits[0]
    cache_ = cache.remove_batch_dim()
    resid_SLD = torch.stack([cache_.cache_dict[hp] for hp in hookpoints ]).transpose(0, 1)
    return seq_logits_SV, resid_SLD, toks_S


def tokenwise_kl(probs_P_SV: torch.Tensor, probs_Q_SV: torch.Tensor):
    """S = seq, V = vocab"""
    tokens_kl_S = torch.sum(probs_P_SV * torch.log(probs_P_SV / probs_Q_SV), dim=-1)
    return tokens_kl_S



@dataclass
class SeqData:
    input_tokens_S: torch.Tensor
    math_pred_toks_S: torch.Tensor
    r1_pred_toks_S: torch.Tensor
    kl_div_S: torch.Tensor
    mse_SL: torch.Tensor


def get_seq_data(prompt: str, llm_math: HookedTransformer, llm_r1_tuned: HookedTransformer,) -> list[SeqData]:
    math_logits_SV, math_resid_SLD, math_toks_S = get_logits_and_resid(prompt, llm_math)
    r1_logits_SV, r1_resid_SLD, r1_toks_S = get_logits_and_resid(prompt, llm_r1_tuned)

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

    return sections
