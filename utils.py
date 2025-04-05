import numpy as np
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore


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
        rows=5,
        cols=1,
        row_heights=[0.2, 0.2, 0.6, 0.6, 0.6],  # Adjusted heights for better visibility
        specs=[
            [{"type": "table"}],  # For tokens (vertical)
            [{"type": "heatmap"}],  # For KL divergence
            [{"type": "heatmap"}],  # For MSE
            [{"type": "heatmap"}],  # For MSE
            [{"type": "heatmap"}],  # For MSE
        ],
        subplot_titles=(
            "Input Tokens",
            "KL Divergence by Token",
            "MSEs by Token and Layer",
            "MSEs by Token and Layer (normed by layer)",
            "MSEs by Token and Layer (normed by token)",
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

    max_per_token_S = mse_LS.max(axis=0)
    mse_normed_by_token_LS = mse_LS / max_per_token_S[None, :]

    fig.add_trace(
        go.Heatmap(
            z=mse_LS,
            x=list(range(num_tokens)),  # Tokens on x-axis
            y=list(range(num_layers)),  # Layers on y-axis
            coloraxis="coloraxis2",
            hovertemplate="Token index: %{x}<br>Layer: %{y}<br>MSE: %{z:.4f}<extra></extra>",
            # zmin=0,
            # zmax=10,
        ),
        row=3,
        col=1,
    )


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
        row=4,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=mse_normed_by_token_LS,
            x=list(range(num_tokens)),  # Tokens on x-axis
            y=list(range(num_layers)),  # Layers on y-axis
            # coloraxis="viridis",
            hovertemplate="Token index: %{x}<br>Layer: %{y}<br>MSE: %{z:.4f}<extra></extra>",
            # zmin=0,
            # zmax=10,
        ),
        row=5,
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
        coloraxis2=dict(
            colorscale="Viridis",
            colorbar=dict(
                title="MSE",
                y=0.35,  # Position for MSE colorbar
                len=0.5,
            ),
            cmax=10
        ),
        # coloraxis3=dict(
        #     colorscale="Viridis",
        #     colorbar=dict(
        #         title="MSE",
        #         y=0.35,  # Position for MSE colorbar
        #         len=0.5,
        #     )
        # ),
        height=2000,  #max(800, num_layers * 20 + 400),  # Scale height based on layers
        width=max(1000, num_tokens * 60),  # Scale width based on tokens
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
    for i in [3, 4, 5]:
        fig.update_xaxes(
            title="Token Index",
            range=[-0.5, num_tokens - 0.5],
            row=i,
            col=1,
            tickmode="array",
            tickvals=list(range(num_tokens)),
            ticktext=[f"{i}" for i in range(num_tokens)],
        )
        fig.update_yaxes(
            title="Layer",
            range=[-0.5, num_layers - 0.5],
            row=i,
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
