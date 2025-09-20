## Requirements (format this better later)
- `uv` package manager, python 3.12+. Sync the packages with `uv sync`.
- PCam dataset, get the data with `make get-data`.
- Access to DINOv3 weights (request access from Meta), access with a CLI HuggingFace login.
- Basic command: `make model`

## Pruning methods applied in the code

We use three graph-safe compressions that keep external I/O shapes intact while shrinking internal channels/rank. Each uses an energy keep-fraction $\tau \in (0,1]$ and always keeps at least one unit.

**1) Attention-head pruning (by $o\_\text{proj}$ energy).**
Let a block have $H$ heads, model dim $d_{\text{model}}$, head dim $d_h=d_{\text{model}}/H$, and output projection $W_o\in\mathbb{R}^{d_{\text{model}}\times d_{\text{model}}}$. For head $h$, define its input-column slice $S_h=\{h d_h,\dots,(h+1)d_h-1\}$. Score and energy:

$$
s_h=\left\lVert W_o[:,S_h]\right\rVert_F,\qquad e_h=s_h^2.
$$

Sort $e_{(1)}\ge\cdots\ge e_{(H)}$. Choose the smallest $K$ s.t.

$$
\frac{\sum_{i=1}^{K} e_{(i)}}{\sum_{j=1}^{H} e_{(j)}}\ \ge\ \tau.
$$

Keep those $K$ heads. Implement by selecting the corresponding output rows in $W_q,W_k,W_v$ and input columns in $W_o$ (bias of $o$ unchanged); update the block’s head count to $K$.

**2) MLP-neuron pruning (width reduction by multiplicative salience).**
For an MLP with $W_{\text{up}}\in\mathbb{R}^{r\times d_{\text{in}}}$ and $W_{\text{down}}\in\mathbb{R}^{d_{\text{out}}\times r}$, define for hidden unit $i$:

$$
a_i=\bigl\lVert (W_{\text{up}})_{i,:}\bigr\rVert_2,\quad
b_i=\bigl\lVert (W_{\text{down}})_{:,i}\bigr\rVert_2,\quad
s_i=a_i\,b_i,\quad e_i=s_i^2.
$$

Sort $e_{(1)}\ge\cdots\ge e_{(r)}$. Keep the smallest $k$ s.t.

$$
\frac{\sum_{i=1}^{k} e_{(i)}}{\sum_{j=1}^{r} e_{(j)}}\ \ge\ \tau.
$$

Subselect rows of $W_{\text{up}}$ and columns of $W_{\text{down}}$ to width $k$ (biases subset accordingly). Input/output dims are unchanged.

**3) Truncated-SVD linear compression (per layer).**
For a linear $y=xW^\top+b$ with $W\in\mathbb{R}^{\text{out}\times\text{in}}$, compute the thin SVD

$$
W=U\Sigma V^\top,\quad \Sigma=\mathrm{diag}(\sigma_1,\dots,\sigma_m),\ m=\min(\text{out},\text{in}).
$$

Pick the smallest rank $r$ s.t.

$$
\frac{\sum_{i=1}^{r}\sigma_i^{\,2}}{\sum_{j=1}^{m}\sigma_j^{\,2}}\ \ge\ \tau,
$$

then use the rank-$r$ approximation $W\approx U_r\Sigma_r V_r^\top$ and realize it as two linears:

$$
xW^\top \approx (xV_r)\,(U_r\Sigma_r)^\top.
$$

We apply this only if it reduces parameters:

$$
r(\text{in}+\text{out}) + \mathbf{1}_{\{\text{bias}\}}\cdot \text{out}
\ <\
\text{in}\cdot \text{out} + \mathbf{1}_{\{\text{bias}\}}\cdot \text{out}.
$$

**Notes.** $\tau$ is the “energy to keep.” Head/unit/rank selection uses cumulative **squared** energy. All replacements preserve dtype/device and keep external tensor interfaces unchanged.

## License

- Code: MIT (see `LICENSE`).
- Pretrained and fine-tuned **DINOv3** weights: subject to Meta’s **DINOv3 License**; see the
  Hugging Face model card (License: dinov3-license) and Meta’s license page.
- Data: PCam is CC0 (per the PCam repo).

Note: Releasing any weights derived from DINOv3 requires complying with the DINOv3 License
terms (gated access, redistribution conditions).

