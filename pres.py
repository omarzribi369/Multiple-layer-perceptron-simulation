import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

st.set_page_config(page_title="MLP Presentation", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');
html, body, [class*="css"] { font-family: 'Lora', serif; }
h1 { font-family: 'Lora', serif; font-weight: 600; font-size: 22px; margin-bottom: 4px; }
.iter-box {
    background: #fff8f0;
    border-left: 3px solid #e07c39;
    padding: 10px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    line-height: 1.85;
    border-radius: 0 6px 6px 0;
}
.sec {
    font-family: 'Lora', serif; font-size: 13px; font-weight: 600;
    color: #1a1a2e; border-bottom: 1px solid #e0ddd5;
    padding-bottom: 3px; margin: 8px 0 5px 0;
}
[data-testid="stDataFrame"] { font-size: 11px; }
div[data-testid="column"] { padding: 0 4px; }
.ctrl-bar {
    background: #f4f1ec;
    border-radius: 10px;
    padding: 10px 16px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 12px;
}
</style>
""", unsafe_allow_html=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

def sig_d(a):
    return a * (1 - a)

def rand_weights():
    rng = np.random.default_rng()
    return {k: 0.2 for k in ['w00','w01','w10','w11','v0','v1']}
    
def forward(x1, x2, ww):
    zh1 = x1*ww['w00'] + x2*ww['w01'] - 0.1
    h1 = sigmoid(zh1)
    zh2 = x1*ww['w10'] + x2*ww['w11'] - 0.1
    h2 = sigmoid(zh2)
    zo = h1*ww['v0'] + h2*ww['v1'] - 0.1
    yp = sigmoid(zo)
    return zh1, h1, zh2, h2, zo, yp

def compute_grads(x1, x2, y, r, ww):
    # Simple loss: L = y - ŷ
    # ∂L/∂ŷ = -1
    dL_dy = -1.0
    
    # ∂ŷ/∂z_o = σ'(z_o) = ŷ(1-ŷ)
    dy_dzo = sig_d(r['yp'])
    
    # δ_o = ∂L/∂ŷ · ∂ŷ/∂z_o
    delta_o = dL_dy * dy_dzo
    
    # ∂L/∂v = δ_o · h
    ddv0 = delta_o * r['h1']
    ddv1 = delta_o * r['h2']
    
    # δ_h = δ_o · v · σ'(z_h)
    dh1 = delta_o * ww['v0'] * sig_d(r['h1'])
    dh2 = delta_o * ww['v1'] * sig_d(r['h2'])
    
    # ∂L/∂w = δ_h · x
    return {
        'dL_dy': dL_dy, 
        'dy_dzo': dy_dzo, 
        'delta_o': delta_o,
        'ddv0': ddv0, 
        'ddv1': ddv1, 
        'dh1': dh1, 
        'dh2': dh2,
        'dw00': dh1 * x1, 
        'dw01': dh1 * x2,
        'dw10': dh2 * x1, 
        'dw11': dh2 * x2,
    }

def apply_step(ww, bp, lr):
    return {
        'w00': ww['w00'] - lr * bp['dw00'], 
        'w01': ww['w01'] - lr * bp['dw01'],
        'w10': ww['w10'] - lr * bp['dw10'], 
        'w11': ww['w11'] - lr * bp['dw11'],
        'v0': ww['v0'] - lr * bp['ddv0'], 
        'v1': ww['v1'] - lr * bp['ddv1'],
    }

# Session state initialization
if 'w' not in st.session_state: 
    st.session_state.w = rand_weights()
if 'iteration' not in st.session_state: 
    st.session_state.iteration = 0
if 'x1' not in st.session_state: 
    st.session_state.x1 = 0.5
if 'x2' not in st.session_state: 
    st.session_state.x2 = 0.8
if 'y_true' not in st.session_state: 
    st.session_state.y_true = 0.7
if 'loss_history' not in st.session_state: 
    st.session_state.loss_history = []
if 'w_history' not in st.session_state: 
    st.session_state.w_history = []

w = st.session_state.w

x1, x2, y = st.session_state.x1, st.session_state.x2, st.session_state.y_true
zh1, h1, zh2, h2, zo, yp = forward(x1, x2, w)
r = {'yp': yp, 'h1': h1, 'h2': h2, 'zh1': zh1, 'zh2': zh2, 'zo': zo}
loss = y - yp  # Simple loss (not squared)
bp = compute_grads(x1, x2, y, r, w)
w_new = apply_step(w, bp, 0.5)

st.title("MLP Presentation")

ctrl_col, net_col = st.columns([1, 4])

with ctrl_col:
    st.markdown('<div class="sec">Sample</div>', unsafe_allow_html=True)
    st.session_state.x1 = st.number_input("x₁", 0.0, 1.0, st.session_state.x1, 0.05, key="inp_x1")
    st.session_state.x2 = st.number_input("x₂", 0.0, 1.0, st.session_state.x2, 0.05, key="inp_x2")
    st.session_state.y_true = st.number_input("y (true)", 0.0, 2.0, st.session_state.y_true, 0.05, key="inp_y")

    x1, x2, y = st.session_state.x1, st.session_state.x2, st.session_state.y_true
    zh1, h1, zh2, h2, zo, yp = forward(x1, x2, w)
    r = {'yp': yp, 'h1': h1, 'h2': h2, 'zh1': zh1, 'zh2': zh2, 'zo': zo}
    loss = y - yp  # Simple loss
    bp = compute_grads(x1, x2, y, r, w)

    st.markdown('<div class="sec">Training</div>', unsafe_allow_html=True)
    lr = st.slider("η (learning rate)", 0.01, 2.0, 0.5, 0.01)
    w_new = apply_step(w, bp, lr)

    st.markdown(f"**Loss (y - ŷ):** `{loss:.6f}`")

with net_col:
    st.markdown('<div class="sec">Network</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.8)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('#fafaf7')
    ax.set_facecolor('#fafaf7')

    pos = {'x1': (1.5, 3.2), 'x2': (1.5, 1.6), 'h1': (5.5, 3.2), 'h2': (5.5, 1.6), 'y': (9.5, 2.4)}
    R = 0.38

    def draw_conn(src, dst, weight, label, off=(0, 0)):
        sx, sy = pos[src]
        dx, dy = pos[dst]
        col = '#2d6a4f' if weight >= 0 else '#c0392b'
        lw = max(0.8, min(5, abs(weight) * 3.5))
        ang = np.arctan2(dy - sy, dx - sx)
        ax.annotate("", xy=(dx - np.cos(ang) * R, dy - np.sin(ang) * R),
                    xytext=(sx + np.cos(ang) * R, sy + np.sin(ang) * R),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, mutation_scale=14))
        ax.text((sx + dx) / 2 + off[0], (sy + dy) / 2 + off[1], f"{label}={weight:.3f}",
                ha='center', va='center', fontsize=7.8, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.2", fc='white', ec=col, alpha=0.93))

    draw_conn('x1', 'h1', w['w00'], 'w₀₀', (0, 0.22))
    draw_conn('x2', 'h1', w['w01'], 'w₀₁', (0, 0.16))
    draw_conn('x1', 'h2', w['w10'], 'w₁₀', (0, -0.16))
    draw_conn('x2', 'h2', w['w11'], 'w₁₁', (0, -0.22))
    draw_conn('h1', 'y', w['v0'], 'v₀', (0, 0.22))
    draw_conn('h2', 'y', w['v1'], 'v₁', (0, -0.22))

    def draw_neuron(key, lbl, val, color):
        cx, cy = pos[key]
        ax.add_patch(mpatches.Circle((cx, cy), R, fc=color, ec='#2a2a2a', lw=2.0, zorder=3))
        ax.text(cx, cy + 0.10, lbl, ha='center', va='center', fontsize=10.5, fontweight='bold', zorder=4)
        ax.text(cx, cy - 0.13, f"{val:.3f}", ha='center', va='center', fontsize=8, color='#333', zorder=4)

    draw_neuron('x1', 'x₁', x1, '#aed9e0')
    draw_neuron('x2', 'x₂', x2, '#aed9e0')
    draw_neuron('h1', 'h₁', h1, '#b7e4c7')
    draw_neuron('h2', 'h₂', h2, '#b7e4c7')
    draw_neuron('y', 'ŷ', yp, '#f4a261')

    for lbl, xp, yp_ in [('Input', 1.5, 4.3), ('Hidden', 5.5, 4.3), ('Output', 9.5, 4.3)]:
        ax.text(xp, yp_, lbl, ha='center', fontsize=9.5, color='#666', style='italic')

    st.pyplot(fig, use_container_width=True)
    plt.close()

st.markdown("---")

bar_reset, bar_back, bar_iter, bar_fwd, bar_spacer = st.columns([1.2, 0.9, 0.7, 0.9, 4.3])

with bar_reset:
    do_reset = st.button("🔀 Random reset", use_container_width=True)
with bar_back:
    step_back = st.button("◀ Back", use_container_width=True,
                          disabled=(len(st.session_state.w_history) == 0))
with bar_iter:
    st.markdown(
        f"<div style='text-align:center;font-family:JetBrains Mono,monospace;"
        f"font-size:12px;padding-top:4px;color:#555;'>"
        f"iter &nbsp;<b style='font-size:15px;color:#1a1a2e'>{st.session_state.iteration}</b></div>",
        unsafe_allow_html=True)
with bar_fwd:
    step_fwd = st.button("Fwd ▶", use_container_width=True, type="primary")

# Apply actions
if do_reset:
    st.session_state.w = rand_weights()
    st.session_state.iteration = 0
    st.session_state.loss_history = []
    st.session_state.w_history = []
    st.rerun()

if step_fwd:
    st.session_state.w_history.append(dict(w))
    st.session_state.loss_history.append(loss)
    st.session_state.w = w_new
    st.session_state.iteration += 1
    st.rerun()

if step_back and st.session_state.w_history:
    st.session_state.w = st.session_state.w_history.pop()
    if st.session_state.loss_history:
        st.session_state.loss_history.pop()
    st.session_state.iteration = max(0, st.session_state.iteration - 1)
    st.rerun()

# Tables / Equations / Iteration
col_tbl, col_eq, col_it = st.columns([1.5, 2.1, 2.8])

with col_tbl:
    st.markdown('<div class="sec">Values</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        'Variable': ['x₁', 'x₂', 'y (true)', 'ŷ (pred)', 'Loss (y - ŷ)'],
        'Value': [f"{x1:.4f}", f"{x2:.4f}", f"{y:.4f}", f"{yp:.4f}", f"{loss:.6f}"],
    }), hide_index=True, use_container_width=True)

    st.markdown('<div class="sec">Weights</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        'w': ['w₀₀', 'w₀₁', 'w₁₀', 'w₁₁', 'v₀', 'v₁'],
        'Previous': [f"{w[k]:.4f}" for k in ['w00', 'w01', 'w10', 'w11', 'v0', 'v1']],
        'Updated': [f"{w_new[k]:.4f}" for k in ['w00', 'w01', 'w10', 'w11', 'v0', 'v1']],
    }), hide_index=True, use_container_width=True)

with col_eq:
    st.markdown('<div class="sec">Equations</div>', unsafe_allow_html=True)
    eq_blocks = [
        ("Forward pass", None, None),
        (None, r"$z_{h_1} = x_1 w_{00} + x_2 w_{01} + b$", r"$h_1 = \sigma(z_{h_1})$"),
        (None, r"$z_{h_2} = x_1 w_{10} + x_2 w_{11} + b$", r"$h_2 = \sigma(z_{h_2})$"),
        (None, r"$z_o = h_1 v_0 + h_2 v_1 + b$", r"$\hat{y} = \sigma(z_o)$"),
        (None, r"$\text{Loss} = y - \hat{y}$", None),
        ("Chain rule", None, None),
        (None, r"$\frac{\partial L}{\partial v} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_o} \cdot \frac{\partial z_o}{\partial v}$", None),
        (None, r"$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_o} \cdot \frac{\partial z_o}{\partial h} \cdot \frac{\partial h}{\partial z_h} \cdot \frac{\partial z_h}{\partial w}$", None),
        ("Backprop — output layer", None, None),
        (None, r"$\frac{\partial L}{\partial \hat{y}} = -1$", r"$\sigma'(z_o) = \hat{y}(1-\hat{y})$"),
        (None, r"$\delta_o = \frac{\partial L}{\partial \hat{y}} \cdot \sigma'(z_o)$", None),
        (None, r"$\frac{\partial L}{\partial v_0} = \delta_o \cdot h_1$", r"$\frac{\partial L}{\partial v_1} = \delta_o \cdot h_2$"),
        ("Backprop — hidden layer", None, None),
        (None, r"$\delta_{h_1} = \delta_o \cdot v_0 \cdot \sigma'(z_{h_1})$", r"$\sigma'(z_{h_1}) = h_1(1-h_1)$"),
        (None, r"$\delta_{h_2} = \delta_o \cdot v_1 \cdot \sigma'(z_{h_2})$", r"$\sigma'(z_{h_2}) = h_2(1-h_2)$"),
        (None, r"$\frac{\partial L}{\partial w_{00}} = \delta_{h_1} \cdot x_1$", r"$\frac{\partial L}{\partial w_{01}} = \delta_{h_1} \cdot x_2$"),
        (None, r"$\frac{\partial L}{\partial w_{10}} = \delta_{h_2} \cdot x_1$", r"$\frac{\partial L}{\partial w_{11}} = \delta_{h_2} \cdot x_2$"),
        ("Gradient descent", None, None),
        (None, r"$v \leftarrow v - \eta \cdot \frac{\partial L}{\partial v}$", r"$w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}$"),
    ]
    
    n_eq = sum(1 for b in eq_blocks if b[0] is None)
    n_sec = sum(1 for b in eq_blocks if b[0] is not None)
    total_u = n_eq * 0.45 + n_sec * 0.30
    fig2, ax2 = plt.subplots(figsize=(6, total_u + 0.2))
    fig2.patch.set_facecolor('#fafaf7')
    ax2.set_facecolor('#fafaf7')
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, total_u)
    yc = total_u - 0.15
    
    for (sec, leq, req) in eq_blocks:
        if sec is not None:
            ax2.text(5, yc, sec, ha='center', va='center',
                     fontsize=11, fontstyle='italic', fontweight='bold', color='#2d6a4f')
            yc -= 0.30
        else:
            if leq:
                ax2.text(0.1, yc, leq, ha='left', va='center', fontsize=11, color='#1a1a2e')
            if req:
                ax2.text(5.1, yc, req, ha='left', va='center', fontsize=11, color='#1a1a2e')
            yc -= 0.45
    
    st.pyplot(fig2, use_container_width=True)
    plt.close()

with col_it:
    st.markdown(f'<div class="sec">Iteration {st.session_state.iteration}</div>', unsafe_allow_html=True)
    p = ' &nbsp;│&nbsp; '
    html = '<div class="iter-box">'
    # Forward pass
    html += "<b>Forward pass</b><br>"
    html += f"zₕ₁ = {x1:.3f}·{w['w00']:.3f} + {x2:.3f}·{w['w01']:.3f} - 0.1 = <b>{zh1:.4f}</b> &nbsp;→&nbsp; h₁ = σ(zₕ₁) = <b>{h1:.4f}</b><br>"
    html += f"zₕ₂ = {x1:.3f}·{w['w10']:.3f} + {x2:.3f}·{w['w11']:.3f} - 0.1 = <b>{zh2:.4f}</b> &nbsp;→&nbsp; h₂ = σ(zₕ₂) = <b>{h2:.4f}</b><br>"
    html += f"z_o = {h1:.4f}·{w['v0']:.3f} + {h2:.4f}·{w['v1']:.3f} - 0.1 = <b>{zo:.4f}</b> &nbsp;→&nbsp; ŷ = σ(z_o) = <b>{yp:.4f}</b>"
    # Loss
    html += f"<br><b>Loss</b><br>"
    html += f"L = y - ŷ = {y:.3f} - {yp:.4f} = <b>{loss:.6f}</b>"
    # Backprop - output layer
    html += "<br><b>Backprop — output layer</b><br>"
    html += f"∂L/∂ŷ = -1<br>"
    html += f"σ'(z_o) = ŷ(1-ŷ) = {yp:.4f}·(1-{yp:.4f}) = <b>{bp['dy_dzo']:.4f}</b><br>"
    html += f"δ_o = ∂L/∂ŷ · σ'(z_o) = -1 · {bp['dy_dzo']:.4f} = <b>{bp['delta_o']:.4f}</b><br>"
    html += f"∂L/∂v₀ = δ_o · h₁ = {bp['delta_o']:.4f} · {h1:.4f} = <b>{bp['ddv0']:.4f}</b><br>"
    html += f"∂L/∂v₁ = δ_o · h₂ = {bp['delta_o']:.4f} · {h2:.4f} = <b>{bp['ddv1']:.4f}</b>"
    # Backprop - hidden layer
    html += "<br><b>Backprop — hidden layer</b><br>"
    html += f"σ'(zₕ₁) = h₁(1-h₁) = {h1:.4f}·(1-{h1:.4f}) = <b>{sig_d(h1):.4f}</b><br>"
    html += f"δ_h₁ = δ_o · v₀ · σ'(zₕ₁) = {bp['delta_o']:.4f} · {w['v0']:.4f} · {sig_d(h1):.4f} = <b>{bp['dh1']:.4f}</b><br>"
    html += f"∂L/∂w₀₀ = δ_h₁ · x₁ = {bp['dh1']:.4f} · {x1:.3f} = <b>{bp['dw00']:.4f}</b><br>"
    html += f"∂L/∂w₀₁ = δ_h₁ · x₂ = {bp['dh1']:.4f} · {x2:.3f} = <b>{bp['dw01']:.4f}</b><br>"
    html += f"<br>σ'(zₕ₂) = h₂(1-h₂) = {h2:.4f}·(1-{h2:.4f}) = <b>{sig_d(h2):.4f}</b><br>"
    html += f"δ_h₂ = δ_o · v₁ · σ'(zₕ₂) = {bp['delta_o']:.4f} · {w['v1']:.4f} · {sig_d(h2):.4f} = <b>{bp['dh2']:.4f}</b><br>"
    html += f"∂L/∂w₁₀ = δ_h₂ · x₁ = {bp['dh2']:.4f} · {x1:.3f} = <b>{bp['dw10']:.4f}</b><br>"
    html += f"∂L/∂w₁₁ = δ_h₂ · x₂ = {bp['dh2']:.4f} · {x2:.3f} = <b>{bp['dw11']:.4f}</b>"
    # Gradient descent
    html += f"<br><b>Gradient descent (η = {lr})</b><br>"
    html += f"v₀_new = {w['v0']:.4f} - {lr}·{bp['ddv0']:.4f} = <b>{w_new['v0']:.4f}</b><br>"
    html += f"v₁_new = {w['v1']:.4f} - {lr}·{bp['ddv1']:.4f} = <b>{w_new['v1']:.4f}</b><br>"
    html += f"w₀₀_new = {w['w00']:.4f} - {lr}·{bp['dw00']:.4f} = <b>{w_new['w00']:.4f}</b><br>"
    html += f"w₀₁_new = {w['w01']:.4f} - {lr}·{bp['dw01']:.4f} = <b>{w_new['w01']:.4f}</b><br>"
    html += f"w₁₀_new = {w['w10']:.4f} - {lr}·{bp['dw10']:.4f} = <b>{w_new['w10']:.4f}</b><br>"
    html += f"w₁₁_new = {w['w11']:.4f} - {lr}·{bp['dw11']:.4f} = <b>{w_new['w11']:.4f}</b><br>"
    html += f"<br><b>New loss would be: {y:.3f} - {yp:.4f} = {loss:.6f}</b>"
    html += "</div>"
    
    st.markdown(html, unsafe_allow_html=True)
