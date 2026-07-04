"""
Rebuild report_datavis_highlighted.tex from scratch.
Uses single-shot replacements (open+close in one block) to avoid mismatches.
"""

SRC = r'C:\Users\Duc Do\Documents\sjepa\report_datavis.tex'
DST = r'C:\Users\Duc Do\Documents\sjepa\report_datavis_highlighted.tex'

with open(SRC, 'r', encoding='utf-8') as f:
    txt = f.read()

# ── 1. Add mdframed to preamble ───────────────────────────────────────────────
txt = txt.replace(
    r'\usepackage{balance}',
    r"""\usepackage{balance}
\usepackage[framemethod=default]{mdframed}
\mdfdefinestyle{newstyle}{
  backgroundcolor=yellow!28,
  linecolor=orange!70,
  linewidth=0.8pt,
  skipabove=3pt, skipbelow=3pt,
  innerleftmargin=5pt, innerrightmargin=5pt,
  innertopmargin=4pt, innerbottommargin=4pt,
}
\newenvironment{newcontent}{\begin{mdframed}[style=newstyle]}{\end{mdframed}}"""
)
print("[OK] preamble")

# ── 2. Email ──────────────────────────────────────────────────────────────────
txt = txt.replace(
    'duc.dna2414618@sis.hust.edu.vn',
    r'\colorbox{yellow!50}{duc.dna2414618@sis.hust.edu.vn}'
)
print("[OK] email")

# ── 2b. Highlight reordered author name (Duc moved to first) ─────────────────
txt = txt.replace(
    r'\IEEEauthorblockN{Do Nguyen Anh Duc (202414618), Nguyen Quang Tung (20233884), Nguyen Thi Thuy Huyen (20233854)}',
    r'\IEEEauthorblockN{\colorbox{yellow!40}{Do Nguyen Anh Duc (202414618)}, Nguyen Quang Tung (20233884), Nguyen Thi Thuy Huyen (20233854)}'
)
print("[OK] author order highlight")

TEXT_REPLACEMENTS = [
    # 2.5 Abstract sentence
    (
        r"to demystify the learned latent feature space. Experimental results demonstrate that the proposed multi-stream S-JEPA framework achieves competitive performance, reaching 78.30\% on NTU-120 X-Sub and 80.50\% on NTU-120 X-Set through weighted late fusion, while maintaining high computational efficiency.",
        r"""to demystify the learned latent feature space. \begin{newcontent}Experimental results demonstrate that the proposed multi-stream S-JEPA framework achieves competitive performance, reaching 78.30\% on NTU-120 X-Sub and 80.50\% on NTU-120 X-Set through weighted late fusion, while maintaining high computational efficiency.\end{newcontent}"""
    ),
]
for old_s, new_s in TEXT_REPLACEMENTS:
    txt = txt.replace(old_s, new_s, 1)

# ── 3. Joint-feature paragraph + itemize (wrap only text, stop before figure*) 
OLD3 = r"""To directly connect these abstract latent features to the physical body joints (bridging SHAP and Gradient Saliency), we compute the Pearson correlation coefficient between the joint-specific tokens and the global aggregated representations across action samples. The Joint-Feature Correlation maps are illustrated in Figure \ref{fig:joint_feature_relation}:
\begin{itemize}
    \item \textbf{Waving Hand (Figure \ref{fig:joint_feat_waving}):} The top latent classifier features (e.g., Features 43, 76) show exceptionally high positive correlation (above $0.70$) specifically with the left and right hand joints ($J_7$ and $J_{11}$), showing that the model's most critical waving features directly encode hand movements.
    \item \textbf{Jumping Action (Figure \ref{fig:joint_feat_jumping}):} The top jumping features (e.g., Features 12, 114) correlate strongly with the lower limb joints ($J_{15}$ and $J_{19}$ for left/right feet, and $J_0$ for the spine base), mathematically proving that the latent features driving the jumping classification correspond directly to physical leg dynamics.
\end{itemize}"""

NEW3 = r"""\begin{newcontent}
To directly connect these abstract latent features to the physical body joints (bridging SHAP and Gradient Saliency), we compute the Pearson correlation coefficient between the joint-specific tokens and the global aggregated representations across action samples. The Joint-Feature Correlation maps are illustrated in Figure \ref{fig:joint_feature_relation}:
\begin{itemize}
    \item \textbf{Waving Hand (Figure \ref{fig:joint_feat_waving}):} The top latent classifier features (e.g., Features 43, 76) show exceptionally high positive correlation (above $0.70$) specifically with the left and right hand joints ($J_7$ and $J_{11}$), showing that the model's most critical waving features directly encode hand movements.
    \item \textbf{Jumping Action (Figure \ref{fig:joint_feat_jumping}):} The top jumping features (e.g., Features 12, 114) correlate strongly with the lower limb joints ($J_{15}$ and $J_{19}$ for left/right feet, and $J_0$ for the spine base), mathematically proving that the latent features driving the jumping classification correspond directly to physical leg dynamics.
\end{itemize}
\end{newcontent}"""
print("[OK] block3" if OLD3 in txt else "[!!] MISS block3")
txt = txt.replace(OLD3, NEW3, 1)

# ── 4. Downstream intro paragraph (after subsection heading, before figure*)  
OLD4 = r"""\subsection{Downstream Model Evaluation and Error Analysis}
To quantitatively evaluate the representational capacity of S-JEPA, we assess the classification accuracy via a Linear Probing model on three individual feature streams (Joint, Bone, Velocity) and an Ensemble combination using the Weighted Late Fusion technique (the pipeline diagram is detailed in Figure \ref{fig:late_fusion_arch})."""

NEW4 = r"""\subsection{Downstream Model Evaluation and Error Analysis}
\begin{newcontent}
To quantitatively evaluate the representational capacity of S-JEPA, we assess the classification accuracy via a Linear Probing model on three individual feature streams (Joint, Bone, Velocity) and an Ensemble combination using the Weighted Late Fusion technique (the pipeline diagram is detailed in Figure \ref{fig:late_fusion_arch}).
\end{newcontent}"""
print("[OK] block4" if OLD4 in txt else "[!!] MISS block4")
txt = txt.replace(OLD4, NEW4, 1)

# ── 5. Section H subsection heading only ─────────────────────────────────────
OLD5 = r"""\subsection{Training Dynamics, Computational Performance, and Finetuning Comparison}"""
NEW5 = r"""\begin{newcontent}
\subsection{Training Dynamics, Computational Performance, and Finetuning Comparison}
\end{newcontent}"""
print("[OK] block5" if OLD5 in txt else "[!!] MISS block5")
txt = txt.replace(OLD5, NEW5, 1)

# ── 6. Training Dynamics text (2 paragraphs, wrap before next subsubsection) 
OLD6 = r"""\subsubsection{Training Dynamics and Convergence}
We analyze the training dynamics of the S-JEPA model in both the self-supervised pre-training and downstream linear probing phases (Figure \ref{fig:training_dynamics}). During the self-supervised pre-training phase, S-JEPA minimizes the L2 distance between the target patch representation and the predicted representation. As shown in Figure \ref{fig:training_dynamics}(a), the pre-training loss converges smoothly from $0.95$ to $0.058$ over 15 epochs. The stable convergence demonstrates S-JEPA's ability to learn consistent spatio-temporal representations from raw skeleton sequences without requiring label supervision.

In the downstream evaluation phase, a linear probing classifier is trained on the frozen representations. Figure \ref{fig:training_dynamics}(b) presents the learning curves (loss and accuracy) over 10 epochs. The training and validation curves track each other closely, reaching a stable validation accuracy of over $73.7\%$ on the Joint stream. The absence of a generalization gap between training and validation sets indicates the high stability and robustness of S-JEPA features against overfitting."""

NEW6 = r"""\begin{newcontent}
\subsubsection{Training Dynamics and Convergence}
We analyze the training dynamics of the S-JEPA model in both the self-supervised pre-training and downstream linear probing phases (Figure \ref{fig:training_dynamics}). During the self-supervised pre-training phase, S-JEPA minimizes the L2 distance between the target patch representation and the predicted representation. As shown in Figure \ref{fig:training_dynamics}(a), the pre-training loss converges smoothly from $0.95$ to $0.058$ over 15 epochs. The stable convergence demonstrates S-JEPA's ability to learn consistent spatio-temporal representations from raw skeleton sequences without requiring label supervision.

In the downstream evaluation phase, a linear probing classifier is trained on the frozen representations. Figure \ref{fig:training_dynamics}(b) presents the learning curves (loss and accuracy) over 10 epochs. The training and validation curves track each other closely, reaching a stable validation accuracy of over $73.7\%$ on the Joint stream. The absence of a generalization gap between training and validation sets indicates the high stability and robustness of S-JEPA features against overfitting.
\end{newcontent}"""
print("[OK] block6" if OLD6 in txt else "[!!] MISS block6")
txt = txt.replace(OLD6, NEW6, 1)

# ── 7. Computational text (before table) ─────────────────────────────────────
OLD7 = r"""\subsubsection{Computational Complexity and Efficiency}
To evaluate the computational efficiency of S-JEPA, we measure the training and inference time complexity. The experiment is conducted on an NVIDIA workstation with an RTX 4090 GPU. The computational times for each stage are detailed in Table \ref{tab:comp_time}.

Pre-training S-JEPA on the NTU RGB+D dataset requires only $1.5$ hours for 15 epochs. Downstream linear probing training is completed in just $5$ minutes. More importantly, the inference latency per sequence is only $2.4$ ms, corresponding to a processing speed of over $400$ frames per second (fps). This high efficiency proves S-JEPA is highly suitable for real-time human action recognition applications on edge devices."""

NEW7 = r"""\begin{newcontent}
\subsubsection{Computational Complexity and Efficiency}
To evaluate the computational efficiency of S-JEPA, we measure the training and inference time complexity. The experiment is conducted on an NVIDIA workstation with an RTX 4090 GPU. The computational times for each stage are detailed in Table \ref{tab:comp_time}.

Pre-training S-JEPA on the NTU RGB+D dataset requires only $1.5$ hours for 15 epochs. Downstream linear probing training is completed in just $5$ minutes. More importantly, the inference latency per sequence is only $2.4$ ms, corresponding to a processing speed of over $400$ frames per second (fps). This high efficiency proves S-JEPA is highly suitable for real-time human action recognition applications on edge devices.
\end{newcontent}"""
print("[OK] block7" if OLD7 in txt else "[!!] MISS block7")
txt = txt.replace(OLD7, NEW7, 1)

# ── 8. LP vs FT text + enumerate (before table) ──────────────────────────────
OLD8 = r"""\subsubsection{Linear Probing vs. Full Fine-Tuning Analysis}
To justify the choice of Linear Probing (frozen S-JEPA backbone) as our primary downstream evaluation method, we compare its performance and characteristics with Full Fine-Tuning reported in the original S-JEPA work \cite{b7}, where all weights of the Context Encoder are updated. The comparative results are summarized in Table \ref{tab:probe_vs_ft}.

Full Fine-Tuning yields slightly higher accuracy (+2.12\% on NTU-120 X-Sub) because the S-JEPA backbone weights are optimized to adapt directly to class boundaries. However, Linear Probing offers significant practical advantages:
\begin{enumerate}
    \item \textbf{Unbiased Evaluation of Self-Supervised Features:} Freezing the S-JEPA backbone guarantees that downstream performance is a direct measure of the representations learned during pre-training. It proves the backbone has captured intrinsic skeleton kinematics (e.g., bone structures, joint dynamics) that are linearly separable.
    \item \textbf{Computational Efficiency:} Freezing the backbone eliminates backpropagation through the Transformer blocks. This reduces GPU memory consumption by more than $60\%$ and speeds up training by a factor of 3 (from 15 minutes to 5 minutes).
    \item \textbf{Robustness to Overfitting:} Since a linear model has extremely low capacity, it is highly robust to overfitting. This is critical when downstream labeled data is small or imbalanced.
    \item \textbf{Catastrophic Forgetting Prevention:} Keeping the backbone frozen allows a single S-JEPA model to serve as a general-purpose feature extractor for multiple downstream tasks (e.g., action recognition, pose estimation, gesture control) simultaneously without cross-task degradation.
\end{enumerate}"""

NEW8 = r"""\begin{newcontent}
\subsubsection{Linear Probing vs. Full Fine-Tuning Analysis}
To justify the choice of Linear Probing (frozen S-JEPA backbone) as our primary downstream evaluation method, we compare its performance and characteristics with Full Fine-Tuning reported in the original S-JEPA work \cite{b7}, where all weights of the Context Encoder are updated. The comparative results are summarized in Table \ref{tab:probe_vs_ft}.

Full Fine-Tuning yields slightly higher accuracy (+2.12\% on NTU-120 X-Sub) because the S-JEPA backbone weights are optimized to adapt directly to class boundaries. However, Linear Probing offers significant practical advantages:
\begin{enumerate}
    \item \textbf{Unbiased Evaluation of Self-Supervised Features:} Freezing the S-JEPA backbone guarantees that downstream performance is a direct measure of the representations learned during pre-training. It proves the backbone has captured intrinsic skeleton kinematics (e.g., bone structures, joint dynamics) that are linearly separable.
    \item \textbf{Computational Efficiency:} Freezing the backbone eliminates backpropagation through the Transformer blocks. This reduces GPU memory consumption by more than $60\%$ and speeds up training by a factor of 3 (from 15 minutes to 5 minutes).
    \item \textbf{Robustness to Overfitting:} Since a linear model has extremely low capacity, it is highly robust to overfitting. This is critical when downstream labeled data is small or imbalanced.
    \item \textbf{Catastrophic Forgetting Prevention:} Keeping the backbone frozen allows a single S-JEPA model to serve as a general-purpose feature extractor for multiple downstream tasks (e.g., action recognition, pose estimation, gesture control) simultaneously without cross-task degradation.
\end{enumerate}
\end{newcontent}"""
print("[OK] block8" if OLD8 in txt else "[!!] MISS block8")
txt = txt.replace(OLD8, NEW8, 1)

# ── 9. Captions of new/modified figures and tables ────────────────────────────
# Use \caption[short text]{highlighted text} so hyperref bookmarks stay clean
CAPS = [
    # Figure 4 (Changed caption)
    (
        r"\caption{3D skeleton before/after geometric normalization (Action A27).}",
        r"\caption[3D skeleton geometric normalization.]{\setlength{\fboxsep}{2pt}\colorbox{yellow!40}{\parbox{\dimexpr\linewidth-4pt\relax}{3D skeleton before/after geometric normalization (Action A27).}}}"
    ),
    # Figure 10 (Saliency heatmap - layout changed/considered modified)
    (
        r"\caption{Spatio-temporal Gradient Saliency Heatmap demonstrating the model's decision sensitivity to skeleton joints over time.}",
        r"\caption[Spatio-temporal Gradient Saliency Heatmap.]{\setlength{\fboxsep}{2pt}\colorbox{yellow!40}{\parbox{\dimexpr\linewidth-4pt\relax}{Spatio-temporal Gradient Saliency Heatmap demonstrating the model's decision sensitivity to skeleton joints over time.}}}"
    ),
    (
        r"\caption{Joint-Feature Correlation maps connecting the top 5 SHAP features (latent space) directly to the physical joint representations (spatial dimensions).}",
        r"\caption[Joint-Feature Correlation maps.]{\setlength{\fboxsep}{2pt}\colorbox{yellow!40}{\parbox{\dimexpr\linewidth-4pt\relax}{Joint-Feature Correlation maps connecting the top 5 SHAP features (latent space) directly to the physical joint representations (spatial dimensions).}}}"
    ),
    (
        r"\caption{Training dynamics of S-JEPA: (a) self-supervised pre-training loss convergence over 15 epochs, (b) downstream linear probing learning curves over 10 epochs.}",
        r"\caption[Training dynamics of S-JEPA.]{\setlength{\fboxsep}{2pt}\colorbox{yellow!40}{\parbox{\dimexpr\linewidth-4pt\relax}{Training dynamics of S-JEPA: (a) self-supervised pre-training loss convergence over 15 epochs, (b) downstream linear probing learning curves over 10 epochs.}}}"
    ),
    (
        r"\caption{Confusion Matrix of the Ensemble Weighted Late Fusion classifier on NTU-120 X-Sub ($78.30\%$ accuracy).}",
        r"\caption[Confusion Matrix -- Ensemble Weighted Late Fusion (78.30\%).]{\setlength{\fboxsep}{2pt}\colorbox{yellow!40}{\parbox{\dimexpr\linewidth-4pt\relax}{Confusion Matrix of the Ensemble Weighted Late Fusion classifier on NTU-120 X-Sub ($78.30\%$ accuracy).}}}"
    ),
    (
        r"\caption{Computational Training and Inference Complexity of S-JEPA.}",
        r"\caption[Computational Complexity of S-JEPA.]{\setlength{\fboxsep}{2pt}\colorbox{yellow!40}{\parbox{\dimexpr\linewidth-4pt\relax}{Computational Training and Inference Complexity of S-JEPA.}}}"
    ),
    (
        r"\caption{Classification Performance Comparison: Linear Probing vs. Full Fine-Tuning.}",
        r"\caption[Linear Probing vs. Full Fine-Tuning.]{\setlength{\fboxsep}{2pt}\colorbox{yellow!40}{\parbox{\dimexpr\linewidth-4pt\relax}{Classification Performance Comparison: Linear Probing vs. Full Fine-Tuning.}}}"
    ),
]
for old_c, new_c in CAPS:
    found = old_c in txt
    print(f"[{'OK' if found else '!!'}] caption: {old_c[9:50]}...")
    if found:
        txt = txt.replace(old_c, new_c, 1)

# ── Write ─────────────────────────────────────────────────────────────────────
with open(DST, 'w', encoding='utf-8') as f:
    f.write(txt)
print(f"\nDone! {len(txt)} bytes -> {DST}")
