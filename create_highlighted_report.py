"""
Rebuild report_datavis_highlighted.tex from scratch.
Strategy:
- Use mdframed to create yellow-background boxes around NEW text blocks
- Highlight email with colorbox
- Do NOT wrap figure* environments (they can't be inside boxes)
"""
import re

SRC = r'C:\Users\Duc Do\Documents\sjepa\report_datavis.tex'
DST = r'C:\Users\Duc Do\Documents\sjepa\report_datavis_highlighted.tex'

with open(SRC, 'r', encoding='utf-8') as f:
    txt = f.read()

# ── 1. Add mdframed package + newcontent env to preamble ─────────────────────
txt = txt.replace(
    r'\usepackage{balance}',
    r'''\usepackage{balance}
\usepackage[framemethod=default]{mdframed}
\mdfdefinestyle{newstyle}{
  backgroundcolor=yellow!28,
  linecolor=orange!70,
  linewidth=0.8pt,
  skipabove=3pt, skipbelow=3pt,
  innerleftmargin=5pt, innerrightmargin=5pt,
  innertopmargin=4pt, innerbottommargin=4pt,
}
\newenvironment{newcontent}{\begin{mdframed}[style=newstyle]}{\end{mdframed}}'''
)
print("[OK] Added mdframed to preamble")

# ── 2. Highlight email in author block ───────────────────────────────────────
txt = txt.replace(
    'duc.dna2414618@sis.hust.edu.vn',
    r'\colorbox{yellow!50}{duc.dna2414618@sis.hust.edu.vn}'
)
print("[OK] Highlighted email")

# ── 3. Highlight the "To directly connect..." paragraph + itemize (NEW block) ─
OLD_CONNECT = r"""To directly connect these abstract latent features to the physical body joints (bridging SHAP and Gradient Saliency), we compute the Pearson correlation coefficient between the joint-specific tokens and the global aggregated representations across action samples. The Joint-Feature Correlation maps are illustrated in Figure \ref{fig:joint_feature_relation}:
\begin{itemize}
    \item \textbf{Waving Hand (Figure \ref{fig:joint_feat_waving}):} The top latent classifier features (e.g., Features 43, 76) show exceptionally high positive correlation (above $0.70$) specifically with the left and right hand joints ($J_7$ and $J_{11}$), showing that the model's most critical waving features directly encode hand movements.
    \item \textbf{Jumping Action (Figure \ref{fig:joint_feat_jumping}):} The top jumping features (e.g., Features 12, 114) correlate strongly with the lower limb joints ($J_{15}$ and $J_{19}$ for left/right feet, and $J_0$ for the spine base), mathematically proving that the latent features driving the jumping classification correspond directly to physical leg dynamics.
\end{itemize}"""

NEW_CONNECT = r"""\begin{newcontent}
To directly connect these abstract latent features to the physical body joints (bridging SHAP and Gradient Saliency), we compute the Pearson correlation coefficient between the joint-specific tokens and the global aggregated representations across action samples. The Joint-Feature Correlation maps are illustrated in Figure \ref{fig:joint_feature_relation}:
\begin{itemize}
    \item \textbf{Waving Hand (Figure \ref{fig:joint_feat_waving}):} The top latent classifier features (e.g., Features 43, 76) show exceptionally high positive correlation (above $0.70$) specifically with the left and right hand joints ($J_7$ and $J_{11}$), showing that the model's most critical waving features directly encode hand movements.
    \item \textbf{Jumping Action (Figure \ref{fig:joint_feat_jumping}):} The top jumping features (e.g., Features 12, 114) correlate strongly with the lower limb joints ($J_{15}$ and $J_{19}$ for left/right feet, and $J_0$ for the spine base), mathematically proving that the latent features driving the jumping classification correspond directly to physical leg dynamics.
\end{itemize}
\end{newcontent}"""

if OLD_CONNECT in txt:
    txt = txt.replace(OLD_CONNECT, NEW_CONNECT, 1)
    print("[OK] Highlighted joint-feature correlation paragraph")
else:
    print("[!!] MISS: joint-feature paragraph")

# ── 4. Highlight new intro paragraph in Downstream Evaluation section ─────────
OLD_DOWNSTREAM_INTRO = r"""To quantitatively evaluate the representational capacity of S-JEPA, we assess the classification accuracy via a Linear Probing model on three individual feature streams (Joint, Bone, Velocity) and an Ensemble combination using the Weighted Late Fusion technique (the pipeline diagram is detailed in Figure \ref{fig:late_fusion_arch}).

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.65\textwidth]{plots/confusion_matrix_ntu120_xsub.png}"""

NEW_DOWNSTREAM_INTRO = r"""\begin{newcontent}
To quantitatively evaluate the representational capacity of S-JEPA, we assess the classification accuracy via a Linear Probing model on three individual feature streams (Joint, Bone, Velocity) and an Ensemble combination using the Weighted Late Fusion technique (the pipeline diagram is detailed in Figure \ref{fig:late_fusion_arch}).
\end{newcontent}

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.65\textwidth]{plots/confusion_matrix_ntu120_xsub.png}"""

if OLD_DOWNSTREAM_INTRO in txt:
    txt = txt.replace(OLD_DOWNSTREAM_INTRO, NEW_DOWNSTREAM_INTRO, 1)
    print("[OK] Highlighted downstream intro paragraph")
else:
    print("[!!] MISS: downstream intro paragraph")

# ── 5. Highlight Section H: Training Dynamics subsection heading ───────────────
OLD_SECTH = r"""\subsection{Training Dynamics, Computational Performance, and Finetuning Comparison}

\begin{figure*}[t]"""

NEW_SECTH = r"""\begin{newcontent}
\subsection{Training Dynamics, Computational Performance, and Finetuning Comparison}
\end{newcontent}

\begin{figure*}[t]"""

if OLD_SECTH in txt:
    txt = txt.replace(OLD_SECTH, NEW_SECTH, 1)
    print("[OK] Highlighted Section H heading")
else:
    print("[!!] MISS: Section H heading")

# ── 6. Highlight subsubsection Training Dynamics + its text ──────────────────
OLD_TRAIN_DYN = r"""\subsubsection{Training Dynamics and Convergence}
We analyze the training dynamics of the S-JEPA model in both the self-supervised pre-training and downstream linear probing phases (Figure \ref{fig:training_dynamics}). During the self-supervised pre-training phase, S-JEPA minimizes the L2 distance between the target patch representation and the predicted representation. As shown in Figure \ref{fig:training_dynamics}(a), the pre-training loss converges smoothly from $0.95$ to $0.058$ over 15 epochs. The stable convergence demonstrates S-JEPA's ability to learn consistent spatio-temporal representations from raw skeleton sequences without requiring label supervision.

In the downstream evaluation phase, a linear probing classifier is trained on the frozen representations. Figure \ref{fig:training_dynamics}(b) presents the learning curves (loss and accuracy) over 10 epochs. The training and validation curves track each other closely, reaching a stable validation accuracy of over $73.7\%$ on the Joint stream. The absence of a generalization gap between training and validation sets indicates the high stability and robustness of S-JEPA features against overfitting."""

NEW_TRAIN_DYN = r"""\begin{newcontent}
\subsubsection{Training Dynamics and Convergence}
We analyze the training dynamics of the S-JEPA model in both the self-supervised pre-training and downstream linear probing phases (Figure \ref{fig:training_dynamics}). During the self-supervised pre-training phase, S-JEPA minimizes the L2 distance between the target patch representation and the predicted representation. As shown in Figure \ref{fig:training_dynamics}(a), the pre-training loss converges smoothly from $0.95$ to $0.058$ over 15 epochs. The stable convergence demonstrates S-JEPA's ability to learn consistent spatio-temporal representations from raw skeleton sequences without requiring label supervision.

In the downstream evaluation phase, a linear probing classifier is trained on the frozen representations. Figure \ref{fig:training_dynamics}(b) presents the learning curves (loss and accuracy) over 10 epochs. The training and validation curves track each other closely, reaching a stable validation accuracy of over $73.7\%$ on the Joint stream. The absence of a generalization gap between training and validation sets indicates the high stability and robustness of S-JEPA features against overfitting.
\end{newcontent}"""

if OLD_TRAIN_DYN in txt:
    txt = txt.replace(OLD_TRAIN_DYN, NEW_TRAIN_DYN, 1)
    print("[OK] Highlighted Training Dynamics subsubsection")
else:
    print("[!!] MISS: Training Dynamics subsubsection")

# ── 7. Highlight Computational Complexity subsubsection + text ────────────────
OLD_COMP = r"""\subsubsection{Computational Complexity and Efficiency}
To evaluate the computational efficiency of S-JEPA, we measure the training and inference time complexity. The experiment is conducted on an NVIDIA workstation with an RTX 4090 GPU. The computational times for each stage are detailed in Table \ref{tab:comp_time}.

Pre-training S-JEPA on the NTU RGB+D dataset requires only $1.5$ hours for 15 epochs. Downstream linear probing training is completed in just $5$ minutes. More importantly, the inference latency per sequence is only $2.4$ ms, corresponding to a processing speed of over $400$ frames per second (fps). This high efficiency proves S-JEPA is highly suitable for real-time human action recognition applications on edge devices."""

NEW_COMP = r"""\begin{newcontent}
\subsubsection{Computational Complexity and Efficiency}
To evaluate the computational efficiency of S-JEPA, we measure the training and inference time complexity. The experiment is conducted on an NVIDIA workstation with an RTX 4090 GPU. The computational times for each stage are detailed in Table \ref{tab:comp_time}.

Pre-training S-JEPA on the NTU RGB+D dataset requires only $1.5$ hours for 15 epochs. Downstream linear probing training is completed in just $5$ minutes. More importantly, the inference latency per sequence is only $2.4$ ms, corresponding to a processing speed of over $400$ frames per second (fps). This high efficiency proves S-JEPA is highly suitable for real-time human action recognition applications on edge devices.
\end{newcontent}"""

if OLD_COMP in txt:
    txt = txt.replace(OLD_COMP, NEW_COMP, 1)
    print("[OK] Highlighted Computational Complexity subsubsection")
else:
    print("[!!] MISS: Computational Complexity subsubsection")

# ── 8. Highlight the computation table ────────────────────────────────────────
# Tables cannot be inside mdframed (float restriction) - skip table highlighting
OLD_TAB_COMP = "SKIP_NO_MATCH_INTENTIONAL_TABLE1"
NEW_TAB_COMP = "SKIP_NO_MATCH_INTENTIONAL_TABLE1"

if OLD_TAB_COMP in txt:
    txt = txt.replace(OLD_TAB_COMP, NEW_TAB_COMP, 1)
    print("[OK] Highlighted Computation Table")
else:
    print("[!!] MISS: Computation Table")

# ── 9. Highlight LP vs FT subsubsection + enumerate + table ───────────────────
OLD_LP_FT_START = r"""\subsubsection{Linear Probing vs. Full Fine-Tuning Analysis}
To justify the choice of Linear Probing (frozen S-JEPA backbone) as our primary downstream evaluation method, we compare its performance and characteristics with Full Fine-Tuning reported in the original S-JEPA work \cite{b7}, where all weights of the Context Encoder are updated. The comparative results are summarized in Table \ref{tab:probe_vs_ft}.

Full Fine-Tuning yields slightly higher accuracy (+2.12\% on NTU-120 X-Sub) because the S-JEPA backbone weights are optimized to adapt directly to class boundaries. However, Linear Probing offers significant practical advantages:
\begin{enumerate}
    \item \textbf{Unbiased Evaluation of Self-Supervised Features:} Freezing the S-JEPA backbone guarantees that downstream performance is a direct measure of the representations learned during pre-training. It proves the backbone has captured intrinsic skeleton kinematics (e.g., bone structures, joint dynamics) that are linearly separable.
    \item \textbf{Computational Efficiency:} Freezing the backbone eliminates backpropagation through the Transformer blocks. This reduces GPU memory consumption by more than $60\%$ and speeds up training by a factor of 3 (from 15 minutes to 5 minutes).
    \item \textbf{Robustness to Overfitting:} Since a linear model has extremely low capacity, it is highly robust to overfitting. This is critical when downstream labeled data is small or imbalanced.
    \item \textbf{Catastrophic Forgetting Prevention:} Keeping the backbone frozen allows a single S-JEPA model to serve as a general-purpose feature extractor for multiple downstream tasks (e.g., action recognition, pose estimation, gesture control) simultaneously without cross-task degradation.
\end{enumerate}"""

NEW_LP_FT_START = r"""\begin{newcontent}
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

if OLD_LP_FT_START in txt:
    txt = txt.replace(OLD_LP_FT_START, NEW_LP_FT_START, 1)
    print("[OK] Highlighted LP vs FT subsubsection")
else:
    print("[!!] MISS: LP vs FT subsubsection")

# ── 10. Highlight the LP vs FT table ─────────────────────────────────────────
# Tables cannot be inside mdframed (float restriction) - skip table highlighting
OLD_TAB_LPFT = "SKIP_NO_MATCH_INTENTIONAL_TABLE2"
NEW_TAB_LPFT = "SKIP_NO_MATCH_INTENTIONAL_TABLE2"

if OLD_TAB_LPFT in txt:
    txt = txt.replace(OLD_TAB_LPFT, NEW_TAB_LPFT, 1)
    print("[OK] Highlighted LP vs FT Table")
else:
    print("[!!] MISS: LP vs FT Table")

# ── Write output ──────────────────────────────────────────────────────────────
with open(DST, 'w', encoding='utf-8') as f:
    f.write(txt)

print(f"\nDone! Saved to {DST}")
print(f"File size: {len(txt)} bytes")
