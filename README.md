# RJ-PINNs
I introduce RJ-PINNs: A breakthrough PINNs framework using Jacobian-based least-squares (TRF) to directly minimize residuals without traditional loss functions. First method to eliminate gradient optimizers in PINNs, offering unmatched robustness for inverse,direct PDE problems 


For more information, please refer to the following:(https://github.com/dadesso17/RJ-PINNs/)

\section{Key Differences}
\begin{table}[H]
\centering
\small % Use a smaller font size for the table
\begin{tabular}{|p{3cm}|p{5cm}|p{5cm}|} % Adjust column widths as needed
\hline 
\textbf{Aspect} & \textbf{Traditional PINNs} & \textbf{RJ-PINNs} \\
\hline 
\hline 
Objective & Minimize a loss function $\mathcal{L}(\theta)$ & Minimize the residuals $R(\theta)$ \\
\hline 
Gradient & Compute $\nabla\mathcal{L}(\theta)$ & Compute $\nabla R(\theta)$ \\
\hline 
Optimization & Use gradient-based optimizers (e.g., Adam, L-BFGS) & Use least-squares optimizer (e.g., TRF) \\
\hline 
Implementation & Define a loss function and its gradient & Define residuals and their Jacobian \\
\hline 
Convergence & Not guaranteed & Robust convergence \\
\hline 
\end{tabular}
\caption{Comparison between Traditional PINNs and RJ-PINNs}
\label{tab:key_differences}
\end{table}


