"""Build publication diagrams with explicit data flow and clock boundaries."""
# TikZ node specifications are kept on one line for visual source editing.
# ruff: noqa: E501

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
PREAMBLE = (
    (FIG / "rev_architecture.tex")
    .read_text(encoding="utf-8")
    .split(r"\begin{document}")[0]
    .replace(r"\usepackage{amsmath}", r"\usepackage{amsmath}\usepackage{graphicx}")
)
PREAMBLE += r"""
\hyphenpenalty=10000\exhyphenpenalty=10000
\tikzset{box/.append style={font=\sffamily\fontsize{8}{10}\selectfont},
note/.append style={font=\sffamily\fontsize{8}{10}\selectfont}}
"""

DIAGRAMS = {
    "publication_history": r"""
\begin{tikzpicture}[x=1cm,y=1cm]
\path[use as bounding box] (0,-.2) rectangle (16.2,5.5);
\node[heading,anchor=west] at (.1,5.2) {Observation history and service response use distinct clocks};
\node[note,anchor=west] at (.1,4.65) {Sensor timestamps: observations admitted up to query anchor $t$};
\fill[pale] (2.3,2.7) rectangle (14.5,4.15);
\node[note,text=event,anchor=east] at (2.1,3.75) {Events};
\node[note,text=rgb,anchor=east] at (2.1,3.1) {RGB};
\foreach \x in {2.5,2.7,3.4,4.1,4.3,5.4,5.8,6.3,7.9,8.1,8.8,9.2,10.4,11.5,12.2,12.5,13.3,14.2}{\draw[event] (\x,3.55)--(\x,3.95);}
\foreach \x in {2.8,4.9,7,9.1,11.2,14.1}{\fill[rgb] (\x,3.1) circle (2pt);}
\draw[arrow] (2.3,2.6)--(15.1,2.6);
\draw[dashed,ink] (14.5,2.45)--(14.5,4.25);
\node[note] at (8.1,2.2) {Up to 650 ms of past support};
\node[note] at (14.5,2.2) {$t$};
\node[note,anchor=west] at (.1,1.55) {Service clock: scheduled arrival to complete host output};
\draw[arrow,fusion] (2.3,.8)--node[above,note,text=fusion]{Response $L=\tau_o-\tau_a$}(14.5,.8);
\node[note] at (2.3,.35) {Arrival $\tau_a$};
\node[note] at (14.5,.35) {Output $\tau_o$};
\node[note] at (8,-.05) {Schematic observations; horizontal distances are not to scale across clocks.};
\end{tikzpicture}
""",
    "publication_architecture": r"""
\begin{tikzpicture}[x=1cm,y=1cm]
\path[use as bounding box] (0,-.25) rectangle (16.2,8.8);
\node[heading,anchor=west] at (.1,8.5) {Inference: nine fitted components, one causal prediction};
\node[note,anchor=west] at (.1,8) {Both branches use supplied object regions and observations available by query time.};
\foreach \y/\col/\stream/\na/\nb/\np/\nc in {5.35/event/Events/1/2/3/4,1.75/rgb/RGB/5/6/7/8}{
 \draw[draw=\col!25,fill=\col!3,rounded corners=3pt] (0,\y-1.5) rectangle (11.55,\y+1.5);
 \node[box,draw=\col!60,text width=1.1cm] (in\na) at (.85,\y) {\stream\\inputs};
 \node[box,draw=\col!60,text width=1.7cm] (a\na) at (3,\y+.65) {\textbf{\na} Transport\\producer};
 \node[box,draw=\col!60,text width=1.7cm] (b\na) at (3,\y-.8) {\textbf{\nb} Multi-scale\\producer};
 \node[box,draw=\col!60,text width=1.65cm] (p\na) at (5.55,\y+.65) {\textbf{\np} Pairwise\\phase readout};
 \node[box,draw=\col!60,text width=1.55cm,minimum height=16mm] (f\na) at (8,\y) {17 fields\\+ clocks\\+ mask};
 \node[box,draw=\col!60,text width=1.65cm,minimum height=16mm] (c\na) at (10.4,\y) {\textbf{\nc} Temporal\\refiner\\causal history};
 \draw[arrow,\col] (in\na.east)--(a\na.west);
 \draw[arrow,\col] (in\na.east)--(b\na.west);
 \draw[arrow,\col] (a\na.east)--(p\na.west);
 \draw[arrow,\col] (p\na.east)--(f\na.west);
 \draw[arrow,\col] (b\na.east)--(f\na.south west);
 \draw[arrow,\col] (a\na.north)--++(0,.52)-|node[pos=.45,above,note,text=\col]{producer phase}(f\na.north);
 \draw[arrow,\col] (f\na.east)--(c\na.west);
}
\node[fb,text width=2.25cm,minimum height=19mm] (fusion) at (13.8,3.55) {\textbf{9} Residual fusion\\20 named fields\\RGB anchor bypass};
\draw[arrow,event] (c1.east)--(12,5.35)|-([yshift=4mm]fusion.west);
\draw[arrow,rgb] (c5.east)--(12,1.75)|-([yshift=-4mm]fusion.west);
\node[fb,text width=2.25cm] (out) at (13.8,.9) {Signed TTC\\Phase bounds};
\draw[arrow,fusion] (fusion.south)--(out.north);
\node[note,text width=3.3cm] at (13.75,6.5) {Arrows carry model outputs.\\The RGB bypass carries\\location and width logits.};
\end{tikzpicture}
""",
    "publication_oof": r"""
\begin{tikzpicture}[x=1cm,y=1cm]
\path[use as bounding box] (0,-.2) rectangle (16.2,8.9);
\node[heading,anchor=west] at (.1,8.65) {A. Generate held-out features: repeat with five disjoint-group rotations};
\node[eb,text width=3.7cm,minimum height=17mm] (p) at (2.15,7.15) {\textbf{24 producer groups}\\Fit 4 producers\\and 2 pairwise readouts};
\node[rb,text width=3.7cm,minimum height=17mm] (c) at (8.05,7.15) {\textbf{8 context groups}\\Fit 2 context refiners\\and normalizers};
\node[fb,text width=3.7cm,minimum height=17mm] (v) at (13.95,7.15) {\textbf{8 held-out groups}\\Generate 17-field inputs\\and 20-field outputs};
\draw[arrow] (p.east)--node[above,note,text width=1.7cm]{frozen\\predictors}(c.west);
\draw[arrow] (c.east)--node[above,note,text width=1.7cm]{frozen\\fold chain}(v.west);
\node[note,text width=15.7cm] at (8.05,5.7) {Arrows transfer fitted predictors, not training sequences. Held-out targets enter only after feature generation.};
\draw[rule] (.1,5.25)--(16,5.25);
\node[heading,anchor=west] at (.1,4.85) {B. Fit the deployment components from three distinct training inputs};
\node[eb,text width=5.5cm] (data) at (3.2,3.8) {All 40 training groups};
\node[eb,text width=5.2cm] (prod) at (10.5,3.8) {4 final producers + 2 pairwise readouts};
\node[rb,text width=5.5cm] (seventeen) at (3.2,2.55) {Pooled held-out 17-field context inputs};
\node[rb,text width=5.2cm] (ctx) at (10.5,2.55) {2 deployment context refiners};
\node[fb,text width=5.5cm] (twenty) at (3.2,1.3) {Pooled held-out 20-field fusion inputs\\generated by fold-local context refiners};
\node[fb,text width=5.2cm] (fuse) at (10.5,1.3) {1 final fusion head};
\draw[arrow] (data)--node[above,note]{fit}(prod);
\draw[arrow] (seventeen)--node[above,note]{fit}(ctx);
\draw[arrow] (twenty)--node[above,note]{fit}(fuse);
\draw[decorate,decoration={brace,amplitude=5pt},fusion] (13.45,4.3)--(13.45,.7);
\node[note,text width=2cm,text=fusion] at (14.9,2.55) {Nine fitted\\components\\for inference};
\node[note,text width=15.8cm] at (8.05,.05) {Deployment context refiners do not generate the fusion head's training inputs.};
\end{tikzpicture}
""",
    "publication_fusion": r"""
\begin{tikzpicture}[x=1cm,y=1cm]
\path[use as bounding box] (0,-.15) rectangle (16.2,6.3);
\node[heading,anchor=west] at (.1,6.05) {Fuse named measurements; keep an explicit RGB anchor};
\node[rb,text width=2.1cm,minimum height=12mm] (r) at (1.6,4.1) {RGB context\\refiner};
\node[eb,text width=2.1cm,minimum height=12mm] (e) at (1.6,1.6) {Event context\\refiner};
\node[rb,text width=2.7cm,minimum height=14mm] (rd) at (5.25,4.1) {10 named fields\\5 phase, 4 timing\\1 history-support};
\node[eb,text width=2.7cm,minimum height=14mm] (ed) at (5.25,1.6) {10 named fields\\5 phase, 4 timing\\1 history-support};
\node[fb,text width=2.25cm,minimum height=15mm] (g) at (9.7,2.85) {Scale and concatenate\\$\mathbf z\in\mathbb R^{20}$\\Residual network $g_\theta$};
\node[fb,text width=2.6cm,minimum height=16mm] (out) at (14.1,2.85) {RGB + correction\\Finite TTC mapping\\Phase-interval bounds};
\draw[arrow,rgb] (r)--(rd);
\draw[arrow,event] (e)--(ed);
\draw[arrow,rgb] (rd.east)--(g.north west);
\draw[arrow,event] (ed.east)--(g.south west);
\draw[arrow,fusion] (g.east)--node[above,note]{$\mathbf c\in\mathbb R^3$}(out.west);
\draw[arrow,rgb] (r.north)--(1.6,5.25)--node[above,note,text=rgb]{Direct bypass: RGB raw location $\ell_R$ and width logits $\mathbf w_R$}(14.1,5.25)--(out.north);
\node[note,text width=15.7cm] at (8.05,.35) {Only named fields enter $g_\theta$. Zero final-layer initialization gives zero correction. Widths require calibration.};
\end{tikzpicture}
""",
    "publication_teaser": r"""
\begin{tikzpicture}[x=1cm,y=1cm]
\path[use as bounding box] (0,0) rectangle (16.2,5.1);
\node[heading,anchor=west] at (.1,4.8) {(a) Two views of motion};
\node[heading,anchor=west] at (5.65,4.8) {(b) A common phase interface};
\node[heading,anchor=west] at (11.4,4.8) {(c) Complete-system result};
\node[inner sep=0] at (2.65,3.2) {\includegraphics[width=5.1cm]{evttc_ccrm_poster.jpg}};
\node[note,text width=4.9cm] at (2.65,1.5) {RGB appearance and signed event counts\\Recorded EvTTC observations\\Separate camera coordinates};
\node[rb,text width=1.75cm] (r) at (6.8,3.7) {RGB motion\\Timing / support};
\node[eb,text width=1.75cm] (e) at (9.55,3.7) {Event motion\\Timing / support};
\node[fb,text width=3.8cm] (f) at (8.2,2.2) {20 named features\\Learned residual correction};
\node[fb,text width=3.8cm] (o) at (8.2,.7) {RGB anchor + correction\\$\to$ signed TTC};
\draw[arrow,rgb] (r.south)--(f.north west);
\draw[arrow,event] (e.south)--(f.north east);
\draw[arrow,fusion] (f.south)--(o.north);
\draw[arrow,rgb] (r.west)--(5.65,3.7)|-(o.west);
\node[note,text=rgb,rotate=90] at (5.48,2.1) {anchor};
\node[heading,text=fusion,align=center] at (13.7,3.7) {\OfficialPhasePublishedOverallMiD{} vs. \OfficialGarlOverallMiD{}};
\node[note,text width=4.3cm] at (13.7,2.95) {Overall MiD: PHASE-TTC / Garl-TTC\\\OfficialOverallRelativeGain{} lower error};
\node[box,text width=3.8cm] at (13.7,1.75) {Rank 1 of 3 displayed entries\\0\% scorer-defined failures};
\node[note,text width=4.3cm] at (13.7,.5) {GarlTTC Test, 28 Oct 2026\\Complete-system comparison};
\draw[rule] (5.25,.15)--(5.25,4.5);
\draw[rule] (11.1,.15)--(11.1,4.5);
\end{tikzpicture}
""",
}


def main() -> None:
    """Compile new figures without overwriting retained historical assets."""
    for name, body in DIAGRAMS.items():
        (FIG / f"{name}.tex").write_text(
            PREAMBLE
            + "\\input{../tables/official_macros.tex}\n\\begin{document}\n"
            + body
            + "\\end{document}\n",
            encoding="utf-8",
        )
    # Preserve the valid schematic but clarify the exact second-clock boundary.
    clock = (FIG / "rev_runtime_clock.tex").read_text(encoding="utf-8")
    clock = clock.replace("RAW data\\\\available", "RAW acquired,\\\\copied and sealed")
    clock = clock.replace("Secondary: RAW availability", "Secondary: payload sealed")
    (FIG / "publication_runtime_clock.tex").write_text(clock, encoding="utf-8")
    runtime = (FIG / "rev_runtime.tex").read_text(encoding="utf-8")
    runtime = runtime.replace("RAW-availability response", "Payload-sealed response")
    (FIG / "publication_runtime.tex").write_text(runtime, encoding="utf-8")
    for name in [*DIAGRAMS, "publication_runtime_clock", "publication_runtime"]:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"],
            cwd=FIG,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode:
            raise RuntimeError(result.stdout[-4000:])
        log = (FIG / f"{name}.log").read_text(encoding="utf-8")
        if "Overfull" in log or "Missing character" in log:
            raise RuntimeError(f"Figure layout warning: {name}")
        print(f"Built {name}")


if __name__ == "__main__":
    main()
