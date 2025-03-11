import plotly.graph_objects as go
import numpy as np
from scipy import stats

def plot_rules_plotly(rules, preds, baselines, weights, max_rulelen=None,
                      other_preds=None, preds_distr=None, b_box_pred=None,
                      round_digits=3):
    """
    A Plotly-based visualization tool for BELLATREX, a local random forest explainability toolbox.
    """
    assert len(rules) == len(preds) == len(baselines) == len(weights), "All input lists must have the same length."
    
    # max_rulelen is the maximum length (decision tree depth) of the rules to be displayed
    max_rulelen = min(max_rulelen or max(len(rule) for rule in rules), max(len(rule) for rule in rules))
    # nrules is the number of rules selected by Bellatrex
    nrules = len(rules) 

    # Truncate rules and predictions if they exceed max_rulelen
    for i in range(nrules):
        if len(rules[i]) > max_rulelen:
            omitted = len(rules[i]) - max_rulelen + 1
            rules[i][max_rulelen-1] = f"+{omitted} other rule splits"
        rules[i] = rules[i][:max_rulelen]
        preds[i] = preds[i][:max_rulelen]

    maxdev = max(max(abs(np.array(pred) - baseline)) for pred, baseline in zip(preds, baselines))
    margin = 0.02 * maxdev
    min_x, max_x = min(baselines) - maxdev - margin, max(baselines) + maxdev + margin

    if preds_distr is not None:
        density = stats.gaussian_kde(preds_distr)
        x = np.linspace(min(preds_distr), max(preds_distr), 100)
        min_x, max_x = min(min_x, x[0]), max(max_x, x[-1])

    fig = go.Figure()

    if other_preds:
        for other_pred in other_preds:
            x = [baselines[0]] + other_pred
            y = list(range(len(other_pred)+1))
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                line=dict(color='lightgrey', width=1),
                marker=dict(
                    size=10,
                    symbol=['circle'] + ['arrow'] * len(other_pred),
                    angleref='previous',
                    color='lightgrey'
                ),
                showlegend=False,
                hoverinfo='skip'
            ))

    for i in range(nrules):
        rule, pred, baseline = rules[i], preds[i], baselines[i]
        x = [baseline] + pred
        y = list(range(len(pred)+1))
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers', 
            line=dict(color="lightgrey", width=3),
            marker=dict(
                size=10,
                symbol=['circle'] + ['arrow'] * len(pred),
                angleref='previous',
                color='lightgrey',
            ),
            showlegend=False,
            visible=False,
            hoverinfo='skip'
        ))

        if preds_distr is not None:
            fig.add_trace(go.Scatter(x=x, y=density(x), mode='lines', 
                                     line=dict(color='black', width=1),
                                     showlegend=False, visible=False))
            fig.add_trace(go.Scatter(x=[baseline, pred[-1]], y=[density(baseline), density(pred[-1])],
                                     mode='markers', marker=dict(size=10, color=['gray', pred[-1]],
                                     colorscale='RdYlGn', cmin=baseline-maxdev, cmax=baseline+maxdev),
                                     showlegend=False, visible=False))

    fig.update_yaxes(title_text="Rule depth", range=[max_rulelen+0.5, -1])
    fig.update_xaxes(title_text="Prediction")
    fig.update_layout(
        height=60 * (max_rulelen + 3),
        width=900,
        showlegend=False,
        coloraxis=dict(colorscale='RdYlGn', cmin=-maxdev, cmax=maxdev,
                       colorbar=dict(title="Change w.r.t. baseline", y=0.85, len=0.7)),
    )

    #final_pred = sum(weights[i] * preds[i][-1] for i in range(nrules))
    #ig.add_annotation(x=0.5, y=-0.15, xref="paper", yref="paper", text=f"BELLATREX weighted prediction: {final_pred:.{round_digits}f}",
    #                   showarrow=False, font=dict(size=14))

    #if b_box_pred is not None:
    #    fig.add_annotation(x=0.5, y=-0.2, xref="paper", yref="paper", text=f"Black-box model prediction: {b_box_pred:.{round_digits}f}",
    #                       showarrow=False, font=dict(size=14))

    return fig

def parse(rulesplit):
    """Parses a rulesplit outputted by Bellatrex into a form suitable for visualization."""
    return rulesplit.split("(")[0].strip().replace("≤", "<=").replace("≥", ">=")

def read_rules(file, file_extra=None):
    rules, preds, baselines, weights = [], [], [], []
    with open(file, "r") as f:
        btrex_rules = f.readlines()
    for line in btrex_rules:
        if "RULE WEIGHT" in line:
            weights.append(float(line.split(":")[1].strip("\n").strip(" #")))
        if "Baseline prediction" in line:
            baselines.append(float(line.split(":")[1].strip(" \n")))
            rule, pred = [], []
        if "node" in line:
            fullrule = line.split(":")[1].strip().strip("\n").split("-->")
            index_thresh = max(fullrule[0].find(char) for char in ["=", "<", ">"])
            fullrule[0] = fullrule[0][:index_thresh+8]
            rule.append(fullrule[0])
            pred.append(float(fullrule[1]))
        if "leaf" in line:
            rules.append(rule)
            preds.append(pred)

    other_rules, other_preds, other_baselines = [], [], []
    if file_extra:
        with open(file_extra, "r") as f:
            btrex_extra = f.readlines()
        for line in btrex_extra:
            if "Baseline prediction" in line:
                other_baselines.append(float(line.split(":")[1].strip(" \n")))
                rule, pred = [], []
            if "node" in line:
                fullrule = line.split(":")[1].strip().strip("\n").split("-->")
                index_thresh = max(fullrule[0].find(char) for char in ["=", "<", ">"])
                fullrule[0] = fullrule[0][:index_thresh+8]
                rule.append(fullrule[0])
                pred.append(float(fullrule[1]))
            if "leaf" in line:
                other_rules.append(rule)
                other_preds.append(pred)

    return rules, preds, baselines, weights, other_rules, other_preds, other_baselines or None
