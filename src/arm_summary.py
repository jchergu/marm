import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def _phase_load_results(itemsets_csv: str, rules_csv: str):
    print("\n[ARM SUMMARY] Loading FP-Growth results...")
    itemsets = pd.read_csv(itemsets_csv)
    rules = pd.read_csv(rules_csv)
    print(f"[ARM SUMMARY] Loaded {len(itemsets)} itemsets and {len(rules)} rules.")
    return itemsets, rules


def _phase_plot_itemset_support(itemsets: pd.DataFrame, output_dir: Path):
    print("[arm summary] Plotting itemset support distribution")
    plt.figure(figsize=(10, 4))
    sns.histplot(itemsets["support"], bins=40)
    plt.title("Distribution of Itemset Supports")
    plt.xlabel("Support")
    plt.ylabel("Count")
    plt.tight_layout()
    out = output_dir / "itemsets_support_distribution.png"
    plt.savefig(out)
    plt.close()
    return out


def _phase_plot_rules_scatter(rules: pd.DataFrame, output_dir: Path):
    print("[arm summary] Plotting rules support vs confidence")
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=rules,
        x="support",
        y="confidence",
        hue="lift",
        palette="viridis",
        alpha=0.7,
    )
    plt.title("Rules: Support vs Confidence (color = lift)")
    plt.tight_layout()
    out = output_dir / "rules_support_confidence.png"
    plt.savefig(out)
    plt.close()
    return out


def _phase_simplify_rules(rules: pd.DataFrame, output_dir: Path):
    print("[arm summary] Simplifying rules to human-readable text")

    def simplify_rule(row):
        return f"{row['antecedents_str']}  →  {row['consequents_str']} " \
               f"(conf={row['confidence']:.2f}, lift={row['lift']:.2f})"

    rules = rules.copy()
    rules["simple_rule"] = rules.apply(simplify_rule, axis=1)
    simple_rules_path = output_dir / "rules_simplified.txt"
    with open(simple_rules_path, "w", encoding="utf-8") as f:
        for r in rules.sort_values("lift", ascending=False)["simple_rule"]:
            f.write(r + "\n")
    return simple_rules_path, rules


def _phase_find_interesting_rules(rules: pd.DataFrame, output_dir: Path,
                                  lift_thresh=1.2, conf_thresh=0.6, supp_thresh=0.05):
    print("[arm summary] Selecting interesting rules using thresholds")
    interesting = rules[
        (rules["lift"] > lift_thresh) &
        (rules["confidence"] > conf_thresh) &
        (rules["support"] > supp_thresh)
    ].sort_values("lift", ascending=False)
    out_path = output_dir / "top_interesting_rules.csv"
    interesting.to_csv(out_path, index=False)
    return out_path, interesting


def summarize_arm_results(itemsets_csv: str, rules_csv: str, output_dir: Path):
    itemsets, rules = _phase_load_results(itemsets_csv, rules_csv)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[arm summary] Top 10 frequent itemsets:")
    top_itemsets = itemsets.sort_values("support", ascending=False).head(10)
    print(top_itemsets[["support", "itemset_str"]])

    _phase_plot_itemset_support(itemsets, output_dir)

    print("\n[arm summary] Top 10 rules by lift:")
    top_rules = rules.sort_values("lift", ascending=False).head(10)
    print(top_rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]])

    _phase_plot_rules_scatter(rules, output_dir)

    simple_path, rules = _phase_simplify_rules(rules, output_dir)
    print(f"\n[arm summary] Wrote simplified rules to: {simple_path}")

    interesting_path, interesting = _phase_find_interesting_rules(rules, output_dir)
    print("\n[arm summary] Interesting strong rules (lift>1.2, conf>0.6, support>0.05):")
    if interesting.empty:
        print("  None found.")
    else:
        print(interesting[["antecedents_str", "consequents_str", "support", "confidence", "lift"]])

    print("\n[arm summary] Saved all plots + simplified rules.")
    print("[arm summary] Done.\n")
