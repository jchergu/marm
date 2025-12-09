import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def summarize_arm_results(itemsets_csv: str, rules_csv: str, output_dir: Path):
    print("\n[ARM SUMMARY] Loading FP-Growth results...")

    itemsets = pd.read_csv(itemsets_csv)
    rules = pd.read_csv(rules_csv)

    print(f"[ARM SUMMARY] Loaded {len(itemsets)} itemsets and {len(rules)} rules.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Itemsets Overview
    print("\n[ARM SUMMARY] Top 10 frequent itemsets:")
    top_itemsets = itemsets.sort_values("support", ascending=False).head(10)
    print(top_itemsets[["support", "itemset_str"]])

    # Plot support distribution
    plt.figure(figsize=(10, 4))
    sns.histplot(itemsets["support"], bins=40)
    plt.title("Distribution of Itemset Supports")
    plt.xlabel("Support")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "itemsets_support_distribution.png")
    plt.close()

    # 2. Rules Overview
    print("\n[ARM SUMMARY] Top 10 rules by lift:")
    top_rules = rules.sort_values("lift", ascending=False).head(10)
    print(top_rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]])

    # Scatter: support vs confidence
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
    plt.savefig(output_dir / "rules_support_confidence.png")
    plt.close()

    # 3. Simplification of rules
    def simplify_rule(row):
        return f"{row['antecedents_str']}  →  {row['consequents_str']} " \
               f"(conf={row['confidence']:.2f}, lift={row['lift']:.2f})"

    rules["simple_rule"] = rules.apply(simplify_rule, axis=1)

    simple_rules_path = output_dir / "rules_simplified.txt"
    with open(simple_rules_path, "w") as f:
        for r in rules.sort_values("lift", ascending=False)["simple_rule"]:
            f.write(r + "\n")

    print(f"\n[ARM SUMMARY] Wrote simplified rules to: {simple_rules_path}")


    # 4. Highlight interesting patterns automatically
    interesting = rules[
        (rules["lift"] > 1.2) &
        (rules["confidence"] > 0.6) &
        (rules["support"] > 0.05)
    ].sort_values("lift", ascending=False)

    print("\n[ARM SUMMARY] Interesting strong rules (lift>1.2, conf>0.6, support>0.05):")
    if len(interesting) == 0:
        print("  None found.")
    else:
        print(interesting[["antecedents_str", "consequents_str", "support", "confidence", "lift"]])

    interesting.to_csv(output_dir / "top_interesting_rules.csv", index=False)

    print("\n[ARM SUMMARY] Saved all plots + simplified rules.")
    print("[ARM SUMMARY] Done.\n")
