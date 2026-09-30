# Energy Consumption Pattern Analysis Using Primary Survey Data
# GitHub-ready analysis script for the CEP project.

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cleaned_energy_consumption_survey_data.csv"
OUT = ROOT / "visualizations"
OUT.mkdir(exist_ok=True)

# Load anonymized public dataset
raw = pd.read_csv(DATA)

# Rename fields for easier analysis
rename = {
    "1. What type of house do you live in?": "House Type",
    "2. How many people live in your household?": "Household Size",
    "3. What is your approximate monthly household electricity bill?": "Electricity Bill",
    "4. Which electrical appliances are regularly used in your household?": "Appliances",
    "5. Approximately how many hours per day are electrical appliances used in your household?": "Daily Appliance Usage",
    "6. Which factor has the greatest influence on your household's electricity consumption?": "Consumption Factor",
    "7. During which season does your household usually have the highest electricity consumption?": "Season",
    "8. How often are electrical appliances used during peak evening hours (6 PM–10 PM)?": "Peak Evening Usage",
    "9. Does your household use any renewable energy source, such as solar energy?": "Renewable Energy",
    "10. How willing would your household be to change its electricity-use habits to reduce energy consumption and monthly bills?": "Willingness to Change",
}
df = raw.rename(columns=rename).copy()

print("Rows:", len(df))
print("Columns:", list(df.columns))

# ---------- Helper ----------
def chisq(row_field, col_field):
    table = pd.crosstab(df[row_field], df[col_field])
    chi2, p, dof, expected = chi2_contingency(table)
    print(f"\n{row_field} vs {col_field}")
    print("Chi-square =", round(chi2, 4))
    print("p-value    =", p)
    return table, chi2, p

# ---------- Hypothesis tests ----------
table1, chi1, p1 = chisq("Household Size", "Electricity Bill")
table2, chi2, p2 = chisq("Peak Evening Usage", "Electricity Bill")
table3, chi3, p3 = chisq("Renewable Energy", "Willingness to Change")

# ---------- Figure 1: Household Size vs Electricity Bill ----------
plt.figure(figsize=(10, 6))
sns.heatmap(table1, annot=True, fmt="g")
plt.title("Household Size vs Monthly Electricity Bill")
plt.xlabel("Monthly Electricity Bill")
plt.ylabel("Household Size")
plt.tight_layout()
plt.savefig(OUT / "01_household_size_vs_bill_heatmap.png", dpi=300)
plt.close()

# ---------- Figure 2: Daily Appliance Usage vs Electricity Bill ----------
ct = pd.crosstab(df["Daily Appliance Usage"], df["Electricity Bill"])
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
ct_pct.plot(kind="bar", stacked=True, figsize=(10, 6))
plt.title("Electricity Bill Distribution by Daily Appliance-Use Duration")
plt.xlabel("Daily Appliance Usage")
plt.ylabel("Percentage of Households")
plt.legend(title="Electricity Bill", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUT / "02_daily_usage_vs_bill_stacked.png", dpi=300)
plt.close()

# ---------- Figure 3: Peak Evening Usage vs Electricity Bill ----------
plt.figure(figsize=(10, 6))
sns.heatmap(table2, annot=True, fmt="g")
plt.title("Peak Evening Appliance Usage vs Monthly Electricity Bill")
plt.xlabel("Monthly Electricity Bill")
plt.ylabel("Peak Evening Usage (6 PM–10 PM)")
plt.tight_layout()
plt.savefig(OUT / "03_peak_evening_vs_bill_heatmap.png", dpi=300)
plt.close()

# ---------- Figure 4: Renewable Energy vs Willingness ----------
ct2 = pd.crosstab(df["Renewable Energy"], df["Willingness to Change"])
ct2_pct = ct2.div(ct2.sum(axis=1), axis=0) * 100
ct2_pct.plot(kind="bar", stacked=True, figsize=(10, 6))
plt.title("Renewable Energy Use vs Willingness to Change Electricity Habits")
plt.xlabel("Renewable Energy Use")
plt.ylabel("Percentage of Households")
plt.legend(title="Willingness", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUT / "04_renewable_vs_willingness_stacked.png", dpi=300)
plt.close()

# ---------- Figure 5: Appliance Adoption ----------
appliance_counts = {}
for value in df["Appliances"].dropna():
    for item in str(value).split(","):
        item = item.strip()
        if item:
            appliance_counts[item] = appliance_counts.get(item, 0) + 1
app = pd.Series(appliance_counts).sort_values(ascending=True)
plt.figure(figsize=(10, 6))
app.plot(kind="barh")
plt.title("Regularly Used Electrical Appliances")
plt.xlabel("Number of Households")
plt.ylabel("Appliance")
plt.tight_layout()
plt.savefig(OUT / "05_appliance_adoption_horizontal_bar.png", dpi=300)
plt.close()

# ---------- Figure 6: Household Size vs Daily Usage ----------
table4, chi4, p4 = chisq("Household Size", "Daily Appliance Usage")
plt.figure(figsize=(10, 6))
sns.heatmap(table4, annot=True, fmt="g")
plt.title("Household Size vs Daily Appliance-Use Duration")
plt.xlabel("Daily Appliance Usage")
plt.ylabel("Household Size")
plt.tight_layout()
plt.savefig(OUT / "06_household_size_vs_usage_heatmap.png", dpi=300)
plt.close()

print("\nAnalysis complete. Figures saved to:", OUT)
