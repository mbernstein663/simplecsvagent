from pathlib import Path
from datetime import datetime
import json
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class LLMCSVCleaningAgent:
    def __init__(self, csv_path, output_dir="agent_output"):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.log_path = self.output_dir / "llm_agent_actions.md"
        self.cleaned_path = self.output_dir / f"cleaned_{self.csv_path.name}"

        self.df = None
        self.profile = None
        self.plan = None
        self.actions = []

    def log(self, title, content):
        self.actions.append(f"## {title}\n\n{content}\n")

    def observe(self):
        self.df = pd.read_csv(self.csv_path)

        self.profile = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing_values": self.df.isna().sum().to_dict(),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "sample_rows": self.df.head(5).to_dict(orient="records"),
        }

        self.log("OBSERVE", f"Dataset profile:\n\n```json\n{json.dumps(self.profile, indent=2, default=str)}\n```")

    def decide(self):
        prompt = f"""
You are a CSV cleaning agent.

Given this dataset profile, return a JSON cleaning plan.

Allowed actions:
- standardize_column_names
- drop_duplicate_rows
- drop_unnamed_columns
- fill_numeric_missing_with_median
- fill_text_missing_with_unknown
- drop_constant_columns

Dataset profile:
{json.dumps(self.profile, indent=2, default=str)}

Return ONLY valid JSON in this format:

{{
  "summary": "short explanation of what seems wrong",
  "actions": [
    {{
      "action": "standardize_column_names",
      "reason": "why this action is useful"
    }}
  ],
  "warnings": [
    "things the user should inspect manually"
  ]
}}
"""

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )

        text = response.output_text
        self.plan = json.loads(text)

        self.log("DECIDE", f"LLM cleaning plan:\n\n```json\n{json.dumps(self.plan, indent=2)}\n```")

    def act(self):
        df = self.df.copy()
        performed = []

        action_names = [item["action"] for item in self.plan["actions"]]

        if "standardize_column_names" in action_names:
            df.columns = (
                df.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_", regex=False)
                .str.replace("-", "_", regex=False)
            )
            performed.append("standardized column names")

        if "drop_unnamed_columns" in action_names:
            unnamed_cols = [col for col in df.columns if "unnamed" in col.lower()]
            df = df.drop(columns=unnamed_cols, errors="ignore")
            performed.append(f"dropped unnamed columns: {unnamed_cols}")

        if "drop_duplicate_rows" in action_names:
            before = len(df)
            df = df.drop_duplicates()
            performed.append(f"removed duplicate rows: {before - len(df)}")

        if "drop_constant_columns" in action_names:
            constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
            df = df.drop(columns=constant_cols, errors="ignore")
            performed.append(f"dropped constant columns: {constant_cols}")

        if "fill_numeric_missing_with_median" in action_names:
            numeric_cols = df.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
            performed.append("filled numeric missing values with median")

        if "fill_text_missing_with_unknown" in action_names:
            text_cols = df.select_dtypes(include="object").columns
            for col in text_cols:
                df[col] = df[col].fillna("Unknown")
            performed.append("filled text missing values with 'Unknown'")

        self.df = df
        self.df.to_csv(self.cleaned_path, index=False)

        self.log("ACT", "\n".join(f"- {item}" for item in performed))

    def report(self):
        final_report = {
            "final_shape": self.df.shape,
            "remaining_missing_values": int(self.df.isna().sum().sum()),
            "cleaned_file": str(self.cleaned_path),
            "llm_summary": self.plan.get("summary"),
            "warnings": self.plan.get("warnings", []),
        }

        self.log("REPORT", f"Final report:\n\n```json\n{json.dumps(final_report, indent=2, default=str)}\n```")

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("# LLM CSV Cleaning Agent Log\n\n")
            f.write(f"Run time: `{datetime.now()}`\n\n")
            f.write(f"Input file: `{self.csv_path}`\n\n")
            f.write("---\n\n")
            f.write("\n\n---\n\n".join(self.actions))

    def run(self):
        self.observe()
        self.decide()
        self.act()
        self.report()

        print("Done.")
        print(f"Cleaned CSV: {self.cleaned_path}")
        print(f"Agent log: {self.log_path}")


if __name__ == "__main__":
    agent = LLMCSVCleaningAgent("your_file.csv")
    agent.run()
