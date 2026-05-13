"""
Modern Result Display Page
"""

import tkinter as tk

from src.gui.styles import COLORS, FONTS


class ResultView(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=COLORS["bg"])

        self.grid_columnconfigure(0, weight=1)

        tk.Label(
            self,
            text="Prediction Result",
            bg=COLORS["bg"],
            fg=COLORS["text"],
            font=FONTS["page_title"],
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        tk.Label(
            self,
            text="Review the model output, probabilities, and flagged laboratory values.",
            bg=COLORS["bg"],
            fg=COLORS["muted"],
            font=FONTS["body"],
        ).grid(row=1, column=0, sticky="w", pady=(0, 24))

        self.card = tk.Frame(
            self,
            bg=COLORS["surface"],
            padx=30,
            pady=26,
            bd=0,
            highlightthickness=0,
        )
        self.card.grid(row=2, column=0, sticky="ew")
        self.card.grid_columnconfigure(0, weight=1)

        self.badge = tk.Label(
            self.card,
            text="No prediction yet",
            bg="#E5E7EB",
            fg=COLORS["text"],
            font=FONTS["result"],
            padx=18,
            pady=10,
        )
        self.badge.grid(row=0, column=0, sticky="w", pady=(0, 18))

        self.confidence_label = tk.Label(
            self.card,
            text="",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=FONTS["section_title"],
        )
        self.confidence_label.grid(row=1, column=0, sticky="w", pady=(0, 8))

        self.probability_label = tk.Label(
            self.card,
            text="",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=FONTS["body"],
        )
        self.probability_label.grid(row=2, column=0, sticky="w", pady=(0, 20))

        self.recommendation_label = tk.Label(
            self.card,
            text="",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=FONTS["body"],
            wraplength=780,
            justify="left",
        )
        self.recommendation_label.grid(row=3, column=0, sticky="w", pady=(0, 22))

        info_grid = tk.Frame(self.card, bg=COLORS["surface"])
        info_grid.grid(row=4, column=0, sticky="ew")
        info_grid.grid_columnconfigure(0, weight=1)
        info_grid.grid_columnconfigure(1, weight=1)

        self.flags_box = self.create_info_box(info_grid, "Abnormal Lab Flags")
        self.flags_box.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        self.features_box = self.create_info_box(info_grid, "Top Important Features")
        self.features_box.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

    def create_info_box(self, parent, title):
        box = tk.Frame(
            parent,
            bg="#F9FAFB",
            padx=18,
            pady=16,
            bd=0,
            highlightthickness=0,
        )

        tk.Label(
            box,
            text=title,
            bg="#F9FAFB",
            fg=COLORS["text"],
            font=FONTS["section_title"],
        ).pack(anchor="w", pady=(0, 10))

        content = tk.Label(
            box,
            text="",
            bg="#F9FAFB",
            fg=COLORS["muted"],
            font=FONTS["body"],
            justify="left",
            anchor="nw",
        )
        content.pack(anchor="w", fill="both", expand=True)

        box.content_label = content
        return box

    def show_result(self, result: dict):
        label = result.get("prediction_label", "Unknown")
        confidence = result.get("confidence", 0)
        low_prob = result.get("probability_low_risk", "N/A")
        risk_prob = result.get("probability_possible_risk", "N/A")
        recommendation = result.get("recommendation", "No recommendation available.")

        if label == "Low Risk":
            self.badge.configure(
                text="Low Risk",
                bg=COLORS["low_risk_bg"],
                fg=COLORS["low_risk_text"],
            )
        else:
            self.badge.configure(
                text="Possible Risk",
                bg=COLORS["risk_bg"],
                fg=COLORS["risk_text"],
            )

        self.confidence_label.configure(
            text=f"Model confidence: {confidence}"
        )

        self.probability_label.configure(
            text=f"Low risk probability: {low_prob}    |    Possible risk probability: {risk_prob}"
        )

        self.recommendation_label.configure(
            text=f"Recommendation: {recommendation}"
        )

        abnormal_flags = result.get("abnormal_flags", {})
        abnormal_lines = []

        for feature, is_abnormal in abnormal_flags.items():
            if is_abnormal:
                abnormal_lines.append(f"• {feature}: Abnormal")

        if not abnormal_lines:
            abnormal_lines.append("No abnormal lab values were flagged.")

        self.flags_box.content_label.configure(
            text="\n".join(abnormal_lines)
        )

        top_features = result.get("top_features") or result.get("key_markers") or {}

        if top_features:
            feature_lines = [
                f"• {feature}: {round(float(importance), 4) if isinstance(importance, (int, float)) else importance}"
                for feature, importance in top_features.items()
            ]
        else:
            feature_lines = ["Feature importance is not available."]

        self.features_box.content_label.configure(
            text="\n".join(feature_lines)
        )