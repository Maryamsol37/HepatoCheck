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
        recommendation = result.get("recommendation", "No recommendation available.")

        label_text = str(label).strip()
        normalized_label = label_text.lower()
        is_low_risk = "low" in normalized_label and "risk" in normalized_label

        if is_low_risk:
            self.badge.configure(
                text=label_text,
                bg=COLORS["low_risk_bg"],
                fg=COLORS["low_risk_text"],
            )
        else:
            self.badge.configure(
                text=label_text,
                bg=COLORS["risk_bg"],
                fg=COLORS["risk_text"],
            )

        self.confidence_label.configure(
            text=f"Model confidence: {confidence}"
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

        # Handle different types of top_features
        feature_lines = []

        if top_features:
            if isinstance(top_features, dict):
                # If it's a dictionary with feature: importance pairs
                for feature, importance in top_features.items():
                    if isinstance(importance, (int, float)):
                        feature_lines.append(f"• {feature}: {importance:.4f}")
                    else:
                        feature_lines.append(f"• {feature}: {importance}")
            
            elif isinstance(top_features, list):
                # If it's a list of feature names (no importance values)
                for feature in top_features:
                    feature_lines.append(f"• {feature}")
            
            else:
                feature_lines = ["Feature importance data is in an unexpected format."]
        else:
            feature_lines = ["No feature importance data available."]

        self.features_box.content_label.configure(
            text="\n".join(feature_lines)
        )