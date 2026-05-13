"""
Modern Patient Input Form Page
"""

import tkinter as tk
from tkinter import ttk, messagebox

from src.ml.predict import FEATURE_REQUEST_ORDER
from src.gui.styles import COLORS, FONTS, FIELD_LABELS


class PatientInputForm(tk.Frame):
    def __init__(self, parent, controller, on_prediction_success):
        super().__init__(parent, bg=COLORS["bg"])

        self.controller = controller
        self.on_prediction_success = on_prediction_success
        self.entries = {}

        self.grid_columnconfigure(0, weight=1)

        tk.Label(
            self,
            text="Single Patient Screening",
            bg=COLORS["bg"],
            fg=COLORS["text"],
            font=FONTS["page_title"],
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        tk.Label(
            self,
            text="Enter the patient’s demographic and laboratory values.",
            bg=COLORS["bg"],
            fg=COLORS["muted"],
            font=FONTS["body"],
        ).grid(row=1, column=0, sticky="w", pady=(0, 24))

        card = tk.Frame(
            self,
            bg=COLORS["surface"],
            padx=28,
            pady=24,
            bd=0,
            highlightthickness=0,
        )
        card.grid(row=2, column=0, sticky="ew")
        card.grid_columnconfigure(1, weight=1)
        card.grid_columnconfigure(3, weight=1)

        for index, feature in enumerate(FEATURE_REQUEST_ORDER):
            row = index // 2
            col = (index % 2) * 2

            tk.Label(
                card,
                text=FIELD_LABELS.get(feature, feature),
                bg=COLORS["surface"],
                fg=COLORS["text"],
                font=FONTS["body"],
            ).grid(row=row, column=col, sticky="w", padx=(0, 10), pady=10)

            if feature == "Sex":
                var = tk.StringVar(value="m")
                widget = ttk.Combobox(
                    card,
                    textvariable=var,
                    values=["m", "f"],
                    state="readonly",
                    width=24,
                )
                widget.var = var
            else:
                widget = tk.Entry(
                    card,
                    width=28,
                    relief="flat",
                    bd=0,
                    bg="#F9FAFB",
                    fg=COLORS["text"],
                    font=FONTS["body"],
                    insertbackground=COLORS["text"],
                    highlightthickness=1,
                    highlightbackground=COLORS["border"],
                    highlightcolor=COLORS["primary"],
                )

            widget.grid(row=row, column=col + 1, sticky="ew", padx=(0, 28), pady=10, ipady=8)
            self.entries[feature] = widget

        button_frame = tk.Frame(self, bg=COLORS["bg"])
        button_frame.grid(row=3, column=0, sticky="w", pady=24)

        self.primary_button(
            button_frame,
            "Run Prediction",
            self.handle_predict,
        ).pack(side="left", padx=(0, 12))

        self.secondary_button(
            button_frame,
            "Fill Sample",
            self.fill_sample_input,
        ).pack(side="left", padx=(0, 12))

        self.secondary_button(
            button_frame,
            "Clear",
            self.clear_form,
        ).pack(side="left")

    def primary_button(self, parent, text, command):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=COLORS["primary"],
            fg="white",
            activebackground=COLORS["primary_hover"],
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=18,
            pady=10,
            font=FONTS["button"],
            cursor="hand2",
        )

    def secondary_button(self, parent, text, command):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            activebackground="#EEF2FF",
            activeforeground=COLORS["text"],
            relief="flat",
            bd=0,
            padx=18,
            pady=10,
            font=FONTS["button"],
            cursor="hand2",
        )

    def get_form_data(self) -> dict:
        data = {}

        for feature, widget in self.entries.items():
            if feature == "Sex":
                data[feature] = widget.var.get()
            else:
                data[feature] = widget.get().strip()

        return data

    def handle_predict(self):
        data = self.get_form_data()
        response = self.controller.predict(data)

        if not response["success"]:
            messagebox.showerror(
                "Invalid Input",
                "\n".join(response["errors"]),
            )
            return

        self.on_prediction_success(response["result"])

    def clear_form(self):
        for feature, widget in self.entries.items():
            if feature == "Sex":
                widget.var.set("m")
            else:
                widget.delete(0, tk.END)

    def fill_sample_input(self):
        sample = {
            "Age": "45",
            "Sex": "m",
            "ALB": "42",
            "ALP": "85",
            "ALT": "30",
            "AST": "28",
            "BIL": "0.8",
            "CHE": "8.5",
            "CHOL": "4.8",
            "CREA": "0.9",
            "GGT": "35",
            "PROT": "72",
        }

        self.clear_form()

        for feature, value in sample.items():
            widget = self.entries[feature]
            if feature == "Sex":
                widget.var.set(value)
            else:
                widget.insert(0, value)