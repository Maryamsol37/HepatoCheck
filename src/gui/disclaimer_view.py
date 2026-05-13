"""
Modern Medical Disclaimer Page
"""

import tkinter as tk

from src.gui.styles import COLORS, FONTS


class DisclaimerView(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=COLORS["bg"])

        self.grid_columnconfigure(0, weight=1)

        tk.Label(
            self,
            text="Medical Disclaimer",
            bg=COLORS["bg"],
            fg=COLORS["text"],
            font=FONTS["page_title"],
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        tk.Label(
            self,
            text="Important clinical limitations of this application.",
            bg=COLORS["bg"],
            fg=COLORS["muted"],
            font=FONTS["body"],
        ).grid(row=1, column=0, sticky="w", pady=(0, 24))

        card = tk.Frame(
            self,
            bg=COLORS["surface"],
            padx=30,
            pady=26,
            bd=0,
            highlightthickness=0,
        )
        card.grid(row=2, column=0, sticky="ew")

        disclaimer_text = (
            "HepatoCheck is an educational medical data analytics application.\n\n"
            "The prediction shown by this application is based on a machine learning model "
            "trained on a public dataset. It is intended to support screening and learning "
            "purposes only.\n\n"
            "This application does not provide a medical diagnosis.\n\n"
            "A “Possible Risk” result does not confirm liver disease, hepatitis, fibrosis, "
            "or cirrhosis. A “Low Risk” result does not guarantee that the patient is healthy.\n\n"
            "Patients should always consult a licensed clinician for proper diagnosis, medical "
            "interpretation, and treatment decisions.\n\n"
            "Clinical laboratory results must be interpreted together with symptoms, physical "
            "examination, medical history, imaging, and additional confirmatory tests."
        )

        tk.Label(
            card,
            text=disclaimer_text,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI", 12),
            wraplength=820,
            justify="left",
        ).pack(anchor="w")