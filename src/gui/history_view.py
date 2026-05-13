"""
Modern Screening History Page
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from src.gui.styles import COLORS, FONTS


class HistoryView(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=COLORS["bg"])

        self.controller = controller

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        tk.Label(
            self,
            text="Screening History",
            bg=COLORS["bg"],
            fg=COLORS["text"],
            font=FONTS["page_title"],
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        tk.Label(
            self,
            text="Predictions made during the current application session.",
            bg=COLORS["bg"],
            fg=COLORS["muted"],
            font=FONTS["body"],
        ).grid(row=1, column=0, sticky="w", pady=(0, 20))

        button_frame = tk.Frame(self, bg=COLORS["bg"])
        button_frame.grid(row=2, column=0, sticky="w", pady=(0, 18))

        self.button(
            button_frame,
            "Refresh",
            self.refresh,
            primary=True,
        ).pack(side="left", padx=(0, 12))

        self.button(
            button_frame,
            "Export TXT",
            self.export_txt,
        ).pack(side="left")

        table_card = tk.Frame(
            self,
            bg=COLORS["surface"],
            padx=18,
            pady=18,
            bd=0,
            highlightthickness=0,
        )
        table_card.grid(row=3, column=0, sticky="nsew")
        table_card.grid_columnconfigure(0, weight=1)
        table_card.grid_rowconfigure(0, weight=1)

        columns = ["timestamp", "age", "sex", "prediction", "confidence"]

        self.tree = ttk.Treeview(
            table_card,
            columns=columns,
            show="headings",
            height=16,
        )

        headings = {
            "timestamp": "Time",
            "age": "Age",
            "sex": "Sex",
            "prediction": "Prediction",
            "confidence": "Confidence",
        }

        for col in columns:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=150, anchor="center")

        self.tree.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(table_card, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")

    def button(self, parent, text, command, primary=False):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=COLORS["primary"] if primary else COLORS["surface"],
            fg="white" if primary else COLORS["text"],
            activebackground=COLORS["primary_hover"] if primary else "#EEF2FF",
            activeforeground="white" if primary else COLORS["text"],
            relief="flat",
            bd=0,
            padx=18,
            pady=10,
            font=FONTS["button"],
            cursor="hand2",
        )

    def refresh(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        history = self.controller.get_history()

        for item in history:
            patient_input = item["input"]
            result = item["result"]

            self.tree.insert(
                "",
                tk.END,
                values=[
                    item["timestamp"],
                    patient_input.get("Age", ""),
                    patient_input.get("Sex", ""),
                    result.get("prediction_label", ""),
                    result.get("confidence", ""),
                ],
            )

    def export_txt(self):
        filepath = filedialog.asksaveasfilename(
            title="Save History",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")],
        )

        if not filepath:
            return

        try:
            self.controller.export_history_to_txt(filepath)
            messagebox.showinfo("Exported", "History exported successfully.")
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))