"""
Modern Batch CSV Upload Page
"""

import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from src.ml.predict import FEATURE_REQUEST_ORDER
from src.gui.styles import COLORS, FONTS


class BatchUploadView(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=COLORS["bg"])

        self.controller = controller
        self.batch_results = []

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        tk.Label(
            self,
            text="Batch CSV Prediction",
            bg=COLORS["bg"],
            fg=COLORS["text"],
            font=FONTS["page_title"],
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        tk.Label(
            self,
            text="Upload a CSV file containing multiple patient records.",
            bg=COLORS["bg"],
            fg=COLORS["muted"],
            font=FONTS["body"],
        ).grid(row=1, column=0, sticky="w", pady=(0, 10))

        required_text = "Required columns: " + ", ".join(FEATURE_REQUEST_ORDER)

        tk.Label(
            self,
            text=required_text,
            bg=COLORS["bg"],
            fg=COLORS["muted"],
            font=FONTS["small"],
            wraplength=850,
            justify="left",
        ).grid(row=2, column=0, sticky="w", pady=(0, 20))

        button_frame = tk.Frame(self, bg=COLORS["bg"])
        button_frame.grid(row=3, column=0, sticky="w", pady=(0, 18))

        self.button(
            button_frame,
            "Upload CSV",
            self.upload_csv,
            primary=True,
        ).pack(side="left", padx=(0, 12))

        self.button(
            button_frame,
            "Export Results",
            self.export_csv,
        ).pack(side="left", padx=(0, 12))

        self.button(
            button_frame,
            "Clear",
            self.clear_table,
        ).pack(side="left")

        table_card = tk.Frame(
            self,
            bg=COLORS["surface"],
            padx=18,
            pady=18,
            bd=0,
            highlightthickness=0,
        )
        table_card.grid(row=4, column=0, sticky="nsew")
        table_card.grid_columnconfigure(0, weight=1)
        table_card.grid_rowconfigure(0, weight=1)

        columns = [
            "Age", "Sex", "ALB", "ALP", "ALT", "AST", "BIL",
            "CHE", "CHOL", "CREA", "GGT", "PROT",
            "prediction_label", "confidence",
            "probability_low_risk", "probability_possible_risk",
        ]

        self.tree = ttk.Treeview(
            table_card,
            columns=columns,
            show="headings",
            height=15,
        )

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor="center")

        y_scroll = ttk.Scrollbar(table_card, orient="vertical", command=self.tree.yview)
        x_scroll = ttk.Scrollbar(table_card, orient="horizontal", command=self.tree.xview)

        self.tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

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

    def upload_csv(self):
        filepath = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV Files", "*.csv")],
        )

        if not filepath:
            return

        try:
            with open(filepath, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                records = list(reader)

            if not records:
                messagebox.showerror("Empty File", "The selected CSV file is empty.")
                return

            missing_columns = [
                col for col in FEATURE_REQUEST_ORDER
                if col not in records[0]
            ]

            if missing_columns:
                messagebox.showerror(
                    "Missing Columns",
                    "Missing required columns:\n" + "\n".join(missing_columns),
                )
                return

            results, errors = self.controller.predict_csv_records(records)
            self.batch_results = results
            self.populate_table(results)

            if errors:
                messagebox.showwarning(
                    "Some Rows Were Skipped",
                    "\n".join(errors[:10]) +
                    ("\n\nMore errors exist..." if len(errors) > 10 else ""),
                )
            else:
                messagebox.showinfo("Success", "Batch prediction completed successfully.")

        except Exception as e:
            messagebox.showerror("Upload Failed", str(e))

    def populate_table(self, rows: list[dict]):
        self.clear_table()

        for row in rows:
            values = [
                row.get("Age", ""),
                row.get("Sex", ""),
                row.get("ALB", ""),
                row.get("ALP", ""),
                row.get("ALT", ""),
                row.get("AST", ""),
                row.get("BIL", ""),
                row.get("CHE", ""),
                row.get("CHOL", ""),
                row.get("CREA", ""),
                row.get("GGT", ""),
                row.get("PROT", ""),
                row.get("prediction_label", ""),
                row.get("confidence", ""),
                row.get("probability_low_risk", ""),
                row.get("probability_possible_risk", ""),
            ]

            self.tree.insert("", tk.END, values=values)

    def clear_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def export_csv(self):
        if not self.batch_results:
            messagebox.showerror("No Results", "There are no batch results to export.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
        )

        if not filepath:
            return

        try:
            self.controller.export_results_to_csv(self.batch_results, filepath)
            messagebox.showinfo("Exported", "Results exported successfully.")
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))