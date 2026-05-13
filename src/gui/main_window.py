"""
Modern Main Tkinter Window for HepatoCheck
"""

import tkinter as tk
from tkinter import ttk
from pathlib import Path
from PIL import Image, ImageTk

from src.app.app_controller import AppController
from src.gui.styles import APP_TITLE, COLORS, FONTS
from src.gui.input_form import PatientInputForm
from src.gui.result_view import ResultView
from src.gui.batch_upload_view import BatchUploadView
from src.gui.history_view import HistoryView
from src.gui.disclaimer_view import DisclaimerView


class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("1180x720")
        self.root.minsize(1050, 650)
        self.root.configure(bg=COLORS["bg"])

        self.controller = AppController()
        self.active_nav_button = None

        self.setup_style()
        self.build_layout()
        self.show_home()

    def setup_style(self):
        style = ttk.Style()

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Treeview",
            background=COLORS["surface"],
            foreground=COLORS["text"],
            fieldbackground=COLORS["surface"],
            rowheight=30,
            borderwidth=0,
            font=FONTS["body"],
        )

        style.configure(
            "Treeview.Heading",
            background=COLORS["bg"],
            foreground=COLORS["muted"],
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
        )

        style.map(
            "Treeview",
            background=[("selected", COLORS["primary"])],
            foreground=[("selected", "white")],
        )

        style.configure(
            "TCombobox",
            padding=6,
            font=FONTS["body"],
        )

    def build_layout(self):
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.sidebar = tk.Frame(
            self.root,
            bg=COLORS["sidebar"],
            width=240,
        )
        self.sidebar.grid(row=0, column=0, sticky="ns")
        self.sidebar.grid_propagate(False)

        self.main_area = tk.Frame(
            self.root,
            bg=COLORS["bg"],
        )
        self.main_area.grid(row=0, column=1, sticky="nsew")
        self.main_area.grid_columnconfigure(0, weight=1)
        self.main_area.grid_rowconfigure(0, weight=1)

        self.content = tk.Frame(
            self.main_area,
            bg=COLORS["bg"],
        )
        self.content.grid(row=0, column=0, sticky="nsew", padx=32, pady=28)
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        self.build_sidebar()

    def build_sidebar(self):
        logo_frame = tk.Frame(self.sidebar, bg=COLORS["sidebar"])
        logo_frame.pack(fill="x", padx=16, pady=(24, 28))

        logo_path = Path(__file__).resolve().parents[2] / "assets" / "logo2.png"

        try:
            logo_image = Image.open(logo_path)

            # Resize logo to fit sidebar width
            logo_image.thumbnail((190, 120), Image.LANCZOS)

            self.logo_photo = ImageTk.PhotoImage(logo_image)

            tk.Label(
                logo_frame,
                image=self.logo_photo,
                bg=COLORS["sidebar"],
                bd=0,
                highlightthickness=0,
            ).pack(anchor="center")

        except Exception:
            # Fallback text if logo is missing or cannot be loaded
            tk.Label(
                logo_frame,
                text="HepatoCheck",
                bg=COLORS["sidebar"],
                fg="white",
                font=FONTS["app_title"],
            ).pack(anchor="w")

            tk.Label(
                logo_frame,
                text="Liver Risk Screening",
                bg=COLORS["sidebar"],
                fg="#CBD5E1",
                font=FONTS["small"],
            ).pack(anchor="w", pady=(4, 0))

        self.nav_buttons = {}

        nav_items = [
            ("Home", self.show_home),
            ("Patient Input", self.show_input_form),
            ("Batch Upload", self.show_batch_upload),
            ("History", self.show_history),
            ("Disclaimer", self.show_disclaimer),
        ]

        for text, command in nav_items:
            btn = tk.Button(
                self.sidebar,
                text=text,
                command=lambda t=text, c=command: self.handle_nav(t, c),
                bg=COLORS["sidebar"],
                fg="#E5E7EB",
                activebackground=COLORS["sidebar_hover"],
                activeforeground="white",
                relief="flat",
                bd=0,
                font=FONTS["button"],
                anchor="w",
                padx=22,
                pady=12,
                cursor="hand2",
            )
            btn.pack(fill="x", padx=12, pady=2)
            self.nav_buttons[text] = btn

        bottom = tk.Frame(self.sidebar, bg=COLORS["sidebar"])
        bottom.pack(side="bottom", fill="x", padx=22, pady=24)

        tk.Label(
            bottom,
            text="Educational ML Tool",
            bg=COLORS["sidebar"],
            fg="#94A3B8",
            font=FONTS["small"],
        ).pack(anchor="w")

    def handle_nav(self, name, command):
        self.set_active_nav(name)
        command()

    def set_active_nav(self, active_name):
        for name, btn in self.nav_buttons.items():
            if name == active_name:
                btn.configure(bg=COLORS["primary"], fg="white")
            else:
                btn.configure(bg=COLORS["sidebar"], fg="#E5E7EB")

    def clear_content(self):
        for widget in self.content.winfo_children():
            widget.destroy()

    def create_page(self):
        page = tk.Frame(self.content, bg=COLORS["bg"])
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_columnconfigure(0, weight=1)
        return page

    def create_card(self, parent):
        card = tk.Frame(
            parent,
            bg=COLORS["surface"],
            padx=28,
            pady=24,
            highlightthickness=0,
            bd=0,
        )
        return card

    def show_home(self):
        self.clear_content()
        self.set_active_nav("Home")

        page = self.create_page()

        tk.Label(
            page,
            text="Welcome to HepatoCheck",
            bg=COLORS["bg"],
            fg=COLORS["text"],
            font=FONTS["page_title"],
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        tk.Label(
            page,
            text="A clean medical data analytics app for hepatitis C and liver risk screening.",
            bg=COLORS["bg"],
            fg=COLORS["muted"],
            font=FONTS["body"],
        ).grid(row=1, column=0, sticky="w", pady=(0, 24))

        card = self.create_card(page)
        card.grid(row=2, column=0, sticky="ew")
        card.grid_columnconfigure(0, weight=1)

        intro = (
            "HepatoCheck allows healthcare users to enter patient laboratory values, "
            "analyze them using a trained machine learning model, and receive a clear "
            "risk classification with supporting information."
        )

        tk.Label(
            card,
            text=intro,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI", 12),
            wraplength=780,
            justify="left",
        ).grid(row=0, column=0, sticky="w", pady=(0, 22))

        features = [
            ("Single Patient Screening", "Enter one patient’s lab values and get an immediate prediction."),
            ("Batch CSV Prediction", "Upload multiple patient records and export the results."),
            ("Screening History", "Review predictions made during the current app session."),
        ]

        for i, (title, desc) in enumerate(features, start=1):
            tk.Label(
                card,
                text=title,
                bg=COLORS["surface"],
                fg=COLORS["text"],
                font=FONTS["section_title"],
            ).grid(row=i, column=0, sticky="w", pady=(10, 0))

            tk.Label(
                card,
                text=desc,
                bg=COLORS["surface"],
                fg=COLORS["muted"],
                font=FONTS["body"],
            ).grid(row=i + 10, column=0, sticky="w", pady=(2, 4))

        action_frame = tk.Frame(page, bg=COLORS["bg"])
        action_frame.grid(row=3, column=0, sticky="w", pady=24)

        self.primary_button(
            action_frame,
            "Start Screening",
            self.show_input_form,
        ).pack(side="left", padx=(0, 12))

        self.secondary_button(
            action_frame,
            "Upload CSV",
            self.show_batch_upload,
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

    def show_input_form(self):
        self.clear_content()
        self.set_active_nav("Patient Input")

        form = PatientInputForm(
            self.content,
            controller=self.controller,
            on_prediction_success=self.show_result,
        )
        form.grid(row=0, column=0, sticky="nsew")

    def show_result(self, result: dict):
        self.clear_content()

        result_view = ResultView(self.content)
        result_view.grid(row=0, column=0, sticky="nsew")
        result_view.show_result(result)

    def show_batch_upload(self):
        self.clear_content()
        self.set_active_nav("Batch Upload")

        batch_view = BatchUploadView(
            self.content,
            controller=self.controller,
        )
        batch_view.grid(row=0, column=0, sticky="nsew")

    def show_history(self):
        self.clear_content()
        self.set_active_nav("History")

        history_view = HistoryView(
            self.content,
            controller=self.controller,
        )
        history_view.grid(row=0, column=0, sticky="nsew")
        history_view.refresh()

    def show_disclaimer(self):
        self.clear_content()
        self.set_active_nav("Disclaimer")

        disclaimer = DisclaimerView(self.content)
        disclaimer.grid(row=0, column=0, sticky="nsew")

    def run(self):
        self.root.mainloop()