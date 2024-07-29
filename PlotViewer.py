import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

class PlotViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Figure Viewer")
        self.geometry("800x600")
        self.figures = []
        self.index = -1

        self.btn_frame = ttk.Frame(self)
        self.btn_frame.pack(fill=tk.X, pady=5)
        self.prev_button = ttk.Button(self.btn_frame, text="Previous", command=self.show_prev_figure)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        self.next_button = ttk.Button(self.btn_frame, text="Next", command=self.show_next_figure)
        self.next_button.pack(side=tk.RIGHT, padx=5)

        self.canvas = None
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def show_figure(self):
        if self.index == -1 or self.index >= len(self.figures):
            return

        # Clear the previous plot if it exists
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

        # Get the current figure and display it
        fig = self.figures[self.index]
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def show_next_figure(self):
        if self.figures:
            self.index = (self.index + 1) % len(self.figures)
            self.show_figure()

    def show_prev_figure(self):
        if self.figures:
            self.index = (self.index - 1) % len(self.figures)
            self.show_figure()

    def add_figure(self, fig):
        self.figures.append(fig)
        if self.index == -1:
            self.index = 0
        self.show_figure()

    def on_closing(self):
        plt.close('all')
        self.quit()
        self.destroy()
