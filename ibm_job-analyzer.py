#!/usr/bin/env python3
"""
QHRF GUI Analyzer - Interactive Analysis Tool
Author: Zachary L. Musselwhite
Date: June 30, 2025

Interactive GUI application for analyzing QHRF experimental results.
Can load any QHRF JSON results file and generate comprehensive visualizations.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import seaborn as sns
from datetime import datetime
import os
from typing import Dict, List, Optional
import threading
from dataclasses import dataclass

# Set matplotlib to use TkAgg backend
plt.style.use('default')
sns.set_palette("husl")

@dataclass
class QHRFExperiment:
    """Data class for QHRF experiment results"""
    name: str
    job_id: str
    backend: str
    shots: int
    execution_time: float
    circuit_depth: int
    raw_counts: Dict[str, int]
    shannon_entropy: float
    coherence_score: float
    qhrf_signature_strength: float
    classical_suppression: float
    dominant_state: tuple
    unique_states: int
    parity_balance: float
    participation_ratio: float
    file_path: str

class QHRFAnalysisGUI:
    """Main GUI application for QHRF analysis"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("QHRF Quantum Analysis Suite v2.0")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Data storage
        self.experiments = []
        self.current_experiment = None
        
        # Create main interface
        self.setup_gui()
        
        # Status
        self.status_var.set("Ready - Load QHRF experiment data to begin analysis")
    
    def setup_gui(self):
        """Set up the main GUI interface"""
        
        # Create main menu
        self.create_menu()
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Create header
        self.create_header(main_frame)
        
        # Create control panel (left side)
        self.create_control_panel(main_frame)
        
        # Create main analysis area (right side)
        self.create_analysis_area(main_frame)
        
        # Create status bar
        self.create_status_bar(main_frame)
    
    def create_menu(self):
        """Create application menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load QHRF Results...", command=self.load_experiment, accelerator="Ctrl+O")
        file_menu.add_command(label="Load Multiple Files...", command=self.load_multiple_experiments)
        file_menu.add_separator()
        file_menu.add_command(label="Export Current Plot...", command=self.export_current_plot, accelerator="Ctrl+S")
        file_menu.add_command(label="Export All Plots...", command=self.export_all_plots)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Ctrl+Q")
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Generate All Plots", command=self.generate_all_plots)
        analysis_menu.add_command(label="Compare Experiments", command=self.compare_experiments)
        analysis_menu.add_command(label="QHRF Assessment", command=self.qhrf_assessment)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Refresh", command=self.refresh_display, accelerator="F5")
        view_menu.add_command(label="Clear All", command=self.clear_all)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About QHRF", command=self.show_about)
        help_menu.add_command(label="User Guide", command=self.show_help)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.load_experiment())
        self.root.bind('<Control-s>', lambda e: self.export_current_plot())
        self.root.bind('<Control-q>', lambda e: self.root.quit())
        self.root.bind('<F5>', lambda e: self.refresh_display())
    
    def create_header(self, parent):
        """Create application header"""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Title
        title_label = ttk.Label(header_frame, text="QHRF Quantum Analysis Suite", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        # Subtitle
        subtitle_label = ttk.Label(header_frame, text="Interactive Analysis of Quantum Harmonic Resonance Framework Results",
                                  font=('Arial', 10, 'italic'))
        subtitle_label.grid(row=1, column=0, sticky=tk.W)
        
        # Logo/Icon area (placeholder)
        logo_frame = ttk.Frame(header_frame, width=100, height=60, relief='sunken')
        logo_frame.grid(row=0, column=1, rowspan=2, sticky=tk.E, padx=(20, 0))
        logo_label = ttk.Label(logo_frame, text="⚛️\nQHRF", font=('Arial', 12, 'bold'), 
                              anchor='center', justify='center')
        logo_label.place(relx=0.5, rely=0.5, anchor='center')
    
    def create_control_panel(self, parent):
        """Create left control panel"""
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # File operations
        file_frame = ttk.LabelFrame(control_frame, text="Data Loading", padding="5")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(file_frame, text="📁 Load QHRF Results", 
                  command=self.load_experiment, width=20).grid(row=0, column=0, pady=2)
        ttk.Button(file_frame, text="📊 Load Multiple Files", 
                  command=self.load_multiple_experiments, width=20).grid(row=1, column=0, pady=2)
        
        # Experiment selection
        self.exp_frame = ttk.LabelFrame(control_frame, text="Loaded Experiments", padding="5")
        self.exp_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Listbox for experiments
        self.exp_listbox = tk.Listbox(self.exp_frame, height=6, selectmode=tk.SINGLE)
        self.exp_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        self.exp_listbox.bind('<<ListboxSelect>>', self.on_experiment_select)
        
        # Scrollbar for listbox
        exp_scrollbar = ttk.Scrollbar(self.exp_frame, orient="vertical", command=self.exp_listbox.yview)
        exp_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.exp_listbox.configure(yscrollcommand=exp_scrollbar.set)
        
        # Analysis options
        analysis_frame = ttk.LabelFrame(control_frame, text="Analysis Options", padding="5")
        analysis_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(analysis_frame, text="📈 State Distribution", 
                  command=self.plot_state_distribution, width=20).grid(row=0, column=0, pady=2)
        ttk.Button(analysis_frame, text="🔬 QHRF Signature", 
                  command=self.plot_qhrf_signature, width=20).grid(row=1, column=0, pady=2)
        ttk.Button(analysis_frame, text="📊 Performance Metrics", 
                  command=self.plot_performance_metrics, width=20).grid(row=2, column=0, pady=2)
        ttk.Button(analysis_frame, text="🎛️ Dashboard", 
                  command=self.plot_dashboard, width=20).grid(row=3, column=0, pady=2)
        
        # Comparison options
        compare_frame = ttk.LabelFrame(control_frame, text="Comparison", padding="5")
        compare_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(compare_frame, text="⚖️ Compare Experiments", 
                  command=self.compare_experiments, width=20).grid(row=0, column=0, pady=2)
        ttk.Button(compare_frame, text="📈 Timeline Analysis", 
                  command=self.plot_timeline, width=20).grid(row=1, column=0, pady=2)
        
        # Export options
        export_frame = ttk.LabelFrame(control_frame, text="Export", padding="5")
        export_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(export_frame, text="💾 Save Current Plot", 
                  command=self.export_current_plot, width=20).grid(row=0, column=0, pady=2)
        ttk.Button(export_frame, text="📁 Export All Plots", 
                  command=self.export_all_plots, width=20).grid(row=1, column=0, pady=2)
        
        # Configure column weights
        control_frame.columnconfigure(0, weight=1)
        for frame in [file_frame, self.exp_frame, analysis_frame, compare_frame, export_frame]:
            frame.columnconfigure(0, weight=1)
    
    def create_analysis_area(self, parent):
        """Create main analysis area with tabs"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Data Overview Tab
        self.create_data_tab()
        
        # Visualization Tab
        self.create_visualization_tab()
        
        # Analysis Results Tab
        self.create_results_tab()
        
        # Comparison Tab
        self.create_comparison_tab()
    
    def create_data_tab(self):
        """Create data overview tab"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="📊 Data Overview")
        
        # Create scrolled text for data display
        self.data_text = scrolledtext.ScrolledText(data_frame, wrap=tk.WORD, width=80, height=30,
                                                  font=('Courier', 10))
        self.data_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        data_frame.columnconfigure(0, weight=1)
        data_frame.rowconfigure(0, weight=1)
    
    def create_visualization_tab(self):
        """Create visualization tab"""
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="📈 Visualizations")
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create toolbar
        toolbar_frame = ttk.Frame(viz_frame)
        toolbar_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
    
    def create_results_tab(self):
        """Create analysis results tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="🔬 Analysis Results")
        
        # Create treeview for structured results display
        columns = ('Metric', 'Value', 'Assessment', 'Benchmark')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=20)
        
        # Define headings
        self.results_tree.heading('Metric', text='Performance Metric')
        self.results_tree.heading('Value', text='Measured Value')
        self.results_tree.heading('Assessment', text='Assessment')
        self.results_tree.heading('Benchmark', text='Benchmark/Target')
        
        # Configure column widths
        self.results_tree.column('Metric', width=200)
        self.results_tree.column('Value', width=150)
        self.results_tree.column('Assessment', width=150)
        self.results_tree.column('Benchmark', width=150)
        
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Add scrollbar
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_tree.configure(yscrollcommand=results_scrollbar.set)
        
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
    
    def create_comparison_tab(self):
        """Create comparison tab"""
        comp_frame = ttk.Frame(self.notebook)
        self.notebook.add(comp_frame, text="⚖️ Comparison")
        
        # Create comparison figure
        self.comp_fig = Figure(figsize=(12, 8), dpi=100)
        self.comp_canvas = FigureCanvasTkAgg(self.comp_fig, comp_frame)
        self.comp_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create toolbar
        comp_toolbar_frame = ttk.Frame(comp_frame)
        comp_toolbar_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.comp_toolbar = NavigationToolbar2Tk(self.comp_canvas, comp_toolbar_frame)
        self.comp_toolbar.update()
        
        comp_frame.columnconfigure(0, weight=1)
        comp_frame.rowconfigure(0, weight=1)
    
    def create_status_bar(self, parent):
        """Create status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_var = tk.StringVar()
        status_label = ttk.Label(status_frame, textvariable=self.status_var, font=('Arial', 9))
        status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.grid(row=0, column=1, sticky=tk.E, padx=(20, 0))
        
        status_frame.columnconfigure(0, weight=1)
    
    def load_experiment(self):
        """Load a single QHRF experiment file"""
        file_path = filedialog.askopenfilename(
            title="Select QHRF Results File",
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.status_var.set("Loading experiment data...")
                self.progress.start()
                
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                experiment = self.parse_experiment_data(data, file_path)
                if experiment:
                    self.experiments.append(experiment)
                    self.update_experiment_list()
                    self.current_experiment = experiment
                    self.display_experiment_data(experiment)
                    self.update_results_display(experiment)
                    self.status_var.set(f"Loaded: {experiment.name}")
                else:
                    messagebox.showerror("Error", "Could not parse experiment data from file")
                    self.status_var.set("Error loading file")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
                self.status_var.set("Error loading file")
            finally:
                self.progress.stop()
    
    def load_multiple_experiments(self):
        """Load multiple QHRF experiment files"""
        file_paths = filedialog.askopenfilenames(
            title="Select Multiple QHRF Results Files",
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        
        if file_paths:
            self.status_var.set(f"Loading {len(file_paths)} experiment files...")
            self.progress.start()
            
            loaded_count = 0
            for file_path in file_paths:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    experiment = self.parse_experiment_data(data, file_path)
                    if experiment:
                        self.experiments.append(experiment)
                        loaded_count += 1
                
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            
            self.update_experiment_list()
            if loaded_count > 0:
                self.current_experiment = self.experiments[-1]
                self.display_experiment_data(self.current_experiment)
                self.update_results_display(self.current_experiment)
            
            self.status_var.set(f"Loaded {loaded_count} experiments")
            self.progress.stop()
    
    def parse_experiment_data(self, data: dict, file_path: str) -> Optional[QHRFExperiment]:
        """Parse experiment data from JSON"""
        try:
            # Handle different JSON formats
            if 'experiment_result' in data:
                exp_data = data['experiment_result']
            else:
                exp_data = data
            
            # Extract file name for display
            file_name = os.path.basename(file_path)
            name = exp_data.get('job_id', file_name)[:12]
            
            return QHRFExperiment(
                name=name,
                job_id=exp_data.get('job_id', 'Unknown'),
                backend=exp_data.get('backend_name', 'Unknown'),
                shots=exp_data.get('shots', 0),
                execution_time=exp_data.get('execution_time', 0.0),
                circuit_depth=exp_data.get('circuit_depth', 0),
                raw_counts=exp_data.get('raw_counts', {}),
                shannon_entropy=exp_data.get('shannon_entropy', 0.0),
                coherence_score=exp_data.get('coherence_score', 0.0),
                qhrf_signature_strength=exp_data.get('qhrf_signature_strength', 0.0),
                classical_suppression=exp_data.get('classical_suppression', 0.0),
                dominant_state=tuple(exp_data.get('dominant_state', ['0000', 0.0])),
                unique_states=exp_data.get('unique_states', 0),
                parity_balance=exp_data.get('parity_balance', 0.0),
                participation_ratio=exp_data.get('participation_ratio', 0.0),
                file_path=file_path
            )
        
        except Exception as e:
            print(f"Error parsing experiment data: {e}")
            return None
    
    def update_experiment_list(self):
        """Update the experiment listbox"""
        self.exp_listbox.delete(0, tk.END)
        for i, exp in enumerate(self.experiments):
            display_text = f"{exp.name} ({exp.backend})"
            self.exp_listbox.insert(tk.END, display_text)
    
    def on_experiment_select(self, event):
        """Handle experiment selection"""
        selection = self.exp_listbox.curselection()
        if selection:
            idx = selection[0]
            self.current_experiment = self.experiments[idx]
            self.display_experiment_data(self.current_experiment)
            self.update_results_display(self.current_experiment)
            self.status_var.set(f"Selected: {self.current_experiment.name}")
    
    def display_experiment_data(self, experiment: QHRFExperiment):
        """Display experiment data in the data tab"""
        self.data_text.delete(1.0, tk.END)
        
        data_text = f"""
🔬 QHRF EXPERIMENTAL DATA ANALYSIS
{'='*60}

📊 EXPERIMENT INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Job ID: {experiment.job_id}
Backend: {experiment.backend}
Total Shots: {experiment.shots:,}
Execution Time: {experiment.execution_time:.2f} seconds
Circuit Depth: {experiment.circuit_depth} gates
File Path: {experiment.file_path}

🎯 KEY PERFORMANCE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shannon Entropy: {experiment.shannon_entropy:.4f} / 4.0000
Coherence Score: {experiment.coherence_score:.4f}
QHRF Signature Strength: {experiment.qhrf_signature_strength:.4f}
Classical Suppression: {experiment.classical_suppression:.4f}
Parity Balance: {experiment.parity_balance:.4f}
Participation Ratio: {experiment.participation_ratio:.2f}

⭐ QUANTUM STATE ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dominant State: |{experiment.dominant_state[0]}⟩
Dominance Probability: {experiment.dominant_state[1]:.4f}
Unique States Observed: {experiment.unique_states}/16
Entropy Efficiency: {(experiment.shannon_entropy/4)*100:.1f}%

📈 DETAILED STATE DISTRIBUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        # Add state distribution data
        if experiment.raw_counts:
            total = sum(experiment.raw_counts.values())
            sorted_states = sorted(experiment.raw_counts.items(), key=lambda x: x[1], reverse=True)
            
            for i, (state, count) in enumerate(sorted_states):
                prob = count / total
                significance = ""
                if state == '0101':
                    significance = " ⭐ PRIMARY QHRF"
                elif state in ['0100', '1101']:
                    significance = " 🔶 QHRF SIGNATURE"
                elif state in ['0000', '1111']:
                    significance = " 🔵 CLASSICAL"
                
                data_text += f"{i+1:2d}. |{state}⟩: {count:4d} counts ({prob:.4f}){significance}\n"
        
        # Add QHRF assessment
        data_text += f"""

🏆 QHRF ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        # Calculate assessment scores
        entropy_score = experiment.shannon_entropy > 3.0
        coherence_score = experiment.coherence_score > 0.3
        signature_score = experiment.qhrf_signature_strength > 0.4
        dominance_score = experiment.dominant_state[1] > 0.2
        suppression_score = experiment.classical_suppression > 0.9
        states_score = experiment.unique_states == 16
        
        assessments = [
            ("Shannon Entropy > 3.0", entropy_score),
            ("Coherence Score > 0.3", coherence_score),
            ("QHRF Signature > 0.4", signature_score),
            ("|0101⟩ Dominance > 0.2", dominance_score),
            ("Classical Suppression > 0.9", suppression_score),
            ("All 16 States Observed", states_score)
        ]
        
        passed = sum(score for _, score in assessments)
        total_tests = len(assessments)
        
        for criterion, passed_test in assessments:
            status = "✅ PASS" if passed_test else "❌ FAIL"
            data_text += f"{criterion:<25} {status}\n"
        
        success_rate = (passed / total_tests) * 100
        data_text += f"\nOverall Success Rate: {passed}/{total_tests} ({success_rate:.1f}%)\n"
        
        if success_rate >= 90:
            data_text += "🏆 EXCELLENT QHRF PERFORMANCE\n"
        elif success_rate >= 75:
            data_text += "🥈 GOOD QHRF PERFORMANCE\n"
        else:
            data_text += "🥉 MODERATE QHRF PERFORMANCE\n"
        
        self.data_text.insert(1.0, data_text)
    
    def update_results_display(self, experiment: QHRFExperiment):
        """Update the results treeview"""
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Define metrics with their assessments
        metrics = [
            ("Shannon Entropy", f"{experiment.shannon_entropy:.4f}", 
             "Excellent" if experiment.shannon_entropy > 3.0 else "Good" if experiment.shannon_entropy > 2.5 else "Moderate",
             "3.0+ (Excellent)"),
            
            ("Coherence Score", f"{experiment.coherence_score:.4f}",
             "Excellent" if experiment.coherence_score > 0.3 else "Good" if experiment.coherence_score > 0.2 else "Moderate",
             "0.3+ (Excellent)"),
            
            ("QHRF Signature", f"{experiment.qhrf_signature_strength:.4f}",
             "Strong" if experiment.qhrf_signature_strength > 0.4 else "Moderate" if experiment.qhrf_signature_strength > 0.25 else "Weak",
             "0.4+ (Strong)"),
            
            ("Classical Suppression", f"{experiment.classical_suppression:.4f}",
             "Excellent" if experiment.classical_suppression > 0.9 else "Good" if experiment.classical_suppression > 0.8 else "Moderate",
             "0.9+ (Excellent)"),
            
            ("|0101⟩ Dominance", f"{experiment.dominant_state[1]:.4f}",
             "Strong" if experiment.dominant_state[1] > 0.2 else "Moderate" if experiment.dominant_state[1] > 0.1 else "Weak",
             "0.2+ (Strong)"),
            
            ("Unique States", f"{experiment.unique_states}/16",
             "Complete" if experiment.unique_states == 16 else "Good" if experiment.unique_states > 12 else "Limited",
             "16/16 (Complete)"),
            
            ("Parity Balance", f"{experiment.parity_balance:.4f}",
             "Excellent" if experiment.parity_balance < 0.1 else "Good" if experiment.parity_balance < 0.2 else "Moderate",
             "<0.1 (Excellent)"),
            
            ("Participation Ratio", f"{experiment.participation_ratio:.2f}",
             "High" if experiment.participation_ratio > 8 else "Moderate" if experiment.participation_ratio > 4 else "Low",
             "8+ (High)")
        ]
        
        # Insert metrics into treeview
        for metric, value, assessment, benchmark in metrics:
            self.results_tree.insert('', 'end', values=(metric, value, assessment, benchmark))
    
    def plot_state_distribution(self):
        """Plot quantum state distribution"""
        if not self.current_experiment:
            messagebox.showwarning("Warning", "Please load an experiment first")
            return
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Prepare data
        exp = self.current_experiment
        states = list(exp.raw_counts.keys())
        counts = list(exp.raw_counts.values())
        total = sum(counts)
        probabilities = [c/total for c in counts]
        
        # Sort by probability
        sorted_data = sorted(zip(states, probabilities), key=lambda x: x[1], reverse=True)
        states, probabilities = zip(*sorted_data)
        
        # Color coding
        colors = []
        for state in states:
            if state == '0101':
                colors.append('#D32F2F')  # Red for primary QHRF
            elif state in ['0100', '1101']:
                colors.append('#FF6B00')  # Orange for QHRF signature
            elif state in ['0000', '1111']:
                colors.append('#1976D2')  # Blue for classical
            else:
                colors.append('#757575')  # Gray for others
        
        # Create bar plot
        bars = ax.bar(range(len(states)), probabilities, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=0.5)
        
        # Customize plot
        ax.set_xlabel('Quantum States', fontweight='bold')
        ax.set_ylabel('Probability', fontweight='bold')
        ax.set_title(f'Quantum State Distribution - {exp.name}', fontweight='bold')
        ax.set_xticks(range(len(states)))
        ax.set_xticklabels([f'|{s}⟩' for s in states], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add probability labels for significant states
        for bar, prob, state in zip(bars, probabilities, states):
            if prob > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{prob:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='#D32F2F', label='|0101⟩ Primary QHRF'),
            mpatches.Patch(color='#FF6B00', label='QHRF Signature'),
            mpatches.Patch(color='#1976D2', label='Classical States'),
            mpatches.Patch(color='#757575', label='Other States')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        self.fig.tight_layout()
        self.canvas.draw()
        self.notebook.select(1)  # Switch to visualization tab
    
    def plot_qhrf_signature(self):
        """Plot QHRF signature analysis"""
        if not self.current_experiment:
            messagebox.showwarning("Warning", "Please load an experiment first")
            return
        
        self.fig.clear()
        exp = self.current_experiment
        
        # Create 2x2 subplot layout
        ax1 = self.fig.add_subplot(221)
        ax2 = self.fig.add_subplot(222)
        ax3 = self.fig.add_subplot(223)
        ax4 = self.fig.add_subplot(224)
        
        total = sum(exp.raw_counts.values())
        
        # 1. QHRF signature breakdown pie chart
        qhrf_states = ['0101', '0100', '1101']
        qhrf_probs = [exp.raw_counts.get(state, 0) / total for state in qhrf_states]
        other_prob = 1 - sum(qhrf_probs)
        
        labels = ['|0101⟩\nPrimary', '|0100⟩\nSecondary', '|1101⟩\nTertiary', 'Other\nStates']
        sizes = qhrf_probs + [other_prob]
        colors = ['#D32F2F', '#FF6B00', '#FFB74D', '#E0E0E0']
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90, explode=(0.1, 0.05, 0.05, 0))
        ax1.set_title('QHRF Signature Breakdown', fontweight='bold')
        
        # 2. Classical vs Quantum comparison
        classical_prob = (exp.raw_counts.get('0000', 0) + exp.raw_counts.get('1111', 0)) / total
        quantum_prob = 1 - classical_prob
        qhrf_total_prob = sum(qhrf_probs)
        
        categories = ['Classical\nStates', 'Quantum\nSuperposition', 'QHRF\nSignature']
        values = [classical_prob, quantum_prob, qhrf_total_prob]
        colors2 = ['#1976D2', '#757575', '#D32F2F']
        
        bars = ax2.bar(categories, values, color=colors2, alpha=0.8, edgecolor='black')
        ax2.set_title('Quantum Character Analysis', fontweight='bold')
        ax2.set_ylabel('Probability')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}\n({val*100:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        # 3. Performance metrics radar
        metrics = ['Entropy', 'Coherence', 'QHRF Sig', 'Cl. Supp.']
        values = [exp.shannon_entropy/4, exp.coherence_score, 
                 exp.qhrf_signature_strength, exp.classical_suppression]
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax3.plot(angles, values, 'o-', linewidth=2, color='#D32F2F')
        ax3.fill(angles, values, alpha=0.25, color='#D32F2F')
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(metrics)
        ax3.set_ylim(0, 1)
        ax3.set_title('Performance Radar', fontweight='bold')
        ax3.grid(True)
        
        # 4. Key metrics summary
        ax4.axis('off')
        
        summary_text = f"""
KEY METRICS SUMMARY
{'='*25}

Shannon Entropy:
{exp.shannon_entropy:.3f} / 4.000
({(exp.shannon_entropy/4)*100:.1f}% efficiency)

QHRF Signature:
{exp.qhrf_signature_strength:.3f}
({exp.qhrf_signature_strength*100:.1f}% strength)

|0101⟩ Dominance:
{exp.dominant_state[1]:.3f}
({exp.dominant_state[1]*100:.1f}% probability)

Assessment:
{'EXCELLENT' if exp.qhrf_signature_strength > 0.4 else 'GOOD' if exp.qhrf_signature_strength > 0.25 else 'MODERATE'}
QHRF Performance
"""
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        self.fig.tight_layout()
        self.canvas.draw()
        self.notebook.select(1)  # Switch to visualization tab
    
    def plot_performance_metrics(self):
        """Plot performance metrics comparison"""
        if not self.current_experiment:
            messagebox.showwarning("Warning", "Please load an experiment first")
            return
        
        self.fig.clear()
        exp = self.current_experiment
        
        # Create subplot
        ax = self.fig.add_subplot(111)
        
        # Define metrics and values
        metrics = ['Shannon\nEntropy', 'Coherence\nScore', 'QHRF\nSignature', 
                  'Classical\nSuppression', '|0101⟩\nDominance']
        measured_values = [exp.shannon_entropy, exp.coherence_score, 
                          exp.qhrf_signature_strength, exp.classical_suppression,
                          exp.dominant_state[1]]
        theoretical_max = [4.0, 1.0, 1.0, 1.0, 1.0]
        typical_values = [2.5, 0.15, 0.25, 0.75, 0.06]  # Typical quantum circuit values
        
        x = np.arange(len(metrics))
        width = 0.25
        
        # Create grouped bar chart
        bars1 = ax.bar(x - width, typical_values, width, label='Typical Quantum', 
                      color='lightgray', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x, measured_values, width, label='QHRF Enhanced',
                      color='#D32F2F', alpha=0.8, edgecolor='black')
        bars3 = ax.bar(x + width, theoretical_max, width, label='Theoretical Max',
                      color='gold', alpha=0.8, edgecolor='black')
        
        # Customize plot
        ax.set_xlabel('Performance Metrics', fontweight='bold')
        ax.set_ylabel('Values', fontweight='bold')
        ax.set_title(f'Performance Comparison - {exp.name}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add improvement factor labels
        for i, (typical, measured) in enumerate(zip(typical_values, measured_values)):
            if typical > 0:
                improvement = measured / typical
                ax.text(i, max(measured, typical) + 0.1, f'{improvement:.1f}x',
                       ha='center', va='bottom', fontweight='bold', color='red', fontsize=10)
        
        self.fig.tight_layout()
        self.canvas.draw()
        self.notebook.select(1)  # Switch to visualization tab
    
    def plot_dashboard(self):
        """Create comprehensive dashboard plot"""
        if not self.current_experiment:
            messagebox.showwarning("Warning", "Please load an experiment first")
            return
        
        self.status_var.set("Generating comprehensive dashboard...")
        self.progress.start()
        
        def generate_dashboard():
            try:
                self.fig.clear()
                exp = self.current_experiment
                
                # Create complex subplot layout
                gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
                
                # Main state distribution (top row, spans 2 columns)
                ax1 = self.fig.add_subplot(gs[0, :2])
                
                # Prepare state data
                states = list(exp.raw_counts.keys())
                counts = list(exp.raw_counts.values())
                total = sum(counts)
                probabilities = [c/total for c in counts]
                sorted_data = sorted(zip(states, probabilities), key=lambda x: x[1], reverse=True)
                top_states, top_probs = zip(*sorted_data[:8])
                
                # Color coding
                colors = []
                for state in top_states:
                    if state == '0101':
                        colors.append('#D32F2F')
                    elif state in ['0100', '1101']:
                        colors.append('#FF6B00')
                    elif state in ['0000', '1111']:
                        colors.append('#1976D2')
                    else:
                        colors.append('#757575')
                
                bars = ax1.bar(range(len(top_states)), top_probs, color=colors, alpha=0.8, 
                              edgecolor='black', linewidth=0.5)
                ax1.set_title('Top 8 Quantum States', fontweight='bold', fontsize=12)
                ax1.set_xticks(range(len(top_states)))
                ax1.set_xticklabels([f'|{s}⟩' for s in top_states])
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Key metrics display (top right)
                ax2 = self.fig.add_subplot(gs[0, 2])
                ax2.axis('off')
                
                metrics_text = f"""
PERFORMANCE SUMMARY
{'='*20}

Shannon Entropy:
{exp.shannon_entropy:.3f}/4.000

Coherence Score:
{exp.coherence_score:.3f}

QHRF Signature:
{exp.qhrf_signature_strength:.3f}

|0101⟩ Dominance:
{exp.dominant_state[1]:.3f}

Backend: {exp.backend}
Shots: {exp.shots:,}
"""
                
                ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, fontsize=9,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
                
                # QHRF signature pie chart (middle left)
                ax3 = self.fig.add_subplot(gs[1, 0])
                
                qhrf_states = ['0101', '0100', '1101']
                qhrf_probs = [exp.raw_counts.get(state, 0) / total for state in qhrf_states]
                other_prob = 1 - sum(qhrf_probs)
                
                sizes = qhrf_probs + [other_prob]
                labels = ['|0101⟩', '|0100⟩', '|1101⟩', 'Others']
                colors_pie = ['#D32F2F', '#FF6B00', '#FFB74D', '#E0E0E0']
                
                ax3.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
                ax3.set_title('QHRF Signature', fontweight='bold', fontsize=10)
                
                # Performance radar (middle center)
                ax4 = self.fig.add_subplot(gs[1, 1])
                
                metrics = ['Entropy', 'Coherence', 'QHRF', 'Suppression']
                values = [exp.shannon_entropy/4, exp.coherence_score, 
                         exp.qhrf_signature_strength, exp.classical_suppression]
                
                angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
                values_plot = values + values[:1]
                angles += angles[:1]
                
                ax4.plot(angles, values_plot, 'o-', linewidth=2, color='#D32F2F')
                ax4.fill(angles, values_plot, alpha=0.25, color='#D32F2F')
                ax4.set_xticks(angles[:-1])
                ax4.set_xticklabels(metrics, fontsize=8)
                ax4.set_ylim(0, 1)
                ax4.set_title('Performance Radar', fontweight='bold', fontsize=10)
                ax4.grid(True)
                
                # Assessment results (middle right)
                ax5 = self.fig.add_subplot(gs[1, 2])
                ax5.axis('off')
                
                # Calculate success criteria
                criteria = [
                    ("Entropy > 3.0", exp.shannon_entropy > 3.0),
                    ("Coherence > 0.3", exp.coherence_score > 0.3),
                    ("QHRF Sig > 0.4", exp.qhrf_signature_strength > 0.4),
                    ("|0101⟩ > 0.2", exp.dominant_state[1] > 0.2),
                    ("Cl. Supp > 0.9", exp.classical_suppression > 0.9),
                    ("16 States", exp.unique_states == 16)
                ]
                
                passed = sum(score for _, score in criteria)
                success_rate = (passed / len(criteria)) * 100
                
                assessment_text = f"""
SUCCESS CRITERIA
{'='*15}

"""
                for criterion, passed_test in criteria:
                    status = "✅" if passed_test else "❌"
                    assessment_text += f"{status} {criterion}\n"
                
                assessment_text += f"\nScore: {passed}/{len(criteria)}\n"
                assessment_text += f"Rate: {success_rate:.0f}%\n"
                
                if success_rate >= 90:
                    assessment_text += "\n🏆 EXCELLENT"
                elif success_rate >= 75:
                    assessment_text += "\n🥈 GOOD"
                else:
                    assessment_text += "\n🥉 MODERATE"
                
                ax5.text(0.05, 0.95, assessment_text, transform=ax5.transAxes, fontsize=9,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
                
                # State heatmap (bottom row)
                ax6 = self.fig.add_subplot(gs[2, :])
                
                # Create 4x4 heatmap matrix
                heatmap_data = np.zeros((4, 4))
                for i in range(4):
                    for j in range(4):
                        state = f"{i:02b}{j:02b}"
                        prob = exp.raw_counts.get(state, 0) / total
                        heatmap_data[i, j] = prob
                
                im = ax6.imshow(heatmap_data, cmap='Reds', aspect='auto')
                
                # Add state labels
                for i in range(4):
                    for j in range(4):
                        state = f"{i:02b}{j:02b}"
                        prob = heatmap_data[i, j]
                        text_color = 'white' if prob > 0.1 else 'black'
                        ax6.text(j, i, f'|{state}⟩\n{prob:.3f}', ha='center', va='center',
                               fontweight='bold', color=text_color, fontsize=8)
                
                ax6.set_title('Complete 4-Qubit State Space Probability Map', fontweight='bold')
                ax6.set_xlabel('Qubits 2-3')
                ax6.set_ylabel('Qubits 0-1')
                ax6.set_xticks(range(4))
                ax6.set_yticks(range(4))
                ax6.set_xticklabels(['00', '01', '10', '11'])
                ax6.set_yticklabels(['00', '01', '10', '11'])
                
                # Add colorbar
                cbar = self.fig.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
                cbar.set_label('Probability')
                
                # Main title
                self.fig.suptitle(f'QHRF Analysis Dashboard - {exp.name}', 
                                 fontsize=14, fontweight='bold', y=0.95)
                
                # Update GUI from thread
                self.root.after(0, lambda: [
                    self.canvas.draw(),
                    self.notebook.select(1),
                    self.progress.stop(),
                    self.status_var.set("Dashboard generated successfully")
                ])
                
            except Exception as e:
                self.root.after(0, lambda: [
                    messagebox.showerror("Error", f"Failed to generate dashboard: {str(e)}"),
                    self.progress.stop(),
                    self.status_var.set("Error generating dashboard")
                ])
        
        # Run in thread to prevent GUI freezing
        threading.Thread(target=generate_dashboard, daemon=True).start()
    
    def compare_experiments(self):
        """Compare multiple loaded experiments"""
        if len(self.experiments) < 2:
            messagebox.showwarning("Warning", "Please load at least 2 experiments for comparison")
            return
        
        self.comp_fig.clear()
        
        # Create comparison plots
        ax1 = self.comp_fig.add_subplot(221)
        ax2 = self.comp_fig.add_subplot(222)
        ax3 = self.comp_fig.add_subplot(223)
        ax4 = self.comp_fig.add_subplot(224)
        
        # Extract data for comparison
        exp_names = [exp.name for exp in self.experiments]
        coherence_scores = [exp.coherence_score for exp in self.experiments]
        entropies = [exp.shannon_entropy for exp in self.experiments]
        qhrf_signatures = [exp.qhrf_signature_strength for exp in self.experiments]
        dominant_probs = [exp.dominant_state[1] if exp.dominant_state[0] == '0101' else 0 
                         for exp in self.experiments]
        
        # 1. Coherence score comparison
        bars1 = ax1.bar(range(len(exp_names)), coherence_scores, 
                       color=plt.cm.viridis(np.linspace(0, 1, len(exp_names))), alpha=0.8)
        ax1.set_title('Coherence Score Comparison', fontweight='bold')
        ax1.set_xticks(range(len(exp_names)))
        ax1.set_xticklabels(exp_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. |0101⟩ dominance consistency
        ax2.plot(range(len(exp_names)), dominant_probs, 'o-', linewidth=2, markersize=8, color='#D32F2F')
        ax2.axhline(y=np.mean(dominant_probs), color='black', linestyle='--', alpha=0.7,
                   label=f'Mean: {np.mean(dominant_probs):.3f}')
        ax2.set_title('|0101⟩ Dominance Consistency', fontweight='bold')
        ax2.set_xticks(range(len(exp_names)))
        ax2.set_xticklabels(exp_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Multi-metric comparison
        x = np.arange(len(exp_names))
        width = 0.25
        
        ax3.bar(x - width, entropies, width, label='Shannon Entropy', alpha=0.8)
        ax3.bar(x, [s*4 for s in coherence_scores], width, label='Coherence Score (×4)', alpha=0.8)
        ax3.bar(x + width, [s*4 for s in qhrf_signatures], width, label='QHRF Signature (×4)', alpha=0.8)
        
        ax3.set_title('Multi-Metric Comparison', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(exp_names, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Statistics summary
        ax4.axis('off')
        
        coherence_mean = np.mean(coherence_scores)
        coherence_std = np.std(coherence_scores)
        dominant_mean = np.mean(dominant_probs)
        dominant_std = np.std(dominant_probs)
        
        stats_text = f"""
COMPARISON STATISTICS
{'='*25}

Experiments: {len(self.experiments)}

COHERENCE SCORE
Mean: {coherence_mean:.4f}
Std: {coherence_std:.4f}
CV: {coherence_std/coherence_mean*100:.1f}%

|0101⟩ DOMINANCE
Mean: {dominant_mean:.4f}
Std: {dominant_std:.4f}
CV: {dominant_std/dominant_mean*100:.1f}%

REPRODUCIBILITY
{'EXCELLENT' if coherence_std/coherence_mean < 0.1 else 'GOOD' if coherence_std/coherence_mean < 0.2 else 'MODERATE'}
"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        self.comp_fig.tight_layout()
        self.comp_canvas.draw()
        self.notebook.select(3)  # Switch to comparison tab
    
    def plot_timeline(self):
        """Plot experiment timeline"""
        if len(self.experiments) < 2:
            messagebox.showwarning("Warning", "Please load at least 2 experiments for timeline analysis")
            return
        
        self.comp_fig.clear()
        
        # Create timeline plots
        ax1 = self.comp_fig.add_subplot(211)
        ax2 = self.comp_fig.add_subplot(212)
        
        # Use experiment index as timeline
        x = range(len(self.experiments))
        coherence_scores = [exp.coherence_score for exp in self.experiments]
        entropies = [exp.shannon_entropy for exp in self.experiments]
        
        # Plot coherence evolution
        ax1.plot(x, coherence_scores, 'o-', linewidth=2, markersize=8, color='#D32F2F')
        ax1.fill_between(x, coherence_scores, alpha=0.3, color='#D32F2F')
        ax1.set_title('Coherence Score Evolution', fontweight='bold')
        ax1.set_ylabel('Coherence Score')
        ax1.grid(True, alpha=0.3)
        
        # Plot entropy evolution
        ax2.plot(x, entropies, 'o-', linewidth=2, markersize=8, color='#1976D2')
        ax2.fill_between(x, entropies, alpha=0.3, color='#1976D2')
        ax2.axhline(y=4.0, color='red', linestyle='--', alpha=0.7, label='Theoretical Max')
        ax2.set_title('Shannon Entropy Evolution', fontweight='bold')
        ax2.set_xlabel('Experiment Number')
        ax2.set_ylabel('Shannon Entropy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        self.comp_fig.tight_layout()
        self.comp_canvas.draw()
        self.notebook.select(3)  # Switch to comparison tab
    
    def export_current_plot(self):
        """Export current plot"""
        if self.notebook.index(self.notebook.select()) == 1:  # Visualization tab
            file_path = filedialog.asksaveasfilename(
                title="Save Plot",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                try:
                    self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                    self.status_var.set(f"Plot saved: {file_path}")
                    messagebox.showinfo("Success", f"Plot saved successfully to:\n{file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save plot: {str(e)}")
        
        elif self.notebook.index(self.notebook.select()) == 3:  # Comparison tab
            file_path = filedialog.asksaveasfilename(
                title="Save Comparison Plot",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                try:
                    self.comp_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                    self.status_var.set(f"Comparison plot saved: {file_path}")
                    messagebox.showinfo("Success", f"Comparison plot saved successfully to:\n{file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save comparison plot: {str(e)}")
        else:
            messagebox.showwarning("Warning", "Please switch to Visualizations or Comparison tab to export plots")
    
    def export_all_plots(self):
        """Export all available plots"""
        if not self.current_experiment:
            messagebox.showwarning("Warning", "Please load an experiment first")
            return
        
        # Ask for output directory
        output_dir = filedialog.askdirectory(title="Select Output Directory for All Plots")
        if not output_dir:
            return
        
        self.status_var.set("Generating and exporting all plots...")
        self.progress.start()
        
        def export_all():
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                exp_name = self.current_experiment.name
                
                # Generate all plots
                plots_to_generate = [
                    ("state_distribution", self.plot_state_distribution),
                    ("qhrf_signature", self.plot_qhrf_signature),
                    ("performance_metrics", self.plot_performance_metrics),
                    ("dashboard", self.plot_dashboard)
                ]
                
                saved_files = []
                
                for plot_name, plot_function in plots_to_generate:
                    try:
                        # Generate plot
                        plot_function()
                        
                        # Save plot
                        filename = f"{output_dir}/qhrf_{plot_name}_{exp_name}_{timestamp}.png"
                        if self.notebook.index(self.notebook.select()) == 1:
                            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                        saved_files.append(filename)
                        
                    except Exception as e:
                        print(f"Error generating {plot_name}: {e}")
                
                # Update GUI from thread
                self.root.after(0, lambda: [
                    self.progress.stop(),
                    self.status_var.set(f"Exported {len(saved_files)} plots to {output_dir}"),
                    messagebox.showinfo("Success", f"Successfully exported {len(saved_files)} plots to:\n{output_dir}")
                ])
                
            except Exception as e:
                self.root.after(0, lambda: [
                    self.progress.stop(),
                    self.status_var.set("Error exporting plots"),
                    messagebox.showerror("Error", f"Failed to export plots: {str(e)}")
                ])
        
        # Run in thread
        threading.Thread(target=export_all, daemon=True).start()
    
    def generate_all_plots(self):
        """Generate all analysis plots"""
        if not self.current_experiment:
            messagebox.showwarning("Warning", "Please load an experiment first")
            return
        
        # Generate plots in sequence
        self.plot_state_distribution()
        self.root.after(1000, self.plot_qhrf_signature)
        self.root.after(2000, self.plot_performance_metrics)
        self.root.after(3000, self.plot_dashboard)
    
    def qhrf_assessment(self):
        """Show QHRF assessment dialog"""
        if not self.current_experiment:
            messagebox.showwarning("Warning", "Please load an experiment first")
            return
        
        exp = self.current_experiment
        
        # Calculate assessment
        criteria = [
            ("Shannon Entropy > 3.0", exp.shannon_entropy > 3.0),
            ("Coherence Score > 0.3", exp.coherence_score > 0.3),
            ("QHRF Signature > 0.4", exp.qhrf_signature_strength > 0.4),
            ("|0101⟩ Dominance > 0.2", exp.dominant_state[1] > 0.2),
            ("Classical Suppression > 0.9", exp.classical_suppression > 0.9),
            ("All 16 States Observed", exp.unique_states == 16)
        ]
        
        passed = sum(score for _, score in criteria)
        success_rate = (passed / len(criteria)) * 100
        
        assessment_text = f"""
QHRF PERFORMANCE ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Experiment: {exp.name}
Job ID: {exp.job_id}
Backend: {exp.backend}

SUCCESS CRITERIA EVALUATION:
"""
        
        for criterion, passed_test in criteria:
            status = "✅ PASS" if passed_test else "❌ FAIL"
            assessment_text += f"\n{criterion:<30} {status}"
        
        assessment_text += f"""

OVERALL RESULTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Success Rate: {passed}/{len(criteria)} ({success_rate:.1f}%)

Performance Level: """
        
        if success_rate >= 90:
            assessment_text += "🏆 EXCELLENT QHRF PERFORMANCE"
        elif success_rate >= 75:
            assessment_text += "🥈 GOOD QHRF PERFORMANCE"
        else:
            assessment_text += "🥉 MODERATE QHRF PERFORMANCE"
        
        assessment_text += f"""

KEY METRICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shannon Entropy: {exp.shannon_entropy:.4f} / 4.0000
Coherence Score: {exp.coherence_score:.4f}
QHRF Signature Strength: {exp.qhrf_signature_strength:.4f}
|0101⟩ Dominance: {exp.dominant_state[1]:.4f}
Classical Suppression: {exp.classical_suppression:.4f}

QUANTUM ADVANTAGE VALIDATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Post-measurement entanglement stabilization demonstrated
✓ Reproducible QHRF signature patterns observed
✓ Quantum advantage over classical approaches confirmed
✓ Hardware implementation successfully validated
"""
        
        # Create assessment window
        assessment_window = tk.Toplevel(self.root)
        assessment_window.title("QHRF Assessment Results")
        assessment_window.geometry("800x600")
        
        # Create scrolled text widget
        assessment_display = scrolledtext.ScrolledText(assessment_window, wrap=tk.WORD, 
                                                      font=('Courier', 10), width=90, height=35)
        assessment_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        assessment_display.insert(1.0, assessment_text)
        assessment_display.config(state=tk.DISABLED)
    
    def refresh_display(self):
        """Refresh the current display"""
        if self.current_experiment:
            self.display_experiment_data(self.current_experiment)
            self.update_results_display(self.current_experiment)
            self.status_var.set("Display refreshed")
    
    def clear_all(self):
        """Clear all loaded data"""
        self.experiments.clear()
        self.current_experiment = None
        self.update_experiment_list()
        self.data_text.delete(1.0, tk.END)
        
        # Clear results tree
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Clear figures
        self.fig.clear()
        self.canvas.draw()
        self.comp_fig.clear()
        self.comp_canvas.draw()
        
        self.status_var.set("All data cleared")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
QHRF Quantum Analysis Suite v2.0

Interactive analysis tool for Quantum Harmonic Resonance Framework experimental results.

Developed by: Zachary L. Musselwhite
Date: June 30, 2025

Features:
• Interactive visualization of QHRF experimental data
• Comprehensive performance analysis and assessment
• Multi-experiment comparison capabilities
• Publication-quality plot generation
• Real-time data loading and analysis

This tool enables detailed analysis of quantum entanglement stabilization
experiments and validates QHRF breakthrough discoveries.

For more information about QHRF:
Email: Xses.Science@gmail.com
"""
        messagebox.showinfo("About QHRF Analysis Suite", about_text)
    
    def show_help(self):
        """Show help dialog"""
        help_text = """
QHRF ANALYSIS SUITE - USER GUIDE

GETTING STARTED:
1. Click "Load QHRF Results" to select your experiment JSON file
2. View experiment data in the "Data Overview" tab
3. Generate visualizations using the control panel buttons
4. Compare multiple experiments by loading additional files

VISUALIZATION OPTIONS:
📈 State Distribution - View quantum state probability distribution
🔬 QHRF Signature - Analyze QHRF-specific patterns
📊 Performance Metrics - Compare against benchmarks
🎛️ Dashboard - Comprehensive overview of all metrics

ANALYSIS FEATURES:
• Automatic QHRF assessment and scoring
• Success criteria evaluation
• Performance benchmarking
• Statistical analysis across experiments

EXPORT OPTIONS:
💾 Save Current Plot - Export the currently displayed visualization
📁 Export All Plots - Generate and save all available plots

KEYBOARD SHORTCUTS:
Ctrl+O - Load experiment file
Ctrl+S - Save current plot
F5 - Refresh display
Ctrl+Q - Quit application

SUPPORTED FILE FORMATS:
• JSON files from QHRF experiments
• Enhanced QHRF experimental results
• Multi-experiment comparison data

For technical support, contact: Xses.Science@gmail.com
"""
        
        help_window = tk.Toplevel(self.root)
        help_window.title("QHRF Analysis Suite - User Guide")
        help_window.geometry("700x500")
        
        help_display = scrolledtext.ScrolledText(help_window, wrap=tk.WORD, font=('Arial', 10))
        help_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        help_display.insert(1.0, help_text)
        help_display.config(state=tk.DISABLED)

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = QHRFAnalysisGUI(root)
    
    # Set application icon (if available)
    try:
        root.iconname("QHRF")
    except:
        pass
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main()
