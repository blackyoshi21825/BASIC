import tkinter as tk
from tkinter import ttk, messagebox
import threading
from basic_core import StockPredictor

class StockPredictorPanel:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Stock Predictor Panel")
        self.root.geometry("500x400")
        self.root.configure(bg='#2b2b2b')
        
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title = tk.Label(self.root, text="BASIC", 
                        font=('Arial', 16, 'bold'), fg='white', bg='#2b2b2b')
        title.pack(pady=10)
        
        # Input frame
        input_frame = tk.Frame(self.root, bg='#2b2b2b')
        input_frame.pack(pady=10)
        
        # Symbol input
        tk.Label(input_frame, text="Stock Symbol:", font=('Arial', 10), 
                fg='white', bg='#2b2b2b').pack(side=tk.LEFT)
        
        self.symbol_entry = tk.Entry(input_frame, font=('Arial', 10), width=10)
        self.symbol_entry.pack(side=tk.LEFT, padx=5)
        

        
        self.predict_btn = tk.Button(input_frame, text="Predict", 
                                   command=self.predict_stock, bg='#4CAF50', 
                                   fg='white', font=('Arial', 10, 'bold'))
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Results frame
        self.results_frame = tk.Frame(self.root, bg='#3b3b3b', relief='raised', bd=2)
        self.results_frame.pack(pady=20, padx=20, fill='both', expand=True)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Enter a stock symbol and click Predict", 
                                   fg='gray', bg='#2b2b2b', font=('Arial', 9))
        self.status_label.pack(pady=5)
        
    def create_result_display(self, result, symbol):
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
            
        # Title
        tk.Label(self.results_frame, text=f"Current Day Analysis for {symbol}", 
                font=('Arial', 14, 'bold'), fg='white', bg='#3b3b3b').pack(pady=10)
        
        # Current price
        tk.Label(self.results_frame, text=f"Current Price: ${result['current_price']:.2f}", 
                font=('Arial', 12), fg='lightblue', bg='#3b3b3b').pack(pady=2)
        
        # Predicted change
        change_color = 'lightgreen' if result['predicted_change_pct'] > 0 else 'lightcoral'
        tk.Label(self.results_frame, text=f"Predicted Change: {result['predicted_change_pct']:.2f}%", 
                font=('Arial', 12), fg=change_color, bg='#3b3b3b').pack(pady=2)
        
        # Signal with color coding
        signal_text = {1: "BUY ↑", 0: "HOLD →", -1: "SELL ↓"}
        signal_color = {1: 'lightgreen', 0: 'yellow', -1: 'lightcoral'}
        
        signal_label = tk.Label(self.results_frame, 
                               text=f"Signal: {signal_text[result['prediction_signal']]}", 
                               font=('Arial', 14, 'bold'), 
                               fg=signal_color[result['prediction_signal']], 
                               bg='#3b3b3b')
        signal_label.pack(pady=5)
        
        # Recommendation
        rec_color = {'STRONG BUY': 'lime', 'BUY': 'lightgreen', 'HOLD': 'yellow', 
                    'SELL': 'orange', 'STRONG SELL': 'red'}
        
        tk.Label(self.results_frame, text=f"Recommendation: {result['recommendation']}", 
                font=('Arial', 12, 'bold'), 
                fg=rec_color.get(result['recommendation'], 'white'), 
                bg='#3b3b3b').pack(pady=2)
        
        # Additional info
        tk.Label(self.results_frame, text=f"Confidence: {result['confidence']:.2f}%", 
                font=('Arial', 10), fg='lightgray', bg='#3b3b3b').pack(pady=1)
        
        tk.Label(self.results_frame, text=f"News Sentiment: {result['sentiment']:.3f}", 
                font=('Arial', 10), fg='lightgray', bg='#3b3b3b').pack(pady=1)
        
        tk.Label(self.results_frame, text=f"Model Accuracy: {result['model_accuracy']}", 
                font=('Arial', 10), fg='lightgray', bg='#3b3b3b').pack(pady=1)
        
        # Chart analysis if available
        if 'chart_analysis' in result:
            chart = result['chart_analysis']
            tk.Label(self.results_frame, text="\n=== Chart Analysis ===", 
                    font=('Arial', 11, 'bold'), fg='cyan', bg='#3b3b3b').pack(pady=2)
            
            tk.Label(self.results_frame, text=f"{chart.get('support_resistance', 'N/A')}", 
                    font=('Arial', 9), fg='lightgray', bg='#3b3b3b').pack(pady=1)
            
            tk.Label(self.results_frame, text=f"{chart.get('swing_direction', 'N/A')}", 
                    font=('Arial', 9), fg='lightgray', bg='#3b3b3b').pack(pady=1)
            
            tk.Label(self.results_frame, text=f"Pattern: {chart.get('pattern_detected', 'None')}", 
                    font=('Arial', 9), fg='lightgray', bg='#3b3b3b').pack(pady=1)
        
    def predict_stock(self):
        symbol = self.symbol_entry.get().upper().strip()
        if not symbol:
            messagebox.showerror("Error", "Please enter a stock symbol")
            return
            
        # Disable button and show loading
        self.predict_btn.config(state='disabled', text='Loading...')
        self.status_label.config(text=f"Analyzing {symbol}...")
        
        # Run prediction in separate thread
        thread = threading.Thread(target=self.run_prediction, args=(symbol,))
        thread.daemon = True
        thread.start()
        
    def run_prediction(self, symbol):
        try:
            predictor = StockPredictor(symbol)
            result = predictor.predict_current_day()
            
            # Update UI in main thread
            self.root.after(0, self.update_results, result, symbol)
            
        except Exception as e:
            self.root.after(0, self.show_error, str(e))
            
    def update_results(self, result, symbol):
        self.create_result_display(result, symbol)
        self.predict_btn.config(state='normal', text='Predict')
        self.status_label.config(text="Prediction complete!")
        
    def show_error(self, error_msg):
        self.predict_btn.config(state='normal', text='Predict')
        self.status_label.config(text="Error occurred")
        messagebox.showerror("Error", f"Prediction failed: {error_msg}")

def main():
    root = tk.Tk()
    app = StockPredictorPanel(root)
    root.mainloop()

if __name__ == "__main__":
    main()