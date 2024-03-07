import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from nsepython import nsefetch
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense

class StockPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Prediction App")

        # Fetch all symbols from NSE website
        self.stock_symbols = self.get_stock_symbols()

        self.stock_symbol = tk.StringVar()
        self.stock_symbol.set(self.stock_symbols[0] if self.stock_symbols else "No Data")

        ttk.Label(root, text="Search Stock Symbol:").grid(row=0, column=0, padx=10, pady=10)
        self.search_entry = ttk.Entry(root)
        self.search_entry.grid(row=0, column=1, padx=10, pady=10)
        self.search_entry.bind('<KeyRelease>', self.filter_symbols)

        ttk.Label(root, text="Select Stock Symbol:").grid(row=1, column=0, padx=10, pady=10)
        self.symbol_dropdown = ttk.Combobox(root, textvariable=self.stock_symbol)
        self.symbol_dropdown.grid(row=1, column=1, padx=10, pady=10)
        self.symbol_dropdown['values'] = self.stock_symbols

        self.predict_button = ttk.Button(root, text="Predict", command=self.predict_stock)
        self.predict_button.grid(row=1, column=2, padx=10, pady=10)

        self.output_text = tk.Text(root, height=10, width=50)
        self.output_text.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

    def get_stock_symbols(self):
        try:
            result = nsefetch('https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O')
            symbols = [data['symbol'] for data in result['data']]
            return symbols
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch stock symbols: {e}")
            return []

    def filter_symbols(self, event=None):
        query = self.search_entry.get().strip().upper()
        filtered_symbols = [symbol for symbol in self.stock_symbols if query in symbol]
        self.symbol_dropdown['values'] = filtered_symbols

    def predict_stock(self):
        try:
            stock_symbol = self.stock_symbol.get()

            # Call respective function based on the button clicked
            if self.root.focus_get() == self.predict_button:
                self.predict(stock_symbol)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def predict(self, stock_symbol):
        # Call respective function based on the button clicked
        self.fetch_data(stock_symbol)
        self.process_data()
        self.build_model()
        self.make_predictions()
        self.compare_with_current_price(stock_symbol)

    def fetch_data(self, stock_symbol):
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)

            # Fetch historical data using the NSE API
            self.historical_data = nsefetch(f'https://www.nseindia.com/api/historical/cm/equity?symbol={stock_symbol}&series=[%22EQ%22]&from={start_date.strftime("%d-%m-%Y")}&to={end_date.strftime("%d-%m-%Y")}')
        except Exception as e:
            raise Exception(f"Error fetching data: {e}")

    def process_data(self):
        try:
            historical_values = [entry['CH_OPENING_PRICE'] for entry in self.historical_data['data']]

            self.combined_data = np.array(historical_values)
            self.scaler = MinMaxScaler()
            self.scaled_data = self.scaler.fit_transform(self.combined_data.reshape(-1, 1))
        except Exception as e:
            raise Exception(f"Error processing data: {e}")

    def build_model(self):
        try:
            look_back = 10
            X = []
            y = []

            for i in range(len(self.scaled_data) - look_back):
                X.append(self.scaled_data[i:i + look_back])
                y.append(self.scaled_data[i + look_back])

            X = np.array(X)
            y = np.array(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            self.X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            self.model = Sequential()
            self.model.add(LSTM(50, input_shape=(look_back, 1)))
            self.model.add(Dense(1))
            self.model.compile(loss='mean_squared_error', optimizer='adam')
            self.model.fit(self.X_train, y_train, epochs=10, batch_size=1, verbose=0)
        except Exception as e:
            raise Exception(f"Error building model: {e}")

    def make_predictions(self):
        try:
            future_input = self.scaled_data[-10:].reshape(1, 10, 1)
            future_price_scaled = self.model.predict(future_input)
            future_price = self.scaler.inverse_transform(future_price_scaled)

            # Get the date of the predicted price
            predicted_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

            self.output_text.insert(tk.END, f"Predicted Future Price ({predicted_date}): {future_price[0][0]}\n")
        except Exception as e:
            raise Exception(f"Error making predictions: {e}")

    def compare_with_current_price(self, stock_symbol):
        try:
            # Fetch current price using the NSE API
            current_price_data = nsefetch(f"https://www.nseindia.com/api/quote-equity?symbol={stock_symbol}")
            current_price = current_price_data['priceInfo']['lastPrice']

            # Fetch the predicted future price
            future_price = self.get_predicted_future_price()

            # Calculate percentage accuracy
            if (future_price > current_price ):
                accuracy = (current_price / future_price) * 100
            accuracy = (future_price / current_price ) * 100

            # Display current price, predicted price, and percentage accuracy
            self.output_text.insert(tk.END, f"Current Price: {current_price}\n")
            self.output_text.insert(tk.END, f"Next Day Predicted Price: {future_price}\n")
            # self.output_text.insert(tk.END, f"Percentage Accuracy: {accuracy:.2f}%\n")
        except Exception as e:
            raise Exception(f"Error comparing with current price: {e}")

    def get_predicted_future_price(self):
        try:
            future_input = self.scaled_data[-10:].reshape(1, 10, 1)
            future_price_scaled = self.model.predict(future_input)
            future_price = self.scaler.inverse_transform(future_price_scaled)
            return future_price[0][0]
        except Exception as e:
            raise Exception(f"Error getting predicted future price: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictionApp(root)
    root.mainloop()



